from dataclasses import dataclass
from typing import Any, List, Optional, Sequence

import math
import os
import shutil

import yaml
import coolname
import hydra
import pydantic
import torch
import torch.distributed as dist
import wandb
from adam_atan2_pytorch import AdamAtan2
from omegaconf import DictConfig
from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from torch import nn
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import DataLoader

from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig, PuzzleDatasetMetadata
from utils.functions import load_model_class, get_model_source_path
from models.layers import CastedSparseEmbeddingSignSGD_Distributed

try:
    # Newer API: averaging function that jointly handles parameters and buffers.
    from torch.optim.swa_utils import get_ema_multi_avg_fn as _get_ema_avg_fn

    _EMA_FN_KIND = "multi"
except Exception:
    try:
        # Fallback API: averaging function for a single parameter tensor.
        from torch.optim.swa_utils import get_ema_avg_fn as _get_ema_avg_fn

        _EMA_FN_KIND = "single"
    except Exception:
        # Very old PyTorch versions: no built-in EMA averaging helper.
        _get_ema_avg_fn = None
        _EMA_FN_KIND = "none"


class LossConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="allow")
    name: str


class ArchConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="allow")
    name: str
    loss: LossConfig


class PretrainConfig(pydantic.BaseModel):
    # Architecture.
    arch: ArchConfig

    # Data.
    data_path: str

    # Core hyperparameters.
    global_batch_size: int
    epochs: int

    lr: float
    lr_min_ratio: float
    lr_warmup_steps: int

    weight_decay: float
    beta1: float
    beta2: float

    # Puzzle embedding optimization.
    puzzle_emb_lr: float
    puzzle_emb_weight_decay: float

    # Naming and checkpointing.
    project_name: Optional[str] = None
    run_name: Optional[str] = None
    checkpoint_path: Optional[str] = None

    # Misc.
    seed: int = 0
    checkpoint_every_eval: bool = False
    eval_interval: Optional[int] = None
    eval_save_outputs: List[str] = []

    wandb_key: Optional[str] = None

    # EMA hyperparameters.
    # EMA decay factor. Values closer to 1.0 produce smoother updates.
    ema_decay: float
    # Whether to enable EMA during training.
    ema_enabled: bool
    # Device on which to keep the EMA model (e.g., "cuda" or "cpu").
    ema_device: str
    # Whether EMA should also track buffers (required for puzzle_emb stored in buffers).
    ema_use_buffers: bool


@dataclass
class TrainState:
    model: nn.Module
    optimizers: Sequence[torch.optim.Optimizer]
    optimizer_lrs: Sequence[float]
    carry: Any

    step: int
    total_steps: int

    # Optional EMA model used primarily for evaluation.
    ema_model: Optional[nn.Module] = None


def create_dataloader(config: PretrainConfig, split: str, rank: int, world_size: int, **kwargs):
    dataset = PuzzleDataset(
        PuzzleDatasetConfig(
            seed=config.seed,
            dataset_path=config.data_path,
            rank=rank,
            num_replicas=world_size,
            **kwargs,
        ),
        split=split,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=1,
        prefetch_factor=8,
        pin_memory=True,
        persistent_workers=True,
    )
    return dataloader, dataset.metadata


def _build_ema_model_if_needed(model: nn.Module, config: PretrainConfig) -> Optional[AveragedModel]:
    """
    Create and return an EMA model (AveragedModel). If EMA is disabled, return None.

    Notes:
        - AveragedModel clones the given model and applies the provided EMA averaging function.
        - use_buffers=True ensures that buffers (for example batch-norm running stats or puzzle_emb
          buffers in this project) are updated as well.
    """
    if not config.ema_enabled:
        return None

    device = torch.device(config.ema_device) if config.ema_device is not None else None
    decay = float(config.ema_decay)

    # Build EMA model in a way that remains compatible with multiple PyTorch versions.
    if _get_ema_avg_fn is not None:
        avg_or_multi_avg_fn = _get_ema_avg_fn(decay)
        try:
            if _EMA_FN_KIND == "multi":
                # Newer interface: multi_avg_fn handles parameters and buffers together.
                ema_model = AveragedModel(
                    model,
                    device=device,
                    multi_avg_fn=avg_or_multi_avg_fn,
                    use_buffers=config.ema_use_buffers,
                )
            else:
                # Older interface: avg_fn handles a single parameter tensor at a time.
                ema_model = AveragedModel(
                    model,
                    device=device,
                    avg_fn=avg_or_multi_avg_fn,
                    use_buffers=config.ema_use_buffers,
                )
            return ema_model
        except TypeError:
            # Some versions do not support the use_buffers argument.
            ema_model = AveragedModel(
                model,
                device=device,
                avg_fn=avg_or_multi_avg_fn if _EMA_FN_KIND != "multi" else None,
            )
            return ema_model

    # Fallback for very old versions without get_ema_* helpers.
    def _ema_avg_fallback(averaged_param, current_param, num_averaged):
        # W_ema = decay * W_ema + (1 - decay) * W
        return averaged_param.mul(decay).add_(current_param, alpha=(1.0 - decay))

    try:
        ema_model = AveragedModel(
            model,
            device=device,
            avg_fn=_ema_avg_fallback,
            use_buffers=config.ema_use_buffers,
        )
    except TypeError:
        # Very early versions do not support use_buffers.
        ema_model = AveragedModel(
            model,
            device=device,
            avg_fn=_ema_avg_fallback,
        )
    return ema_model


def create_model(config: PretrainConfig, train_metadata: PuzzleDatasetMetadata, world_size: int):
    model_cfg = dict(
        **config.arch.__pydantic_extra__,  # type: ignore
        batch_size=config.global_batch_size // world_size,
        vocab_size=train_metadata.vocab_size,
        seq_len=train_metadata.seq_len,
        num_puzzle_identifiers=train_metadata.num_puzzle_identifiers,
        causal=False,  # Non-autoregressive.
    )

    # Instantiate model with loss head.
    model_cls = load_model_class(config.arch.name)
    loss_head_cls = load_model_class(config.arch.loss.name)

    with torch.device("cuda"):
        # Build the uncompiled model first.
        model: nn.Module = model_cls(model_cfg)
        model = loss_head_cls(model, **config.arch.loss.__pydantic_extra__)  # type: ignore

        # Broadcast initial parameters so that all ranks start from the same state.
        if world_size > 1:
            with torch.no_grad():
                for param in list(model.parameters()) + list(model.buffers()):
                    dist.broadcast(param, src=0)

        # Build an EMA model from the uncompiled model to avoid compatibility issues
        # with compiled modules. The EMA model remains uncompiled for stable evaluation.
        ema_model: Optional[AveragedModel] = _build_ema_model_if_needed(model, config)

        # Optionally compile the training model.
        if "DISABLE_COMPILE" not in os.environ:
            model = torch.compile(model, dynamic=True)  # type: ignore

    # Optimizers and learning-rate placeholders.
    optimizers = [
        CastedSparseEmbeddingSignSGD_Distributed(
            model.model.puzzle_emb.buffers(),  # type: ignore
            lr=0,  # Will be set by the scheduler.
            weight_decay=config.puzzle_emb_weight_decay,
            world_size=world_size,
        ),
        AdamAtan2(
            model.parameters(),
            lr=1e-7,  # Will be set by the scheduler.
            weight_decay=config.weight_decay,
            betas=(config.beta1, config.beta2),
        ),
    ]
    optimizer_lrs = [
        config.puzzle_emb_lr,
        config.lr,
    ]

    return model, optimizers, optimizer_lrs, ema_model


def cosine_schedule_with_warmup_lr_lambda(
    current_step: int,
    *,
    base_lr: float,
    num_warmup_steps: int,
    num_training_steps: int,
    min_ratio: float = 0.0,
    num_cycles: float = 0.5,
):
    if current_step < num_warmup_steps:
        return base_lr * float(current_step) / float(max(1, num_warmup_steps))

    progress = float(current_step - num_warmup_steps) / float(
        max(1, num_training_steps - num_warmup_steps)
    )
    return base_lr * (
        min_ratio
        + max(
            0.0,
            (1 - min_ratio)
            * 0.5
            * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)),
        )
    )


def init_train_state(config: PretrainConfig, train_metadata: PuzzleDatasetMetadata, world_size: int):
    # Estimate the total number of training steps.
    total_steps = int(
        config.epochs
        * train_metadata.total_groups
        * train_metadata.mean_puzzle_examples
        / config.global_batch_size
    )

    # Build model and optimizers.
    model, optimizers, optimizer_lrs, ema_model = create_model(
        config, train_metadata, world_size=world_size
    )

    return TrainState(
        step=0,
        total_steps=total_steps,
        model=model,
        optimizers=optimizers,
        optimizer_lrs=optimizer_lrs,
        carry=None,
        ema_model=ema_model,
    )


def save_train_state(config: PretrainConfig, train_state: TrainState):
    # Persist the main model and the EMA model (if available).
    if config.checkpoint_path is None:
        return

    os.makedirs(config.checkpoint_path, exist_ok=True)
    save_obj = {"model": train_state.model.state_dict()}
    if train_state.ema_model is not None:
        save_obj["model_ema"] = train_state.ema_model.state_dict()
    torch.save(save_obj, os.path.join(config.checkpoint_path, f"step_{train_state.step}"))


def compute_lr(base_lr: float, config: PretrainConfig, train_state: TrainState):
    return cosine_schedule_with_warmup_lr_lambda(
        current_step=train_state.step,
        base_lr=base_lr,
        num_warmup_steps=round(config.lr_warmup_steps),
        num_training_steps=train_state.total_steps,
        min_ratio=config.lr_min_ratio,
    )


def train_batch(
    config: PretrainConfig,
    train_state: TrainState,
    batch: Any,
    global_batch_size: int,
    rank: int,
    world_size: int,
    *,
    recurrent_steps_m: int,
):
    train_state.step += 1
    if train_state.step > train_state.total_steps:
        return

    # Move batch to GPU.
    batch = {k: v.cuda() for k, v in batch.items()}

    # Initialize recurrent state if it does not exist yet.
    if train_state.carry is None:
        with torch.device("cuda"):
            train_state.carry = train_state.model.initial_carry(batch)  # type: ignore

    # Recurrent inner loop.
    metrics = {}
    loss = None
    for t in range(recurrent_steps_m):
        is_last = t == recurrent_steps_m - 1

        # Only compute loss and perform a backward pass on the last recurrent step.
        train_state.carry, loss_this, metrics_this, _, _ = train_state.model(
            carry=train_state.carry,
            batch=batch,
            return_keys=[],
            compute_loss=is_last,
            force_halt=is_last,
        )

        if is_last:
            loss = loss_this
            metrics = metrics_this

    # Single backward pass for the normalized loss.
    ((1 / global_batch_size) * loss).backward()  # type: ignore

    # Gradient all-reduce across devices.
    if world_size > 1:
        for param in train_state.model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad)

    # Optimizer step with per-step learning-rate computation.
    lr_this_step = None
    for optim, base_lr in zip(train_state.optimizers, train_state.optimizer_lrs):
        lr_this_step = compute_lr(base_lr, config, train_state)
        for param_group in optim.param_groups:
            param_group["lr"] = lr_this_step
        optim.step()
        optim.zero_grad()

    # Update EMA parameters after the main model has been updated.
    if train_state.carry.halted.all().item():
        if train_state.ema_model is not None:
            train_state.ema_model.update_parameters(train_state.model)

    # Only log metrics once the recurrent computation has halted.
    if not train_state.carry.halted.all().item():
        return None

    # Reduce metrics across devices and convert them to scalars on rank 0.
    if len(metrics):
        assert not any(v.requires_grad for v in metrics.values())
        metric_keys = list(sorted(metrics.keys()))
        metric_values = torch.stack([metrics[k] for k in metric_keys])
        if world_size > 1:
            dist.reduce(metric_values, dst=0)
        if rank == 0:
            metric_values = metric_values.cpu().numpy()
            reduced_metrics = {k: metric_values[i] for i, k in enumerate(metric_keys)}

            count = max(reduced_metrics["count"], 1)
            reduced_metrics = {
                f"train/{k}": v / (global_batch_size if k.endswith("loss") else count)
                for k, v in reduced_metrics.items()
            }

            reduced_metrics["train/lr"] = lr_this_step
            return reduced_metrics


def evaluate(
    config: PretrainConfig,
    train_state: TrainState,
    eval_loader: torch.utils.data.DataLoader,
    eval_metadata: PuzzleDatasetMetadata,
    rank: int,
    world_size: int,
    recurrent_steps_m: int,
):
    # Prefer EMA weights for evaluation when available; otherwise fall back to the training model.
    # Evaluation uses a local recurrent state to avoid mutating train_state.carry.
    with torch.inference_mode():
        if isinstance(train_state.ema_model, AveragedModel):
            model_for_eval: nn.Module = train_state.ema_model.module
        elif train_state.ema_model is not None:
            model_for_eval = train_state.ema_model
        else:
            model_for_eval = train_state.model
        model_for_eval.eval()

        set_ids = {k: idx for idx, k in enumerate(eval_metadata.sets)}
        all_preds = {}

        metric_keys = []
        metric_values = None
        metric_global_batch_size = [0 for _ in range(len(set_ids))]

        for set_name, batch, global_batch_size in eval_loader:
            # Move batch to GPU.
            batch = {k: v.cuda() for k, v in batch.items()}

            # Forward until all sequences finish.
            while True:
                metrics = {}
                loss = None
                for t in range(recurrent_steps_m):
                    is_last = t == recurrent_steps_m - 1

                    train_state.carry, loss_this, metrics_this, preds, all_finish = model_for_eval(
                        carry=train_state.carry,
                        batch=batch,
                        return_keys=[],
                        compute_loss=is_last,
                        force_halt=is_last,
                    )

                    if is_last:
                        carry, loss, metrics, preds, all_finish = (
                            train_state.carry,
                            loss_this,
                            metrics_this,
                            preds,
                            all_finish,
                        )

                if all_finish:
                    break

            for collection in (batch, preds):
                for k, v in collection.items():
                    if k in config.eval_save_outputs:
                        all_preds.setdefault(k, [])
                        # Move tensors to CPU so evaluation does not consume additional GPU memory.
                        all_preds[k].append(v.cpu())

            del carry, loss, preds, batch, all_finish

            # Aggregate metrics.
            set_id = set_ids[set_name]

            if metric_values is None:
                # Sort keys to ensure all processes use the same metric ordering.
                metric_keys = list(sorted(metrics.keys()))
                metric_values = torch.zeros(
                    (len(set_ids), len(metrics.values())),
                    dtype=torch.float32,
                    device="cuda",
                )

            metric_values[set_id] += torch.stack([metrics[k] for k in metric_keys])
            metric_global_batch_size[set_id] += global_batch_size

        if len(all_preds) and config.checkpoint_path is not None:
            all_preds = {k: torch.cat(v, dim=0) for k, v in all_preds.items()}
            os.makedirs(config.checkpoint_path, exist_ok=True)
            torch.save(
                all_preds,
                os.path.join(
                    config.checkpoint_path,
                    f"step_{train_state.step}_all_preds.{rank}",
                ),
            )

        # Reduce metrics to rank 0 for logging.
        if metric_values is not None:
            if world_size > 1:
                dist.reduce(metric_values, dst=0)

            if rank == 0:
                reduced_metrics = metric_values.cpu().numpy()
                reduced_metrics = {
                    set_name: {
                        metric_name: reduced_metrics[set_id, metric_id]
                        for metric_id, metric_name in enumerate(metric_keys)
                    }
                    for set_id, set_name in enumerate(set_ids)
                }

                # Normalize by sample count.
                for set_name, metrics_dict in reduced_metrics.items():
                    count = metrics_dict.pop("count")
                    reduced_metrics[set_name] = {
                        k: v / count for k, v in metrics_dict.items()
                    }

                return reduced_metrics


def save_code_and_config(config: PretrainConfig):
    if config.checkpoint_path is None or wandb.run is None:
        return

    os.makedirs(config.checkpoint_path, exist_ok=True)

    # Copy source files into the checkpoint directory.
    current_script = os.path.realpath(__file__)
    code_list = [
        current_script,
        get_model_source_path(config.arch.name),
        get_model_source_path(config.arch.loss.name),
    ]
    for code_file in code_list:
        if code_file is not None:
            code_name = os.path.basename(code_file)
            shutil.copy(code_file, os.path.join(config.checkpoint_path, code_name))

    # Dump configuration as YAML.
    config_file = os.path.join(config.checkpoint_path, "all_config.yaml")
    with open(config_file, "wt") as f:
        yaml.dump(config.model_dump(), f)

    # Log code snapshot to Weights & Biases.
    wandb.run.log_code(config.checkpoint_path)


def load_synced_config(hydra_config: DictConfig, rank: int, world_size: int) -> PretrainConfig:
    objects = [None]
    if rank == 0:
        config = PretrainConfig(**hydra_config)  # type: ignore

        # Fill default naming fields.
        if config.project_name is None:
            config.project_name = f"SR2-{os.path.basename(config.data_path).capitalize()}"
        if config.run_name is None:
            config.run_name = f"{config.arch.name.split('@')[-1]} {coolname.generate_slug(2)}"
        if config.checkpoint_path is None:
            config.checkpoint_path = os.path.join(
                "checkpoints", config.project_name, config.run_name
            )

        objects = [config]

    if world_size > 1:
        dist.broadcast_object_list(objects, src=0)

    return objects[0]  # type: ignore


@hydra.main(config_path="config", config_name="cfg_pretrain", version_base=None)
def launch(hydra_config: DictConfig):
    RANK = 0
    WORLD_SIZE = 1

    # Initialize distributed training if running under a distributed launcher (e.g., torchrun).
    if "LOCAL_RANK" in os.environ:
        dist.init_process_group(backend="nccl")

        RANK = dist.get_rank()
        WORLD_SIZE = dist.get_world_size()

        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

    # Load synchronized configuration across processes.
    config = load_synced_config(hydra_config, rank=RANK, world_size=WORLD_SIZE)

    # Seed random number generators for reproducibility across ranks.
    torch.random.manual_seed(config.seed + RANK)

    # Dataset and evaluation scheduling.
    train_epochs_per_iter = (
        config.eval_interval if config.eval_interval is not None else config.epochs
    )
    total_iters = config.epochs // train_epochs_per_iter

    assert (
        config.epochs % train_epochs_per_iter == 0
    ), "Eval interval must be a divisor of total epochs."

    train_loader, train_metadata = create_dataloader(
        config,
        "train",
        test_set_mode=False,
        epochs_per_iter=train_epochs_per_iter,
        global_batch_size=config.global_batch_size,
        rank=RANK,
        world_size=WORLD_SIZE,
    )
    eval_loader, eval_metadata = create_dataloader(
        config,
        "test",
        test_set_mode=True,
        epochs_per_iter=1,
        global_batch_size=config.global_batch_size,
        rank=RANK,
        world_size=WORLD_SIZE,
    )

    # Initialize training state.
    train_state = init_train_state(config, train_metadata, world_size=WORLD_SIZE)

    # Progress bar and logging setup.
    progress = None
    progress_task_id = None
    if RANK == 0:
        console = Console()
        progress = Progress(
            TextColumn("[bold blue]Training"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn(" • Step {task.completed}/{task.total}"),
            TextColumn(" • [bold red]lm_loss[/]: [red]{task.fields[lm_loss]}[/]"),
            TextColumn(" • [bold green]accuracy[/]: [green]{task.fields[accuracy]}[/]"),
            TextColumn(" • [bold magenta]exact_accuracy[/]: [magenta]{task.fields[exact_accuracy]}[/]"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
        )
        progress.start()
        progress_task_id = progress.add_task(
            "training",
            total=train_state.total_steps,
            lm_loss="N/A",
            accuracy="N/A",
            exact_accuracy="N/A",
        )

        wandb.login(key=config.wandb_key, relogin=False)
        wandb.init(
            project=config.project_name,
            name=config.run_name,
            config=config.model_dump(),
            settings=wandb.Settings(_disable_stats=True),
        )  # type: ignore
        wandb.log(
            {"num_params": sum(x.numel() for x in train_state.model.parameters())},
            step=0,
        )
        save_code_and_config(config)

    # Training loop.
    for _iter_id in range(total_iters):
        # Training iteration.
        train_state.model.train()
        for set_name, batch, global_batch_size in train_loader:
            metrics = train_batch(
                config,
                train_state,
                batch,
                global_batch_size,
                rank=RANK,
                world_size=WORLD_SIZE,
                recurrent_steps_m=config.arch.recurrent_steps_m,
            )

            if RANK == 0 and metrics is not None:
                wandb.log(metrics, step=train_state.step)

                if progress is not None and progress_task_id is not None:
                    loss_for_bar = metrics.get("train/lm_loss")
                    acc_for_bar = metrics.get("train/accuracy")
                    exact_acc_for_bar = metrics.get("train/exact_accuracy")

                    loss_str = (
                        f"{float(loss_for_bar):.4f}"
                        if loss_for_bar is not None
                        else "N/A"
                    )
                    acc_str = (
                        f"{float(acc_for_bar):.4f}"
                        if acc_for_bar is not None
                        else "N/A"
                    )
                    exact_acc_str = (
                        f"{float(exact_acc_for_bar):.4f}"
                        if exact_acc_for_bar is not None
                        else "N/A"
                    )
                    progress.update(
                        progress_task_id,
                        completed=train_state.step,
                        lm_loss=loss_str,
                        accuracy=acc_str,
                        exact_accuracy=exact_acc_str,
                    )

        # Evaluation (prefer EMA weights when available).
        train_state.model.eval()
        metrics = evaluate(
            config,
            train_state,
            eval_loader,
            eval_metadata,
            rank=RANK,
            world_size=WORLD_SIZE,
            recurrent_steps_m=config.arch.recurrent_steps_m,
        )

        if RANK == 0 and metrics is not None:
            wandb.log(metrics, step=train_state.step)

        # Checkpointing (main model and EMA model).
        if RANK == 0 and (
            config.checkpoint_every_eval or (_iter_id == total_iters - 1)
        ):
            save_train_state(config, train_state)

    # Finalize progress and distributed resources.
    if RANK == 0 and progress is not None:
        progress.stop()

    if dist.is_initialized():
        dist.destroy_process_group()
    wandb.finish()



if __name__ == "__main__":
    launch()
