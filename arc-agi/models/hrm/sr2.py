from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from torch import nn
from pydantic import BaseModel

from models.common import trunc_normal_init_
from models.layers import rms_norm, SwiGLU, Attention, RotaryEmbedding, CosSin, CastedEmbedding, CastedLinear
from models.sparse_embedding import CastedSparseEmbedding


@dataclass
class HierarchicalReasoningModel_ACTV3InnerCarry:
    z_H: torch.Tensor


@dataclass
class HierarchicalReasoningModel_ACTV3Carry:
    inner_carry: HierarchicalReasoningModel_ACTV3InnerCarry

    steps: torch.Tensor
    halted: torch.Tensor

    current_data: Dict[str, torch.Tensor]


class HierarchicalReasoningModel_ACTV3Config(BaseModel):
    batch_size: int
    seq_len: int
    puzzle_emb_ndim: int = 0
    num_puzzle_identifiers: int
    vocab_size: int

    H_cycles: int

    H_layers: int

    # Transformer config
    hidden_size: int
    expansion: float
    num_heads: int
    pos_encodings: str

    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0

    # Halting Q-learning config
    halt_max_steps: int
    halt_exploration_prob: float
    act_enabled: bool = True  # If False, always run halt_max_steps (no early stopping during training)
    act_inference: bool = False  # If True, use adaptive computation during inference

    forward_dtype: str = "bfloat16"


class HierarchicalReasoningModel_ACTV3Block(nn.Module):
    def __init__(self, config: HierarchicalReasoningModel_ACTV3Config) -> None:
        super().__init__()

        self.self_attn = Attention(
            hidden_size=config.hidden_size,
            head_dim=config.hidden_size // config.num_heads,
            num_heads=config.num_heads,
            num_key_value_heads=config.num_heads,
            causal=False,
        )
        self.mlp = SwiGLU(
            hidden_size=config.hidden_size,
            expansion=config.expansion,
        )
        self.norm_eps = config.rms_norm_eps

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        # Post Norm
        # Self Attention
        hidden_states = rms_norm(
            hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states),
            variance_epsilon=self.norm_eps,
        )
        # Fully Connected
        hidden_states = rms_norm(hidden_states + self.mlp(hidden_states), variance_epsilon=self.norm_eps)
        return hidden_states

    
class HierarchicalReasoningModel_ACTV3ReasoningModule(nn.Module):
    """
    Notes:
    - Previously, multiple layers with independent parameters were constructed by passing in a List[Block].
    - Now, only a single shared block is kept, and the same block is repeatedly called in the forward pass via `self.repeats`.
    - To minimize impact on external code, the `self.layers` attribute is kept, but it contains only the single shared block.
    """
    def __init__(self, block: HierarchicalReasoningModel_ACTV3Block, repeats: int):
        super().__init__()
        # Register only one shared block (keep attribute name `layers` to avoid breaking external dependencies)
        self.layers = torch.nn.ModuleList([block])
        self.repeats = int(repeats)

    def forward(self, hidden_states: torch.Tensor, input_injection: torch.Tensor, **kwargs) -> torch.Tensor:
        # Input injection (add)

        # hidden_states = hidden_states + input_injection
        # Use the same block repeatedly n times (weights fully shared)
        shared_block = self.layers[0]
        for _ in range(self.repeats):
            hidden_states = shared_block(hidden_states=hidden_states + input_injection, **kwargs)
        return hidden_states


class HierarchicalReasoningModel_ACTV3_Inner(nn.Module):
    def __init__(self, config: HierarchicalReasoningModel_ACTV3Config) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, self.config.forward_dtype)

        # I/O
        self.embed_scale = math.sqrt(self.config.hidden_size)
        embed_init_std = 1.0 / self.embed_scale

        self.embed_tokens = CastedEmbedding(
            self.config.vocab_size,
            self.config.hidden_size,
            init_std=embed_init_std,
            cast_to=self.forward_dtype,
        )
        self.lm_head = CastedLinear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.q_head = CastedLinear(self.config.hidden_size, 2, bias=True)

        self.puzzle_emb_len = -(self.config.puzzle_emb_ndim // -self.config.hidden_size)  # ceil div
        if self.config.puzzle_emb_ndim > 0:
            # Zero init puzzle embeddings
            self.puzzle_emb = CastedSparseEmbedding(
                self.config.num_puzzle_identifiers,
                self.config.puzzle_emb_ndim,
                batch_size=self.config.batch_size,
                init_std=0,
                cast_to=self.forward_dtype,
            )

        # LM Blocks
        if self.config.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(
                dim=self.config.hidden_size // self.config.num_heads,
                max_position_embeddings=self.config.seq_len + self.puzzle_emb_len,
                base=self.config.rope_theta,
            )
        elif self.config.pos_encodings == "learned":
            self.embed_pos = CastedEmbedding(
                self.config.seq_len + self.puzzle_emb_len,
                self.config.hidden_size,
                init_std=embed_init_std,
                cast_to=self.forward_dtype,
            )
        else:
            raise NotImplementedError()

        
        H_block = HierarchicalReasoningModel_ACTV3Block(self.config)
        self.H_level = HierarchicalReasoningModel_ACTV3ReasoningModule(
            block=H_block,
            repeats=self.config.H_layers
        )

        # Initial states
        self.H_init = nn.Buffer(
            trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1),
            persistent=True,
        )

        # Q head special init
        # Init Q to (almost) zero for faster learning during bootstrapping
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)  # type: ignore

    def _input_embeddings(self, input: torch.Tensor, puzzle_identifiers: torch.Tensor):
        # Token embedding
        embedding = self.embed_tokens(input.to(torch.int32))

        # Puzzle embeddings
        if self.config.puzzle_emb_ndim > 0:
            puzzle_embedding = self.puzzle_emb(puzzle_identifiers)

            pad_count = self.puzzle_emb_len * self.config.hidden_size - puzzle_embedding.shape[-1]
            if pad_count > 0:
                puzzle_embedding = F.pad(puzzle_embedding, (0, pad_count))

            embedding = torch.cat(
                (puzzle_embedding.view(-1, self.puzzle_emb_len, self.config.hidden_size), embedding), dim=-2
            )

        # Position embeddings
        if self.config.pos_encodings == "learned":
            # scale by 1/sqrt(2) to maintain forward variance
            embedding = 0.707106781 * (embedding + self.embed_pos.embedding_weight.to(self.forward_dtype))

        # Scale
        return self.embed_scale * embedding

    def empty_carry(self, batch_size: int):
        return HierarchicalReasoningModel_ACTV3InnerCarry(
            z_H=torch.empty(
                batch_size,
                self.config.seq_len + self.puzzle_emb_len,
                self.config.hidden_size,
                dtype=self.forward_dtype,
            ),
        )

    def reset_carry(self, reset_flag: torch.Tensor, carry: HierarchicalReasoningModel_ACTV3InnerCarry):
        return HierarchicalReasoningModel_ACTV3InnerCarry(
            z_H=torch.where(reset_flag.view(-1, 1, 1), self.H_init, carry.z_H),
        )

    def forward(
        self, carry: HierarchicalReasoningModel_ACTV3InnerCarry, batch: Dict[str, torch.Tensor], carry_steps: torch.Tensor
    ) -> Tuple[HierarchicalReasoningModel_ACTV3InnerCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        seq_info = dict(
            cos_sin=self.rotary_emb() if hasattr(self, "rotary_emb") else None,
        )

        # Input encoding
        input_embeddings = self._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])
        
        # 仅第一步注入 input
        first_step_mask = (carry_steps == 0).view(-1, 1, 1).to(device=input_embeddings.device, dtype=input_embeddings.dtype)
        gated_injection = input_embeddings * first_step_mask  # [B,S_full,D]

        # 1-step grad
        z_H = self.H_level(carry.z_H, gated_injection, **seq_info)

        # LM Outputs
        new_carry = HierarchicalReasoningModel_ACTV3InnerCarry(
            z_H=z_H.detach(),
        )  # New carry no grad
        output = self.lm_head(z_H)[:, self.puzzle_emb_len :]

        # Q head
        q_logits = self.q_head(z_H[:, 0]).to(torch.float32)

        return new_carry, output, (q_logits[..., 0], q_logits[..., 1])


class HierarchicalReasoningModel_ACTV3(nn.Module):
    """ACT wrapper."""

    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = HierarchicalReasoningModel_ACTV3Config(**config_dict)
        self.inner = HierarchicalReasoningModel_ACTV3_Inner(self.config)

    @property
    def puzzle_emb(self):
        return self.inner.puzzle_emb

    def initial_carry(self, batch: Dict[str, torch.Tensor]):
        batch_size = batch["inputs"].shape[0]

        return HierarchicalReasoningModel_ACTV3Carry(
            inner_carry=self.inner.empty_carry(
                batch_size
            ),  # Empty is expected, it will be reseted in first pass as all sequences are halted.
            steps=torch.zeros((batch_size,), dtype=torch.int32),
            halted=torch.ones((batch_size,), dtype=torch.bool),  # Default to halted
            current_data={k: torch.empty_like(v) for k, v in batch.items()},
        )

    def forward(
        self,
        carry: HierarchicalReasoningModel_ACTV3Carry,
        batch: Dict[str, torch.Tensor],
        compute_target_q: bool = False,
    ) -> Tuple[HierarchicalReasoningModel_ACTV3Carry, Dict[str, torch.Tensor]]:
        # Update data, carry (removing halted sequences)
        new_inner_carry = self.inner.reset_carry(carry.halted, carry.inner_carry)

        new_steps = torch.where(carry.halted, 0, carry.steps)

        new_current_data = {
            k: torch.where(carry.halted.view((-1,) + (1,) * (batch[k].ndim - 1)), batch[k], v)
            for k, v in carry.current_data.items()
        }

        # Forward inner model
        new_inner_carry, logits, (q_halt_logits, q_continue_logits) = self.inner(
            new_inner_carry, new_current_data, new_steps
        )

        outputs = {"logits": logits, "q_halt_logits": q_halt_logits, "q_continue_logits": q_continue_logits}

        with torch.no_grad():
            # Step
            new_steps = new_steps + 1
            is_last_step = new_steps >= self.config.halt_max_steps

            halted = is_last_step

            # Check if adaptive computation should be used
            use_adaptive = (self.config.halt_max_steps > 1) and (
                (self.training and self.config.act_enabled)
                or (not self.training and self.config.act_inference)
            )

            if use_adaptive:
                # Halt signal based on Q-values (but always halt at max steps)
                q_halt_signal = q_halt_logits > q_continue_logits
                halted = halted | q_halt_signal

                # Store actual steps used for logging (only during inference)
                if not self.training:
                    outputs["actual_steps"] = new_steps.float()

                # Exploration (only during training)
                if self.training:
                    min_halt_steps = (
                        torch.rand_like(q_halt_logits) < self.config.halt_exploration_prob
                    ) * torch.randint_like(new_steps, low=2, high=self.config.halt_max_steps + 1)
                    halted = halted & (new_steps >= min_halt_steps)

                # Compute target Q (only during training)
                # NOTE: No replay buffer and target networks for computing target Q-value.
                # As batch_size is large, there're many parallel envs.
                # Similar concept as PQN https://arxiv.org/abs/2407.04811
                if self.training and compute_target_q:
                    next_q_halt_logits, next_q_continue_logits = self.inner(
                        new_inner_carry, new_current_data, new_steps
                    )[-1]

                    outputs["target_q_continue"] = torch.sigmoid(
                        torch.where(
                            is_last_step,
                            next_q_halt_logits,
                            torch.maximum(next_q_halt_logits, next_q_continue_logits),
                        )
                    )

        return HierarchicalReasoningModel_ACTV3Carry(
            new_inner_carry, new_steps, halted, new_current_data
        ), outputs