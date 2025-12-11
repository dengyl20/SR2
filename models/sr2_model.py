from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from torch import nn
from pydantic import BaseModel

from models.layers import rms_norm, SwiGLU, Attention, RotaryEmbedding, CosSin, CastedEmbedding, CastedLinear, trunc_normal_init_, CastedSparseEmbedding


# -------------------------------------------------------------------------------
# SR2 model overview
# -------------------------------------------------------------------------------
# This module implements the recurrent SR2 backbone used in the SR² framework.
# A shared Transformer block repeatedly refines a latent state z that combines
# token representations with puzzle-specific embeddings. An outer controller
# (SR2Model) decides how many refinement steps to perform per example (batch). 
# For the full SR² framework, including selection, reflection, self-refinement, 
# and periodic alignment, please refer to the accompanying SR² paper (Figure 3).
# -------------------------------------------------------------------------------


# -------------------------------------------------------------------------------
# Carry structures: latent state and per-example bookkeeping for Deep Supervision
# -------------------------------------------------------------------------------
@dataclass
class InnerCarry:
    # Latent state z with both token and puzzle-embedding positions:
    # shape [B, S_full, D], where S_full = seq_len + puzzle_emb_len.
    z: torch.Tensor


@dataclass
class BatchCarry:
    # Inner latent state carried across refinement steps.
    inner_carry: InnerCarry

    # [B] number of refinement steps already taken for each example.
    steps: torch.Tensor
    # [B] boolean mask indicating whether each example has halted.
    halted: torch.Tensor

    # Cached per-example tensors (e.g., inputs, puzzle identifiers) used by the inner model.
    current_data: Dict[str, torch.Tensor]


class ModelConfig(BaseModel):
    batch_size: int
    seq_len: int
    puzzle_emb_ndim: int = 0
    num_puzzle_identifiers: int
    vocab_size: int

    # Transformer configuration.
    hidden_size: int
    expansion: float
    num_heads: int
    pos_encodings: str

    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0

    # Maximum number of refinement steps for deep supervision.
    deep_supervision_steps_n: int

    # Forward compute dtype used inside the model.
    forward_dtype: str = "bfloat16"


class TransformerBlock(nn.Module):
    """Single Transformer block with attention + SwiGLU MLP and RMSNorm in post-norm form."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()

        self.self_attn = Attention(
            hidden_size=config.hidden_size,
            head_dim=config.hidden_size // config.num_heads,
            num_heads=config.num_heads,
            num_key_value_heads=config.num_heads,
            causal=False
        )
        self.mlp = SwiGLU(
            hidden_size=config.hidden_size,
            expansion=config.expansion,
        )
        self.norm_eps = config.rms_norm_eps

    def forward(self, cos_sin: Optional[CosSin], hidden_states: torch.Tensor) -> torch.Tensor:
        # Post-norm self-attention block.
        hidden_states = rms_norm(
            hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states),
            variance_epsilon=self.norm_eps,
        )
        # Post-norm feed-forward block.
        hidden_states = rms_norm(
            hidden_states + self.mlp(hidden_states),
            variance_epsilon=self.norm_eps,
        )
        return hidden_states


class SingleLayer(nn.Module):
    """
    Shared Transformer layer used as the SR2 reasoning core.

    Historically the model handled a list of independent blocks; this wrapper
    keeps the `self.layers` attribute for backward compatibility but stores
    exactly one shared block that is reused across refinement steps.
    """

    def __init__(self, block: TransformerBlock):
        super().__init__()
        # Register a single shared block while preserving the `layers` attribute.
        self.layers = torch.nn.ModuleList([block])

    def forward(self, hidden_states: torch.Tensor, input_injection: torch.Tensor, **kwargs) -> torch.Tensor:
        # Additive injection of the encoded input into the latent state.
        hidden_states = hidden_states + input_injection
        # Apply the same shared block; repeated refinement is controlled externally.
        shared_block = self.layers[0]
        hidden_states = shared_block(hidden_states=hidden_states, **kwargs)
        return hidden_states


class InnerModel(nn.Module):
    """
    Core SR2 recurrent Transformer.

    Maintains the latent state z, embeds inputs and puzzle identifiers, applies
    positional encodings, and runs the shared Transformer block to iteratively
    refine z. Produces token logits when not in core-only mode.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, self.config.forward_dtype)

        # Token I/O: embeddings and output language-model head.
        self.embed_scale  = math.sqrt(self.config.hidden_size)
        embed_init_std = 1.0 / self.embed_scale

        self.embed_tokens = CastedEmbedding(self.config.vocab_size, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)
        self.lm_head      = CastedLinear(self.config.hidden_size, self.config.vocab_size, bias=False)

        # Puzzle embeddings: length in tokens is ceil(puzzle_emb_ndim / hidden_size).
        self.puzzle_emb_len = -(self.config.puzzle_emb_ndim // -self.config.hidden_size)  # ceil div
        if self.config.puzzle_emb_ndim > 0:
            # Puzzle embeddings are initialized to zero and trained via a sparse optimizer.
            self.puzzle_emb = CastedSparseEmbedding(self.config.num_puzzle_identifiers, self.config.puzzle_emb_ndim,
                                                    batch_size=self.config.batch_size, init_std=0, cast_to=self.forward_dtype)

        # Positional or rotary encodings.
        if self.config.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(dim=self.config.hidden_size // self.config.num_heads,
                                              max_position_embeddings=self.config.seq_len + self.puzzle_emb_len,
                                              base=self.config.rope_theta)
        elif self.config.pos_encodings == "learned":
            self.embed_pos = CastedEmbedding(self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)
        else:
            raise NotImplementedError()

        # ------------------------------------------------------------------
        # Single-layer reasoning module: a shared Transformer block that is
        # reused across refinement steps, implementing a recurrent backbone.
        # ------------------------------------------------------------------
        shared_block = TransformerBlock(self.config)
        self.core = SingleLayer(block=shared_block)

        # Initial latent state vector Z_init used when resetting examples.
        self.Z_init = nn.Buffer(trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1), persistent=True)


    def _input_embeddings(self, input: torch.Tensor, puzzle_identifiers: torch.Tensor):
        # Token embeddings.
        embedding = self.embed_tokens(input.to(torch.int32))

        # Puzzle embeddings (prepended as additional positions).
        if self.config.puzzle_emb_ndim > 0:
            puzzle_embedding = self.puzzle_emb(puzzle_identifiers)

            pad_count = self.puzzle_emb_len * self.config.hidden_size - puzzle_embedding.shape[-1]
            if pad_count > 0:
                puzzle_embedding = F.pad(puzzle_embedding, (0, pad_count))

            embedding = torch.cat((puzzle_embedding.view(-1, self.puzzle_emb_len, self.config.hidden_size), embedding), dim=-2)

        # Learned positional embeddings (if enabled).
        if self.config.pos_encodings == "learned":
            # Scale by 1/sqrt(2) to maintain forward variance.
            embedding = 0.707106781 * (embedding + self.embed_pos.embedding_weight.to(self.forward_dtype))

        # Global scaling to match Transformer initialization.
        return self.embed_scale * embedding

    def empty_carry(self, batch_size: int):
        # Allocate an uninitialized latent state tensor; it will be filled in reset_carry.
        return InnerCarry(
            z=torch.empty(batch_size, self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, dtype=self.forward_dtype),
        )

    def reset_carry(self, reset_flag: torch.Tensor, carry: InnerCarry):
        # Broadcast Z_init into the latent state for samples marked by reset_flag.
        return InnerCarry(
            z=torch.where(reset_flag.view(-1, 1, 1), self.Z_init, carry.z),
        )

    def forward(
        self,
        carry: InnerCarry,
        batch: Dict[str, torch.Tensor],
        steps: torch.Tensor,
        *,
        core_only: bool = False,
        detach_carry_z: Optional[bool] = None,
    ) -> Tuple[InnerCarry, Optional[torch.Tensor], Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]]:
        seq_info = dict(
            cos_sin=self.rotary_emb() if hasattr(self, "rotary_emb") else None,
        )

        # Encode inputs and puzzle identifiers for the current step.
        input_embeddings = self._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])

        # Latent state from the previous carry; gradients flow unless explicitly detached.
        z = carry.z

        # Inject the encoded input only on the first refinement step.
        first_step_mask = (steps == 0).view(-1, 1, 1).to(device=z.device, dtype=z.dtype)
        gated_injection = input_embeddings * first_step_mask  # [B, S_full, D]

        # Core recurrent update of the latent state.
        z = self.core(z, gated_injection, **seq_info)

        # Decide whether to detach the latent state in the new carry.
        if detach_carry_z is None:
            detach_carry_z = (not core_only)
        new_carry = InnerCarry(z=z.detach() if detach_carry_z else z)

        if core_only:
            output = None
        else:
            # Language-model logits for token positions (drop puzzle prefix).
            output = self.lm_head(z)[:, self.puzzle_emb_len:]

        return new_carry, output


class SR2Model(nn.Module):
    """The main model."""

    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = ModelConfig(**config_dict)
        self.inner = InnerModel(self.config)

    @property
    def puzzle_emb(self):
        return self.inner.puzzle_emb

    def initial_carry(self, batch: Dict[str, torch.Tensor]):
        batch_size = batch["inputs"].shape[0]

        return BatchCarry(
            inner_carry=self.inner.empty_carry(batch_size),  # Empty by design; all sequences start halted and will be reset on the first pass.

            steps=torch.zeros((batch_size, ), dtype=torch.int32),
            halted=torch.ones((batch_size, ), dtype=torch.bool),  # Default to halted so that all examples are reset on the first step.

            current_data={k: torch.empty_like(v) for k, v in batch.items()}
        )

    def forward(
        self,
        carry: BatchCarry,
        batch: Dict[str, torch.Tensor],
        *,
        core_only: bool = False,
        detach_carry_z: Optional[bool] = None,
        force_halt: Optional[bool] = None
    ) -> Tuple[BatchCarry, Dict[str, torch.Tensor]]:
        # Update per-example state and cached inputs; reset halted sequences.
        new_inner_carry = self.inner.reset_carry(carry.halted, carry.inner_carry)
        new_steps = torch.where(carry.halted, 0, carry.steps)
        new_current_data = {
            k: torch.where(
                carry.halted.view((-1,) + (1,) * (batch[k].ndim - 1)),
                batch[k], v
            ) for k, v in carry.current_data.items()
        }

        # Forward the inner model for one refinement step (possibly core-only).
        new_inner_carry, logits = self.inner(
            new_inner_carry, new_current_data, new_steps,
            core_only=core_only,
            detach_carry_z=detach_carry_z
        )

        outputs: Dict[str, torch.Tensor] = {}
        if not core_only:
            outputs["logits"] = logits  # type: ignore

        with torch.no_grad():
            # Increase step count only when computing full outputs.
            if not core_only:
                new_steps = new_steps + 1

            # Halt once the maximum number of supervision steps is reached.
            is_last_step = new_steps >= self.config.deep_supervision_steps_n
            halted = is_last_step

        return BatchCarry(
            new_inner_carry, new_steps, halted, new_current_data
        ), outputs
