from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F

try:
    from flash_attn_interface import flash_attn_func  # type: ignore[import]
except ImportError:
    # Fallback to FlashAttention 2
    from flash_attn import flash_attn_func  # type: ignore[import]

from models.common import trunc_normal_init_


CosSin = Tuple[torch.Tensor, torch.Tensor]


def _find_multiple(a, b):
    return (-(a // -b)) * b


def rotate_half(x: torch.Tensor):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    # q, k: [bs, seq_len, num_heads, head_dim]
    # cos, sin: [seq_len, head_dim]
    orig_dtype = q.dtype
    q = q.to(cos.dtype)
    k = k.to(cos.dtype)

    q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
    k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))

    return q_embed.to(orig_dtype), k_embed.to(orig_dtype)


class CastedLinear(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool):
        super().__init__()
        # Truncated LeCun normal init
        self.weight = nn.Parameter(
            trunc_normal_init_(torch.empty((out_features, in_features)), std=1.0 / (in_features ** 0.5))
        )
        self.bias = None
        if bias:
            # Zero init bias
            self.bias = nn.Parameter(torch.zeros((out_features, )))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)


class CastedEmbedding(nn.Module):
    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 init_std: float,
                 cast_to: torch.dtype):
        super().__init__()
        self.cast_to = cast_to

        # Truncated LeCun normal init
        self.embedding_weight = nn.Parameter(
            trunc_normal_init_(torch.empty((num_embeddings, embedding_dim)), std=init_std)
        )
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.embedding(input, self.embedding_weight.to(self.cast_to))


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings, base, device=None):
        super().__init__()

        # RoPE
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)

        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached = nn.Buffer(emb.cos(), persistent=False)
        self.sin_cached = nn.Buffer(emb.sin(), persistent=False)

    def forward(self):
        return self.cos_cached, self.sin_cached
    
class RotaryEmbedding2D(nn.Module):
    """
    2D RoPE（与当前 rotate_half 兼容的实现）
    - 将 head_dim 分为四等份：前一半中的前 1/2 维度用于 x 轴相位，后一半中的前 1/2 维度用于 y 轴相位；
      再通过复制到后半维并配合 rotate_half，实现每个“成对维度”使用对应轴的相位。
    - 要求 head_dim % 4 == 0。
    - 对 puzzle 前缀位置使用 (0,0)，从而不旋转（cos=1, sin=0）。
    """
    def __init__(
        self,
        dim: int,  # = head_dim
        max_position_embeddings: int,
        base: float,
        grid_h: int,
        grid_w: int,
        puzzle_prefix_len: int = 0,
        order: str = "row_major",
        device=None,
    ):
        super().__init__()
        assert dim % 4 == 0, "For 2D RoPE with rotate_half, head_dim must be divisible by 4."
        self.dim = dim
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.puzzle_prefix_len = int(puzzle_prefix_len)
        self.order = order

        # 每个轴的“维度标尺”：按经典 RoPE 的公式，但把“轴内维度”设为 dim/2
        dim_axis = dim // 2  # 每根轴负责 dim/2 维
        inv_freq_axis = 1.0 / (base ** (torch.arange(0, dim_axis, 2, dtype=torch.float32, device=device) / dim_axis))
        # 上式长度为 dim_axis/2 = dim/4，正好是“每轴的成对旋转数目”

        # === 构造 2D 位置 (x,y) ===
        total_seq = max_position_embeddings
        grid_len = grid_h * grid_w
        assert self.puzzle_prefix_len + grid_len <= total_seq, \
            "puzzle_prefix_len + grid_h*grid_w must be <= max_position_embeddings"

        # 网格内索引：默认行优先展开
        idx = torch.arange(grid_len, dtype=torch.int64, device=device)
        if order == "row_major":
            x = torch.div(idx, grid_w, rounding_mode='floor')  # 行号 [0..grid_h-1]
            y = idx % grid_w                                 # 列号 [0..grid_w-1]
        elif order == "col_major":
            x = idx % grid_h
            y = torch.div(idx, grid_h, rounding_mode='floor')
        else:
            raise ValueError(f"Unknown grid order: {order}")

        # 前缀位置 (0,0)，确保不旋转
        if self.puzzle_prefix_len > 0:
            zeros = torch.zeros(self.puzzle_prefix_len, dtype=torch.float32, device=device)
            x_full = torch.cat([zeros, x.to(torch.float32)], dim=0)
            y_full = torch.cat([zeros, y.to(torch.float32)], dim=0)
        else:
            x_full = x.to(torch.float32)
            y_full = y.to(torch.float32)

        # 若 total_seq 超过 prefix+grid_len，则补零（无旋转）
        pad = total_seq - (self.puzzle_prefix_len + grid_len)
        if pad > 0:
            z = torch.zeros(pad, dtype=torch.float32, device=device)
            x_full = torch.cat([x_full, z], dim=0)
            y_full = torch.cat([y_full, z], dim=0)

        # === 计算每轴的相位矩阵 ===
        # 形状: [total_seq, dim/4]
        freqs_x = torch.outer(x_full, inv_freq_axis)
        freqs_y = torch.outer(y_full, inv_freq_axis)

        # 先得到“前半维”的频率：前 1/4 用 x，相邻的 1/4 用 y
        # 形状: [total_seq, dim/2]
        emb_half = torch.cat([freqs_x, freqs_y], dim=-1)

        # 再复制到“后半维”，与 rotate_half 的“跨半维配对”逻辑一致
        # 形状: [total_seq, dim]
        emb = torch.cat([emb_half, emb_half], dim=-1)

        # 缓存 cos/sin
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self):
        return self.cos_cached, self.sin_cached



class Attention(nn.Module):
    def __init__(self, hidden_size, head_dim, num_heads, num_key_value_heads, causal=False):
        super().__init__()

        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.output_size = head_dim * num_heads
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.causal = causal

        self.qkv_proj = CastedLinear(self.hidden_size, (self.num_heads + 2 * self.num_key_value_heads) * self.head_dim, bias=False)
        self.o_proj = CastedLinear(self.output_size, self.hidden_size, bias=False)

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        # hidden_states: [bs, seq_len, num_heads, head_dim]
        qkv = self.qkv_proj(hidden_states)

        # Split head
        qkv = qkv.view(batch_size, seq_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)
        query = qkv[:, :, :self.num_heads]
        key = qkv[:, :, self.num_heads: self.num_heads + self.num_key_value_heads]
        value = qkv[:, :, self.num_heads + self.num_key_value_heads:]

        # RoPE
        if cos_sin is not None:
            cos, sin = cos_sin
            query, key = apply_rotary_pos_emb(query, key, cos, sin)

        # flash attn
        attn_output = flash_attn_func(q=query, k=key, v=value, causal=self.causal)
        if isinstance(attn_output, tuple):  # fa2 and fa3 compatibility
            attn_output = attn_output[0]

        # attn_output: [batch_size, num_heads, seq_len, head_dim]
        attn_output = attn_output.view(batch_size, seq_len, self.output_size)  # type: ignore
        return self.o_proj(attn_output)


class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, expansion: float):
        super().__init__()
        inter = _find_multiple(round(expansion * hidden_size * 2 / 3), 256)

        self.gate_up_proj = CastedLinear(hidden_size, inter * 2, bias=False)
        self.down_proj    = CastedLinear(inter, hidden_size, bias=False)

    def forward(self, x):
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)


def rms_norm(hidden_states: torch.Tensor, variance_epsilon: float) -> torch.Tensor:
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)

    variance = hidden_states.square().mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
    return hidden_states.to(input_dtype)
