import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# FlashAttention path removed from this simplified version.


# Optional FlashAttention (varlen) for packed path (B==1)
try:
    from flash_attn import flash_attn_varlen_func
    FLASH_ATTN_AVAILABLE = True
except Exception:
    FLASH_ATTN_AVAILABLE = False

# --------------------- helpers ---------------------
def count_non_embedding_params(model: nn.Module) -> int:
    """Count trainable parameters excluding token embedding weights.

    Returns
    - int: Number of parameters not in `tok_emb.weight`.
    """
    return sum(
        p.numel()
        for name, p in model.named_parameters()
        if p.requires_grad and "tok_emb.weight" not in name
    )


def build_4d_attn_mask(
    attn_mask_1d: torch.Tensor,
    *,
    sliding_window: Optional[int] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Create additive attention bias from a padding mask and causal/sliding window.

    Args
    - attn_mask_1d: [B, T] with 1 (valid) or 0 (pad). Any non-zero treated as valid.
    - sliding_window: if set, allow attending only to last `sliding_window` tokens.
    - dtype: output dtype (match attention scores dtype).

    Shapes
    - returns: [B, 1, T, T] additive bias with 0 for allowed positions and -inf for disallowed.
    """
    B, T = attn_mask_1d.shape
    device = attn_mask_1d.device
    neg_inf = torch.finfo(dtype).min

    # Start with all zeros
    bias = torch.zeros((B, 1, T, T), device=device, dtype=dtype)

    # Causal: disallow j > i
    causal = torch.triu(torch.ones((T, T), device=device, dtype=torch.bool), diagonal=1)
    bias[:, :, causal] = neg_inf

    # Sliding window: disallow j < i - (W - 1)
    if sliding_window is not None and sliding_window > 0:
        i_idx = torch.arange(T, device=device).view(T, 1)
        j_idx = torch.arange(T, device=device).view(1, T)
        lower_exceed = j_idx < (i_idx - (sliding_window - 1))
        bias[:, :, lower_exceed] = neg_inf

    # Key padding mask: disallow attending to padded keys (where mask == 0)
    key_invalid = (attn_mask_1d == 0).view(B, 1, 1, T)
    bias = bias.masked_fill(key_invalid, neg_inf)

    return bias


def build_rope_frequencies(
    max_position: int,
    head_dim: int,
    base: float,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Precompute RoPE cos/sin lookup tables.

    Args
    - max_position: number of positions, T_max.
    - head_dim: per-head hidden size, D (must be even).
    - base: RoPE base, typically 10000.0.
    - device: tensor device.

    Shapes
    - returns: (cos, sin) each [T_max, D].
    """
    assert head_dim % 2 == 0, "head_dim must be even for RoPE"

    # positions: [T_max]
    pos = torch.arange(max_position, device=device, dtype=torch.float32)
    # inverse frequencies: [D/2]
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    # outer product: [T_max, D/2]
    freqs = torch.einsum("t,f->tf", pos, inv_freq)
    # interleave to [T_max, D]
    cos = torch.cos(freqs).repeat_interleave(2, dim=-1)
    sin = torch.sin(freqs).repeat_interleave(2, dim=-1)
    return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Swap and negate halves of the last dimension.

    Shapes
    - x: [..., D]
    - returns: [..., D]
    """
    d = x.shape[-1]
    x1, x2 = x[..., : d // 2], x[..., d // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to query/key.

    Shapes
    - q, k: [B, H, T, D]
    - cos, sin: [T, D]
    - returns: (q_rot, k_rot) each [B, H, T, D]
    """
    # Broadcast cos/sin over batch and heads: [1, 1, T, D]
    cos = cos.to(q.dtype)
    sin = sin.to(q.dtype)
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot, k_rot

def apply_rotary_pos_emb_flat(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    pos_idx_flat: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to q/k given packed (flat) positions.

    Shapes
    - q, k: [N, H, D]
    - cos, sin: [T_max, D]
    - pos_idx_flat: [N] positions per token in the packed stream
    - returns: (q_rot, k_rot) [N, H, D]
    """
    cos = cos.to(q.dtype)
    sin = sin.to(q.dtype)
    cos_nd = cos.index_select(0, pos_idx_flat.to(torch.long)).unsqueeze(1)
    sin_nd = sin.index_select(0, pos_idx_flat.to(torch.long)).unsqueeze(1)
    q_rot = (q * cos_nd) + (rotate_half(q) * sin_nd)
    k_rot = (k * cos_nd) + (rotate_half(k) * sin_nd)
    return q_rot, k_rot

def prepare_fa_kwargs_from_position_ids(position_ids: torch.Tensor):
    """Build varlen cu_seqlens from position resets (pos == 0) over flattened tokens.

    Args
    - position_ids: [B, T] int64 with positions resetting to 0 at each new sequence.

    Returns
    - cu_seqlens: [num_seqs + 1] int32 cumulative lengths (starts include 0, ends with total tokens)
    - max_seqlen: Python int: max length across sequences
    - pos_idx_flat: [N] int64 positions corresponding to flattened tokens
    """
    assert position_ids.dtype in (torch.int64, torch.long), "position_ids must be int64"
    flat = position_ids.reshape(-1)
    starts = (flat == 0).nonzero(as_tuple=False).reshape(-1).to(torch.int32)
    total = torch.tensor([flat.numel()], device=flat.device, dtype=torch.int32)
    cu = torch.cat([starts, total], dim=0)
    max_len = int(torch.diff(cu).max().item()) if cu.numel() > 1 else int(total.item())
    return cu, max_len, flat

def apply_rotary_pos_emb_with_posids(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to q/k using per-token positions.

    Shapes
    - q, k: [B, H, T, D]
    - cos, sin: [T_max, D]
    - position_ids: [B, T] int positions or 0/1 mask
    - returns: (q_rot, k_rot) [B, H, T, D]
    """
    B, H, T, D = q.shape
    # Derive position indices: if boolean/0-1 use cumulative positions, else use given positions.
    cos = cos.to(q.dtype)
    sin = sin.to(q.dtype)
    if position_ids.dtype == torch.bool or (position_ids.dtype.is_floating_point is False and position_ids.max() <= 1):
        mask = position_ids.to(torch.int32)
        pos_idx = torch.clamp(mask.cumsum(dim=-1) - 1, min=0)  # [B, T]
    else:
        pos_idx = position_ids.to(torch.long)
    # Clip to rope table bounds
    max_pos = cos.shape[0]
    pos_idx = pos_idx.clamp(min=0, max=max_pos - 1)
    # Gather cos/sin per token: [B, T, D] -> [B, 1, T, D]
    flat = pos_idx.view(-1)
    cos_bt = cos.index_select(0, flat).view(B, T, D).unsqueeze(1)
    sin_bt = sin.index_select(0, flat).view(B, T, D).unsqueeze(1)
    q_rot = (q * cos_bt) + (rotate_half(q) * sin_bt)
    k_rot = (k * cos_bt) + (rotate_half(k) * sin_bt)
    return q_rot, k_rot

## Packed RoPE helper removed in simplified version.


## FlashAttention varlen helpers removed in simplified version.

## FlashAttention varlen helpers removed in simplified version.


# --------------------- config ---------------------
@dataclass
class Config:
    vocab_size: int = 50257
    n_layers: int = 2
    n_heads: int = 10
    d_model: int = 640
    d_mlp: int = 2560
    max_seq_len: int = 512
    sliding_window: int = 1024
    rope_base: float = 10000.0
    tie_output: bool = True


# --------------------- modules ---------------------
class Attention(nn.Module):
    """Multi-head attention with RoPE.

    Simplified: only the standard, un-packed attention path is implemented.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.d_model = cfg.d_model
        self.n_heads = cfg.n_heads
        self.head_dim = self.d_model // self.n_heads

        # Projections: fused QKV and output
        self.qkv_weight = nn.Parameter(torch.empty(3 * self.d_model, self.d_model))
        self.qkv_bias = nn.Parameter(torch.zeros(3 * self.d_model))
        self.out_proj_weight = nn.Parameter(torch.empty(self.d_model, self.d_model))
        self.out_proj_bias = nn.Parameter(torch.zeros(self.d_model))

        for param in (self.qkv_weight, self.out_proj_weight):
            nn.init.normal_(param, mean=0.0, std=0.02)

    def _attention_basic(
        self,
        x: torch.Tensor,
        attn_bias: torch.Tensor,
        rope: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Reference attention implementation using PyTorch ops.

        Shapes
        - x: [B, T, C]
        - attn_bias: [B, 1, T, T]
        - rope: (cos, sin) each [T_max, D]
        - returns: [B, T, C]
        """
        B, T, C = x.shape
        H, D = self.n_heads, self.head_dim
        cos, sin = rope

        # Project to QKV: [B, T, 3C] -> [B, T, C] x3
        qkv = F.linear(x, self.qkv_weight, self.qkv_bias)
        q, k, v = qkv.split(C, dim=-1)

        # Reshape to heads: [B, H, T, D]
        q = q.view(B, T, H, D).transpose(1, 2)
        k = k.view(B, T, H, D).transpose(1, 2)
        v = v.view(B, T, H, D).transpose(1, 2)

        # Apply RoPE using first T positions
        q, k = apply_rotary_pos_emb(q, k, cos[:T], sin[:T])

        # Attention
        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(D)  # [B, H, T, T]
        attn_scores = attn_scores + attn_bias  # broadcast [B, 1, T, T]
        attn_probs = F.softmax(attn_scores, dim=-1)
        context = attn_probs @ v  # [B, H, T, D]

        # Merge heads
        context = context.transpose(1, 2).reshape(B, T, C)
        out = F.linear(context, self.out_proj_weight, self.out_proj_bias)
        return out

    def _attention_flash_varlen(
        self,
        x: torch.Tensor,
        rope: Tuple[torch.Tensor, torch.Tensor],
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        """FlashAttention varlen path for packed inputs (B == 1).

        Shapes
        - x: [1, T, C]
        - rope: (cos, sin) [T_max, D]
        - position_ids: [1, T] int64 absolute positions per sequence (resets to 0)
        - returns: [1, T, C]
        """
        assert FLASH_ATTN_AVAILABLE, "FlashAttention is not available"
        assert x.is_cuda, "flash_attn_varlen requires CUDA tensors"

        B, T, C = x.shape
        assert B == 1, "Varlen path expects B == 1 (packed)"
        H, D = self.n_heads, self.head_dim
        cos, sin = rope

        # Project to QKV: [1, T, 3C] -> split
        qkv = F.linear(x, self.qkv_weight, self.qkv_bias)
        q, k, v = qkv.split(C, dim=-1)
        # [1, H, T, D] -> flatten to [T, H, D]
        q = q.view(B, T, H, D).transpose(1, 2).reshape(T, H, D)
        k = k.view(B, T, H, D).transpose(1, 2).reshape(T, H, D)
        v = v.view(B, T, H, D).transpose(1, 2).reshape(T, H, D)

        # Varlen args from position resets
        cu, max_len, pos_flat = prepare_fa_kwargs_from_position_ids(position_ids)
        # Apply RoPE in packed space
        q, k = apply_rotary_pos_emb_flat(q, k, cos, sin, pos_flat.to(torch.long))

        # FlashAttention varlen: [T, H, D]
        out_flat = flash_attn_varlen_func(
            q, k, v,
            cu_seqlens_q=cu,
            cu_seqlens_k=cu,
            max_seqlen_q=max_len,
            max_seqlen_k=max_len,
            dropout_p=0.0,
            softmax_scale=None,
            causal=True,
            sliding_window=self.cfg.sliding_window,
        )
        if isinstance(out_flat, tuple):
            out_flat = out_flat[0]

        # Merge heads and reshape back to [1, T, C]
        context = out_flat.reshape(T, H * D).unsqueeze(0)
        out = F.linear(context, self.out_proj_weight, self.out_proj_bias)
        return out

    def forward(
        self,
        x: torch.Tensor,
        rope: Tuple[torch.Tensor, torch.Tensor],
        attn_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute attention with RoPE.

        API stays the same; internally selects:
        - If B == 1, CUDA, FLASH available, and position_ids provided → varlen.
        - Else → standard attention using attn_mask additive bias [B,1,T,T].
        """
        B = x.shape[0]
        if (
            B == 1
            and position_ids is not None
            and FLASH_ATTN_AVAILABLE
            and x.is_cuda
        ):
            return self._attention_flash_varlen(x, rope, position_ids)
        assert attn_mask is not None, "attn_mask required for standard attention"
        return self._attention_basic(x, attn_mask, rope)


class MLP(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        d_model, d_mlp = cfg.d_model, cfg.d_mlp
        self.fc1_w = nn.Parameter(torch.empty(d_mlp, d_model))
        self.fc1_b = nn.Parameter(torch.zeros(d_mlp))
        self.fc2_w = nn.Parameter(torch.empty(d_model, d_mlp))
        self.fc2_b = nn.Parameter(torch.zeros(d_model))
        for param in (self.fc1_w, self.fc2_w):
            nn.init.normal_(param, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Two-layer MLP with GELU activation.

        Shapes
        - x: [B, T, C]
        - returns: [B, T, C]
        """
        x = F.linear(x, self.fc1_w, self.fc1_b)
        x = F.gelu(x)
        x = F.linear(x, self.fc2_w, self.fc2_b)
        return x


class LayerNorm(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.d_model = cfg.d_model
        self.ln_w = nn.Parameter(torch.ones(self.d_model))
        self.ln_b = nn.Parameter(torch.zeros(self.d_model))
        self.eps = 1e-5
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Affine LayerNorm matching GPT-2 convention.

        Shapes
        - x: [B, T, C]
        - returns: [B, T, C]
        """
        return F.layer_norm(x, (self.d_model,), self.ln_w, self.ln_b, self.eps)

class Block(nn.Module):
    """Transformer block: LayerNorm -> Attention -> residual -> LayerNorm -> MLP -> residual."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.d_model = cfg.d_model
        self.ln1 = LayerNorm(cfg)
        self.ln2 = LayerNorm(cfg)
        self.attn = Attention(cfg)
        self.mlp = MLP(cfg)

    def forward(
        self,
        x: torch.Tensor,
        rope: Tuple[torch.Tensor, torch.Tensor],
        attn_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Attention with pre-LN
        x = x + self.attn(self.ln1(x), rope, attn_mask=attn_mask, position_ids=position_ids)
        # MLP with pre-LN
        x = x + self.mlp(self.ln2(x))
        return x


class MinimalGPT2(nn.Module):
    """A minimal GPT-2-like model with RoPE."""

    def __init__(self, cfg: Config):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0

        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.ln_f_w = nn.Parameter(torch.ones(cfg.d_model))
        self.ln_f_b = nn.Parameter(torch.zeros(cfg.d_model))

        if cfg.tie_output:
            self.out_w = None
            self.out_b = nn.Parameter(torch.zeros(cfg.vocab_size))
        else:
            self.out_w = nn.Parameter(torch.empty(cfg.vocab_size, cfg.d_model))
            self.out_b = nn.Parameter(torch.zeros(cfg.vocab_size))
            nn.init.normal_(self.out_w, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attn_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Shapes
        - input_ids: [B, T]
        - attn_mask: [B, T] (nonzero=valid). If None and B==1 with position_ids, uses varlen.
        - position_ids: [B, T] optional int64; sizes RoPE and enables varlen when B==1.
        - returns: logits [B, T, V]
        """
        cfg = self.cfg
        device = input_ids.device
        B, T = input_ids.shape

        # Token embeddings: [B, T, C]
        x = self.tok_emb(input_ids)

        # RoPE tables up to needed max position
        max_position = int(position_ids.max().item()) + 1 if position_ids is not None else T
        rope = build_rope_frequencies(max_position, self.blocks[0].attn.head_dim, cfg.rope_base, device)

        # Additive attention bias from mask with causal + optional sliding window
        if attn_mask is not None:
            attn_mask = build_4d_attn_mask(attn_mask, sliding_window=cfg.sliding_window, dtype=x.dtype)

        # Transformer blocks
        for block in self.blocks:
            x = block(x, rope, attn_mask=attn_mask, position_ids=position_ids)

        # Final LayerNorm
        x = F.layer_norm(x, (cfg.d_model,), self.ln_f_w, self.ln_f_b, 1e-5)

        # Output logits (optionally tied)
        if cfg.tie_output:
            logits = F.linear(x, self.tok_emb.weight, self.out_b)
        else:
            logits = F.linear(x, self.out_w, self.out_b)
        return logits

    


if __name__ == "__main__":
    # Simple sanity check
    vocab_size = 50257
    cfg = Config(
        vocab_size=vocab_size,
        n_layers=2,
        n_heads=10,
        d_model=640,
        d_mlp=2560,
        max_seq_len=512,
        rope_base=10000.0,
        tie_output=True,
    )
    model = MinimalGPT2(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    x = torch.randint(0, vocab_size, (2, 8), device=device)
    attn_mask = torch.ones((2, 8), dtype=torch.int64, device=device)
    with torch.no_grad():
        logits = model(x, attn_mask=attn_mask)
    print("Logits:", tuple(logits.shape))
