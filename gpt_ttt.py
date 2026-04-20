import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
# Avoid external dependencies; use native reshape/transposes instead of einops.


# Optional FlashAttention (varlen) for packed path (B==1)
try:
    from flash_attn import flash_attn_varlen_func
    FLASH_ATTN_AVAILABLE = True
except Exception:
    FLASH_ATTN_AVAILABLE = False


# --------------------- config ---------------------
@dataclass
class Config:
    vocab_size: int = 50257
    n_layers: int = 4
    n_heads: int = 8
    d_head: int = 64
    d_model: int = 512
    d_mlp: int = 2048
    d_ttt: int = 256
    max_seq_len: int = 1024
    sliding_window: int = 2048
    rope_base: float = 10000.0
    tie_output: bool = True
    use_mlp: bool = True
    ttt_momentum: float = 0.0  # momentum coefficient for TTT fast-weight updates (0 disables)


# --------------------- helpers ---------------------
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., D]
        scale = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * scale * self.weight

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
    - q, k: [H, N, D]
    - cos, sin: [T_max, D]
    - pos_idx_flat: [N] positions per token in the packed stream
    - returns: (q_rot, k_rot) [H, N, D]
    """
    cos = cos.to(q.dtype)
    sin = sin.to(q.dtype)
    # pos_idx_flat: [N]
    # cos/sin: [T_max, D]
    # We want to broadcast cos/sin to [1, N, D] to match [H, N, D]
    cos_nd = cos.index_select(0, pos_idx_flat.to(torch.long)).unsqueeze(0)  # [1, N, D]
    sin_nd = sin.index_select(0, pos_idx_flat.to(torch.long)).unsqueeze(0)  # [1, N, D]
    q_rot = (q * cos_nd) + (rotate_half(q) * sin_nd)
    k_rot = (k * cos_nd) + (rotate_half(k) * sin_nd)
    return q_rot, k_rot


def prepare_fa_kwargs_from_position_ids(position_ids: torch.Tensor):
    """Build varlen cu_seqlens for FlashAttention.

    Args
    - position_ids: [B, T] int64 positions (may reset between samples).

    Returns
    - cu_seqlens: [2] int32 cumulative lengths for a single contiguous sequence [0, N]
    - max_seqlen: Python int: max length across sequences (equals N)
    - pos_idx_flat: [N] int64 positions corresponding to flattened tokens
    """
    assert position_ids.dtype in (torch.int64, torch.long), "position_ids must be int64"
    flat = position_ids.reshape(-1)
    # Treat the entire packed stream as a single contiguous sequence to allow
    # cross-sample attention. Do not split on position resets.
    total = torch.tensor([flat.numel()], device=flat.device, dtype=torch.int32)
    zero = torch.tensor([0], device=flat.device, dtype=torch.int32)
    cu = torch.cat([zero, total], dim=0)
    max_len = int(total.item())
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
        attn_mask: torch.Tensor,
        rope: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Reference attention implementation using PyTorch ops.

        Shapes
        - x: [B, T, C]
        - attn_mask: [B, 1, T, T]
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
        q = q.view(B, T, H, D).transpose(1, 2) # [B, H, T, D]
        k = k.view(B, T, H, D).transpose(1, 2) # [B, H, T, D]
        v = v.view(B, T, H, D).transpose(1, 2) # [B, H, T, D]

        # Apply RoPE using first T positions
        q, k = apply_rotary_pos_emb(q, k, cos[:T], sin[:T]) # [B, H, T, D]

        # Attention
        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(D)  # [B, H, T, T]
        attn_scores = attn_scores + attn_mask  # broadcast [B, 1, T, T]
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



class LaCTLayer(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.d_model = cfg.d_model
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_head
        self.d_ttt = cfg.d_ttt

        self.qkv_w = nn.Parameter(torch.empty(3 * self.d_head * self.n_heads, self.d_model))
        self.qkv_b = nn.Parameter(torch.zeros(3 * self.d_head * self.n_heads))
        self.o_proj_w = nn.Parameter(torch.empty(self.d_model, self.d_head * self.n_heads))
        self.o_proj_b = nn.Parameter(torch.zeros(self.d_model))

        if not cfg.d_ttt == 0:
            self.rms_out = RMSNorm(self.d_model)
            self.learnable_q_scale = nn.Parameter(torch.ones(self.n_heads, self.d_head))
            self.learnable_q_bias = nn.Parameter(torch.zeros(self.n_heads, self.d_head))
            self.learnable_k_scale = nn.Parameter(torch.ones(self.n_heads, self.d_head))
            self.learnable_k_bias = nn.Parameter(torch.zeros(self.n_heads, self.d_head))

            self.scale_per_head_w = nn.Parameter(torch.randn(self.d_head, self.d_head))
            self.scale_per_head_b = nn.Parameter(torch.zeros(self.d_head))
            # Fast-weight parameters per head
            # We use shapes:
            #   w1: [H, dh, dm]  (gate path)
            #   w3: [H, dh, dm]  (hidden path)
            #   w2: [H, dm, dh]  (output back to dh)
            H, dh, dm = self.n_heads, self.d_head, self.d_ttt
            self.w1 = torch.randn(H, dh, dm)
            self.w3 = torch.randn(H, dh, dm)
            self.w2 = torch.randn(H, dm, dh)
            # Momentum buffers for fast-weights (not parameters)
            self.v1 = torch.zeros_like(self.w1)
            self.v2 = torch.zeros_like(self.w2)
            self.v3 = torch.zeros_like(self.w3)

            # Per-token LR projection for potential TTT updates (not used in basic forward)
            self.lr_w = nn.Parameter(torch.empty(self.n_heads * 3, self.d_model))
            self.lr_b = nn.Parameter(torch.zeros(self.n_heads * 3))
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.qkv_w, mean=0.0, std=0.02)
        nn.init.zeros_(self.qkv_b)
        nn.init.normal_(self.o_proj_w, mean=0.0, std=0.02)
        nn.init.zeros_(self.o_proj_b)
        if not self.cfg.d_ttt == 0:
            nn.init.normal_(self.lr_w, mean=0.0, std=0.02)
            nn.init.zeros_(self.lr_b)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        rope: Tuple[torch.Tensor, torch.Tensor],
        *,
        position_ids: Optional[torch.Tensor] = None,
        fast_weight: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
        ttt_config: Optional[list] = None,
        use_muon: bool = False,
    ) -> torch.Tensor:
        B, T, D = x.shape
        H, D_head = self.n_heads, self.d_head

        # QKV projections
        qkv = F.linear(x, self.qkv_w, self.qkv_b)
        q, k, v = qkv.split(self.d_head * self.n_heads, dim=-1)
        q = q.view(B, T, H, D_head).transpose(1, 2)  # [B, H, T, dh]
        k = k.view(B, T, H, D_head).transpose(1, 2)
        v = v.view(B, T, H, D_head).transpose(1, 2)

        #attn_q = q * self.learnable_q_scale[None, :, None, :] + self.learnable_q_bias[None, :, None, :]
        #attn_k = k * self.learnable_k_scale[None, :, None, :] + self.learnable_k_bias[None, :, None, :]

        # Attention output (merge heads to [B,T,D])
        if (
            B == 1
            and position_ids is not None
            and FLASH_ATTN_AVAILABLE
            and x.is_cuda
        ):
            attn_out = self._attention_flash_varlen_from_qkv(q, k, v, rope, position_ids)
        else:
            assert attn_mask is not None, "attn_mask required for standard attention"
            attn_out = self._attention_basic(q, k, v, attn_mask, rope)

        # LaCT TTT component (chunked); returns [B,T,D]
        if self.cfg.d_ttt > 0:
            ttt_out = self._lact_ttt_forward(x, q, k, v, rope, ttt_config=ttt_config, fast_weight=fast_weight, use_muon=use_muon)
            ttt_out = ttt_out.reshape(B, T, H, D_head)
            ttt_out = ttt_out * F.silu(F.linear(ttt_out, self.scale_per_head_w, self.scale_per_head_b))
            ttt_out = ttt_out.reshape(B, T, H * D_head)

            out = attn_out + ttt_out
        else:
            out = attn_out
        out = F.linear(out, self.o_proj_w, self.o_proj_b)
        return out

    def _attention_basic(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: torch.Tensor,
        rope: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Reference attention implementation using PyTorch ops.

        Shapes
        - q: [B, H, T, D_head]
        - k: [B, H, T, D_head]
        - v: [B, H, T, D_head]
        - attn_mask: [B, 1, T, T]
        - rope: (cos, sin) each [T_max, D]
        - returns: [B, T, C]
        """
        B, H, T, D_head = q.shape
        H, D = self.n_heads, self.d_head
        cos, sin = rope

        # Apply RoPE using first T positions
        q, k = apply_rotary_pos_emb(q, k, cos[:T], sin[:T]) # [B, H, T, D_head]

        # Attention
        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(D)  # [B, H, T, T]
        attn_scores = attn_scores + attn_mask  # broadcast [B, 1, T, T]
        attn_probs = F.softmax(attn_scores, dim=-1)  # [B, H, T, T]
        context = attn_probs @ v  # [B,H,T,dh]
        return context.transpose(1, 2).reshape(B, T, H * D_head)


    def _apply_ttt(self, fast_weight, q) -> torch.Tensor:
        """Apply fast-weight gated MLP per head.

        Shapes
        - q: [B,H,T,dh]
        - w1: [H,dh,dm], w3: [H,dh,dm], w2: [H,dm,dh]
        - returns: [B,H,T,dh]
        """
        w1, w2, w3 = fast_weight

        gate_before = torch.einsum("bhtd,hdk->bhtk", q, w1)
        hidden_before = torch.einsum("bhtd,hdk->bhtk", q, w3)
        hidden = F.silu(gate_before) * hidden_before  # [B,H,T,dm]
        out = torch.einsum("bhtm,hmk->bhtk", hidden, w2)  # [B,H,T,dh]
        
        return out

    def _attention_flash_varlen_from_qkv(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        rope: Tuple[torch.Tensor, torch.Tensor],
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        assert q.shape[0] == 1
        B, H, T, dh = q.shape
        cos, sin = rope
        qf = q.transpose(1, 2).reshape(T, H, dh)
        kf = k.transpose(1, 2).reshape(T, H, dh)
        vf = v.transpose(1, 2).reshape(T, H, dh)
        cu, max_len, pos_flat = prepare_fa_kwargs_from_position_ids(position_ids)
        qf, kf = apply_rotary_pos_emb_flat(qf, kf, cos, sin, pos_flat.to(torch.long))
        out_flat = flash_attn_varlen_func(
            qf, kf, vf,
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
        return out_flat.reshape(T, H * dh).unsqueeze(0)

    def _lact_ttt_forward(
        self,
        x: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        rope: Tuple[torch.Tensor, torch.Tensor],
        *,
        fast_weight: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
        ttt_config: Optional[list] = None,
        use_muon: bool = False,
    ) -> torch.Tensor:
        B, H, T, dh = q.shape
        # Normalize q,k per token
        q, k, v = F.silu(q), F.silu(k), F.silu(v)
        qn = q / (q.norm(dim=-1, keepdim=True) + 1e-6)
        kn = k / (k.norm(dim=-1, keepdim=True) + 1e-6)
        # Per-token LR: [B,T,3H] -> [B,H,T,3]
        lr_tok = F.softplus(F.linear(x, self.lr_w, self.lr_b)).view(B, T, H, 3).permute(0, 2, 1, 3).contiguous()

        # Use shared fast weights per head (no batch dimension)
        if fast_weight is None:
            w1, w2, w3 = self.w1, self.w2, self.w3  # [H, dh, dm], [H, dm, dh], [H, dh, dm]
        else:
            self.w1, self.w2, self.w3 = fast_weight
            w1, w2, w3 = self.w1, self.w2, self.w3

        lact_o = q.new_zeros(B, H, T, dh)
        if ttt_config is None:
            ttt_config = [("apply_only", 0, T)]

        for mode, begin, end in ttt_config:
            qi, ki, vi = (t[:, :, begin:end, :] for t in (qn, kn, v))
            lri = lr_tok[:, :, begin:end, :]  # [B,H,Lc,3]
            lr1, lr2, lr3 = (lri[..., i:i+1] for i in range(3))

            if mode == "update_then_apply":
                w1, w2, w3 = self._ttt_update(ki, vi, w1, w2, w3, lr1, lr2, lr3, use_muon)
                lact_o[:, :, begin:end, :] = self._apply_ttt((w1, w2, w3), qi)
            elif mode == "apply_then_update":
                lact_o[:, :, begin:end, :] = self._apply_ttt((w1, w2, w3), qi)
                w1, w2, w3 = self._ttt_update(ki, vi, w1, w2, w3, lr1, lr2, lr3, use_muon)
            elif mode == "update_only":
                w1, w2, w3 = self._ttt_update(ki, vi, w1, w2, w3, lr1, lr2, lr3, use_muon)
            elif mode == "apply_only":
                lact_o[:, :, begin:end, :] = self._apply_ttt((w1, w2, w3), qi)
            else:
                raise ValueError(f"Unknown TTT mode: {mode}")

        # Persist updated fast-weights across calls (batch-to-batch)
        # Detach to avoid linking computation graphs across steps
        self.w1, self.w2, self.w3 = w1.detach().contiguous(), w2.detach().contiguous(), w3.detach().contiguous()
        self.v1, self.v2, self.v3 = self.v1.detach().contiguous(), self.v2.detach().contiguous(), self.v3.detach().contiguous()

        # Per-head RMSNorm and merge heads
        lact_o = lact_o.transpose(1, 2).reshape(B, T, H * dh)
        lact_o = self.rms_out(lact_o)
        return lact_o

    def _ttt_update(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        w3: torch.Tensor,
        lr1: torch.Tensor,
        lr2: torch.Tensor,
        lr3: torch.Tensor,
        use_muon: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Update fast weights. Supports unbatched [H,...] weights.

        If weights have shape [H,...], gradients are aggregated over batch and time.
        """
        # Unbatched weights per head
        # Forward with k
        gate_before = torch.einsum("bhtd,hdm->bhtm", k, w1)
        hidden_before = torch.einsum("bhtd,hdm->bhtm", k, w3)
        hidden = F.silu(gate_before) * hidden_before
        # Backward signals
        dhidden = torch.einsum("bhtd,hdm->bhtm", v, w2.transpose(-1, -2))
        dhidden_before = dhidden * F.silu(gate_before)
        dgate = dhidden * hidden_before
        sig = torch.sigmoid(gate_before)
        dgate_before = dgate * sig * (1 + gate_before * (1 - sig))

        error = v - torch.einsum("bhtm,hmd->bhtd", hidden, w2)
        # Gradients aggregated across batch/time
        w2_grad = -torch.einsum("bhtm,bhtd->hmd", hidden, error * lr2)
        w1_grad = -torch.einsum("bhtd,bhtm->hdm", k * lr1, dgate_before)
        w3_grad = -torch.einsum("bhtd,bhtm->hdm", k * lr3, dhidden_before)

        # Momentum (if enabled via cfg.ttt_momentum)
        beta = getattr(self.cfg, "ttt_momentum", 0.0)
        if beta and beta > 0.0:
            self.v1 = (beta * self.v1 + w1_grad).contiguous()
            self.v2 = (beta * self.v2 + w2_grad).contiguous()
            self.v3 = (beta * self.v3 + w3_grad).contiguous()
            w1_upd, w2_upd, w3_upd = self.v1, self.v2, self.v3
        else:
            w1_upd, w2_upd, w3_upd = w1_grad, w2_grad, w3_grad

        def renorm(prev, grad):
            new = prev + grad
            return new / (new.norm(dim=-1, keepdim=True) + 1e-6) * (prev.norm(dim=-1, keepdim=True) + 1e-6)

        w1 = renorm(w1, w1_upd)
        w2 = renorm(w2, w2_upd)
        w3 = renorm(w3, w3_upd)
        return w1, w2, w3


class TTTBlock(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.lact = LaCTLayer(cfg)
        self.use_mlp = cfg.use_mlp
        if self.use_mlp:
            self.ln2 = nn.LayerNorm(cfg.d_model)
            self.mlp = nn.Sequential(
                nn.Linear(cfg.d_model, cfg.d_mlp),
                nn.GELU(),
                nn.Linear(cfg.d_mlp, cfg.d_model),
            )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        rope: Tuple[torch.Tensor, torch.Tensor],
        *,
        position_ids: Optional[torch.Tensor] = None,
        ttt_config: Optional[list] = None,
        fast_weight: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
        use_muon: bool = False,
    ) -> torch.Tensor:
        x = x + self.lact(
            self.ln1(x), 
            attn_mask, 
            rope, 
            position_ids=position_ids, 
            ttt_config=ttt_config, 
            fast_weight=fast_weight,
            use_muon=use_muon,
        )
        if self.use_mlp:
            x = x + self.mlp(self.ln2(x))
        return x


class GPT_TTT(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(TTTBlock(cfg) for _ in range(cfg.n_layers))
        self.ln_f = nn.LayerNorm(cfg.d_model)
        if cfg.tie_output:
            self.out = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        else:
            self.out = nn.Linear(cfg.d_model, cfg.vocab_size, bias=True)

    def forward(
        self,
        input_ids: torch.Tensor,
        attn_mask_1d: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        *,
        fast_weight: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
        ttt_config: Optional[list] = None,
        use_muon: bool = False,
    ) -> torch.Tensor:
        """Forward through GPT_TTT.

        - input_ids: [B, T]
        - attn_mask_1d: [B, T] nonzero=valid; if None, must use varlen (B==1 with position_ids)
        - position_ids: [B, T] int64, optional; when provided with B==1 and CUDA+flash, varlen is used in LaCT
        - returns: logits [B, T, V]
        """
        B, T = input_ids.shape
        x = self.tok(input_ids)

        # Build RoPE tables
        max_position = int(position_ids.max().item()) + 1 if position_ids is not None else T
        cos, sin = build_rope_frequencies(max_position, self.cfg.d_head, self.cfg.rope_base, x.device)

        # Build additive bias if provided
        attn_mask_4d = None
        if attn_mask_1d is not None:
            attn_mask_4d = build_4d_attn_mask(attn_mask_1d, sliding_window=self.cfg.sliding_window, dtype=x.dtype)

        # Blocks
        for blk in self.blocks:
            x = blk(
                x,
                attn_mask_4d,
                (cos, sin),
                position_ids=position_ids,
                ttt_config=ttt_config,
                fast_weight=fast_weight,
                use_muon=use_muon,
            )

        x = self.ln_f(x)
        logits = self.out(x)
        if self.cfg.tie_output:
            # tie output to token embedding weights if using biasless output
            self.out.weight = self.tok.weight
        return logits
