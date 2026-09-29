"""LoRA-ViT for 12-lead ECG (manuscript section 3.3, Figures 1 and 2).

Forward path, shapes for the default configuration:

    S      (B, 12, 5000)   denoised 12-lead segment
    phi(S) (B, 50, 128)    two-stage Conv1d tokeniser         eqs. (4)-(6)
    Z_0    (B, 51, 128)    [CLS ; X + P] then dropout p=0.1   eq. (7)
    Z_l    (B, 51, 128)    8 x pre-norm LoRA encoder blocks   eqs. (8)-(9)
    Z_L    (B, 51, 128)    final LayerNorm                    eq. (12)
    y_hat  (B, 7)          W_c . Dropout(h_cls) + b_c         eq. (11)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import LoRAConfig, ModelConfig
from .lora import ParameterAccounting, count_parameters, freeze_backbone, inject_lora


class DropPath(nn.Module):
    """Stochastic depth, per sample (Table 2: 'stochastic depth + dropout')."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep)
        return x * mask / keep

    def extra_repr(self) -> str:
        return f"drop_prob={self.drop_prob:.3f}"


class HierarchicalPatchTokeniser(nn.Module):
    """Two-stage CNN tokeniser, equations (4)-(6).

        H1 = sigma(Conv1d^{12->64}_{k1,s1}(S))
        H2 =       Conv1d^{64->128}_{k2,s2}(H1)
        X  = phi(S) in R^{N x D},  N = 50, D = 128

    Stage 1 uses kernel = stride = patch_len // 2 (non-overlapping 50-sample sub-patches,
    100 per record); stage 2 pools adjacent pairs with kernel = stride = 2, giving the
    50 inter-lead patch tokens the manuscript describes. "Hierarchical" is exactly this:
    lead mixing happens at stage 1, temporal grouping at stage 2.
    """

    def __init__(self, in_chans: int = 12, embed_dim: int = 128, patch_len: int = 100) -> None:
        super().__init__()
        if patch_len % 2 != 0:
            raise ValueError(f"patch_len must be even, got {patch_len}")
        self.patch_len = patch_len
        self.embed_dim = embed_dim
        half = embed_dim // 2
        self.conv1 = nn.Conv1d(in_chans, half, kernel_size=patch_len // 2, stride=patch_len // 2)
        self.act = nn.GELU()
        self.conv2 = nn.Conv1d(half, embed_dim, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T) -> (B, N, D)
        h = self.act(self.conv1(x))
        h = self.conv2(h)
        return h.transpose(1, 2)

    def n_patches(self, seq_len: int) -> int:
        return seq_len // self.patch_len


class Attention(nn.Module):
    """Multi-head self-attention, equation (10). `qkv` and `proj` are LoRA targets."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.1,
        proj_drop: float = 0.1,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"embed_dim {dim} is not divisible by num_heads {num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop_p = attn_drop
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self._last_attn: Optional[torch.Tensor] = None
        self.store_attention = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.store_attention:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            self._last_attn = attn.detach()
            attn = self.attn_drop(attn)
            out = attn @ v
        else:
            out = F.scaled_dot_product_attention(
                q, k, v, dropout_p=self.attn_drop_p if self.training else 0.0
            )

        out = out.transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(out))


class MLP(nn.Module):
    """Feed-forward block: FC1 -> GELU -> dropout -> FC2 -> dropout. LoRA targets fc1/fc2."""

    def __init__(self, dim: int, hidden: int, drop: float = 0.1) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden, dim)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop2(self.fc2(self.drop1(self.act(self.fc1(x)))))


class EncoderBlock(nn.Module):
    """Pre-norm transformer block, equations (8)-(9)."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        drop: float = 0.1,
        attn_drop: float = 0.1,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), drop=drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class LoRAViT(nn.Module):
    """The full model of Figure 1."""

    def __init__(self, cfg: Optional[ModelConfig] = None) -> None:
        super().__init__()
        cfg = cfg or ModelConfig()
        self.cfg = cfg
        self.tokeniser = HierarchicalPatchTokeniser(
            in_chans=cfg.in_chans, embed_dim=cfg.embed_dim, patch_len=cfg.patch_len
        )
        self.n_patches = cfg.seq_len // cfg.patch_len

        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches, cfg.embed_dim))
        self.cls_token = (
            nn.Parameter(torch.zeros(1, 1, cfg.embed_dim)) if cfg.use_cls_token else None
        )
        self.pos_drop = nn.Dropout(cfg.drop_rate)

        dpr = torch.linspace(0, cfg.drop_path_rate, cfg.depth).tolist()
        self.blocks = nn.ModuleList(
            [
                EncoderBlock(
                    cfg.embed_dim, cfg.num_heads, cfg.mlp_ratio,
                    drop=cfg.drop_rate, attn_drop=cfg.attn_drop_rate, drop_path=dpr[i],
                )
                for i in range(cfg.depth)
            ]
        )
        self.norm = nn.LayerNorm(cfg.embed_dim)
        self.head = nn.Sequential(
            nn.Dropout(cfg.drop_rate), nn.Linear(cfg.embed_dim, cfg.n_classes)
        )
        self.apply(self._init_module)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.cls_token is not None:
            nn.init.trunc_normal_(self.cls_token, std=0.02)

    @staticmethod
    def _init_module(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)
        elif isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    # -- forward -----------------------------------------------------------
    def forward_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 12, T) -> (B, 1+N, D) after the encoder stack and final LayerNorm."""
        if x.ndim == 4:  # (B, 1, 12, T) tolerated
            x = x.squeeze(1)
        if x.ndim != 3 or x.shape[1] != self.cfg.in_chans:
            raise ValueError(
                f"expected (B, {self.cfg.in_chans}, {self.cfg.seq_len}), got {tuple(x.shape)}"
            )
        z = self.tokeniser(x)                       # (B, N, D)
        if z.shape[1] != self.n_patches:
            raise ValueError(
                f"tokeniser produced {z.shape[1]} tokens, expected {self.n_patches}; "
                f"seq_len={x.shape[-1]} is not {self.cfg.seq_len}"
            )
        z = z + self.pos_embed
        if self.cls_token is not None:
            z = torch.cat([self.cls_token.expand(z.shape[0], -1, -1), z], dim=1)
        z = self.pos_drop(z)
        for blk in self.blocks:
            z = blk(z)
        return self.norm(z)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Pooled global descriptor used for t-SNE (CLS token, eq. 12)."""
        z = self.forward_tokens(x)
        return z[:, 0] if self.cls_token is not None else z.mean(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x))

    # -- introspection -----------------------------------------------------
    def set_store_attention(self, flag: bool) -> None:
        for blk in self.blocks:
            blk.attn.store_attention = flag

    def attention_maps(self) -> List[torch.Tensor]:
        return [b.attn._last_attn for b in self.blocks if b.attn._last_attn is not None]


def build_model(
    model_cfg: Optional[ModelConfig] = None,
    lora_cfg: Optional[LoRAConfig] = None,
) -> Tuple[LoRAViT, ParameterAccounting]:
    """Construct the model, inject LoRA, freeze the backbone, and account for parameters."""
    model_cfg = model_cfg or ModelConfig()
    lora_cfg = lora_cfg or LoRAConfig()
    model = LoRAViT(model_cfg)
    if lora_cfg.enabled:
        inject_lora(model, lora_cfg)
        if lora_cfg.freeze_backbone:
            freeze_backbone(model, lora_cfg)
    return model, count_parameters(model, lora_cfg)


__all__ = [
    "LoRAViT", "EncoderBlock", "Attention", "MLP", "DropPath",
    "HierarchicalPatchTokeniser", "build_model",
]
