
"""
vit_gpt2_model.py

ViT-GPT2 image captioning, two parallel implementations:

  PART 1 - from scratch (educational): ViT-B/16 encoder -> linear cross-attention
    bridge -> GPT-2 decoder with PER-LAYER cross-attention. Not meant
    for training.

  PART 2 - transfer learning (actual use): VisionEncoderDecoderModel with pretrained
    ViT + GPT-2. HF injects the cross-attention.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==================================================================
# PART 1 - FROM-SCRATCH IMPLEMENTATION
# ==================================================================


# vision Transformer
# =============================================

# ViT hyperparameters

@dataclass
class ViTConfig:
    image_size: int = 224
    patch_size: int = 16
    in_chans: int = 3
    dim: int = 768
    depth: int = 12
    heads: int = 12
    mlp_ratio: float = 4.0
    drop: float = 0.0


# MHA (modular design for encoder and decoder self-attn AND the decoder cross-attn )

class MultiHeadAttention(nn.Module):
    """Single module for self- and cross-attn. Can run on 8GB VRAM.
     causal=True -> decoder self-attn; pass x_kv for
    cross-attn. key_padding_mask: (B, Tk) bool, True = keep."""

    def __init__(self, dim, heads, drop=0.0):
        super().__init__()
        assert dim % heads == 0
        self.heads = heads
        self.head_dim = dim // heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
        self.drop = drop

    def forward(self, x_q, x_kv=None, causal=False, key_padding_mask=None):
        x_kv = x_q if x_kv is None else x_kv
        B, Tq, C = x_q.shape
        Tk = x_kv.shape[1]
        q = self.q(x_q).view(B, Tq, self.heads, self.head_dim).transpose(1, 2)
        k = self.k(x_kv).view(B, Tk, self.heads, self.head_dim).transpose(1, 2)
        v = self.v(x_kv).view(B, Tk, self.heads, self.head_dim).transpose(1, 2)
        p = self.drop if self.training else 0.0

        if key_padding_mask is None:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=causal, dropout_p=p)
        else:
            # Combine padding (+ optional causal) into one additive mask; SDPA forbids
            # is_causal together with an explicit attn_mask.
            mask = torch.zeros(B, 1, Tq, Tk, device=q.device, dtype=q.dtype)
            mask.masked_fill_(~key_padding_mask[:, None, None, :], float("-inf"))
            if causal:
                cm = torch.triu(torch.ones(Tq, Tk, device=q.device, dtype=torch.bool), 1)
                mask.masked_fill_(cm, float("-inf"))
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=p)

        out = out.transpose(1, 2).reshape(B, Tq, C)
        return self.proj(out)


# Standard MLP with GELU

class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio, drop=0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(F.gelu(self.fc1(x)))))


# ViT Block

class ViTBlock(nn.Module):
    """Pre-norm(for both MHA and MLP) encoder block."""

    def __init__(self, cfg: ViTConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.dim)
        self.attn = MultiHeadAttention(cfg.dim, cfg.heads, cfg.drop)
        self.norm2 = nn.LayerNorm(cfg.dim)
        self.mlp = MLP(cfg.dim, cfg.mlp_ratio, cfg.drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ViT full model

class VisionTransformer(nn.Module):
    def __init__(self, cfg: ViTConfig):
        super().__init__()
        # patch -> flatten -> linear projection
        self.n_patches = (cfg.image_size // cfg.patch_size) ** 2
        self.patch_embed = nn.Conv2d(cfg.in_chans, cfg.dim, cfg.patch_size, cfg.patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches + 1, cfg.dim))
        self.pos_drop = nn.Dropout(cfg.drop)
        self.blocks = nn.ModuleList([ViTBlock(cfg) for _ in range(cfg.depth)])
        self.norm = nn.LayerNorm(cfg.dim)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, pixel_values): # (B, 3, 224, 224)
        B = pixel_values.shape[0]
        x = self.patch_embed(pixel_values).flatten(2).transpose(1, 2)   # (B, 196, dim)
        x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1) + self.pos_embed # (197 tokens)
        x = self.pos_drop(x)
        for blk in self.blocks: # Encode entire image as a sequence of interacting tokens
            x = blk(x)
        return self.norm(x) # (B, 197, dim) — full seq is the cross-attn memory


# ===================================================