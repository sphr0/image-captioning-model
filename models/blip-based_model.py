"""
blip_model.py
BLIP-base image captioning

From-scratch implementation of the MED (Multimodal mixture of Encoder-Decoder).
The three pre-training functionalities are all present so the architecture is
the same, but captioning inference routes only through the
image-grounded text decoder (LM mode):

Design choices:
  * CapFilt is excluded since it's a data bootstrapping procedure.
  * Encoder & decoder share embeddings (cross-attention and FFN), self-attn not shared (bidirectional vs causal).
  * ITM hard negatives are extracted from ITC similarity matrix.
  * ViT tower is pre-norm, BERT text tower is post-norm.
  * No pixel preprocessing (mean/std normalization). the module consumes normalized pixel_values.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


# CONFIG

@dataclass
class BLIPConfig:
    # ViT-B/16
    image_size: int = 384
    patch_size: int = 16
    vit_width: int = 768
    vit_layers: int = 12
    vit_heads: int = 12
    # BERT-base
    dropout: float = 0.1
    mpl_ratio: int = 4
    # heads / misc will add  later... [TODO] 


# private functions

def _sa_mask(pad, mode, L, device):
  # if given padding, we only add dims for head and query to pad, else we make the whole thing
  key = (torch.ones(1, 1, 1, L, dtype=torch.bool, device=device) if pad is None else pad.bool()[:, None, None, :])
  
  if mode == "multimodal_dec":
    causal = torch.ones(L, L, device=device).tril().bool()[None, None] # [1, 1, L, L]
    return key & causal # both cuasal and padding in a bool tensor
  return key


def _cross_mask(img_mask): #
  return None if img_mask is None else img_mask.bool()[:, None, None, :]

class MHA(nn.Module):
  def __init__(self, dim, n_heads, dropout):
    super().__init__()
    assert dim % n_heads == 0
    self.h, self.d = n_heads, dim // n_heads # d = dim per head
    self.q = nn.Linear(dim, dim)
    self.kv = nn.Linear(dim, dim * 2)
    self.proj = nn.Linear(dim, dim)
    self.p = dropout # p = dropout probability

  def _split(self, x, B):
    return x.view(B, -1, self.h, self.d).transpose(1, 2) # [B,h,L,d]
  
  def forward(self, x, kv=None, attn_mask=None):
    B = x.size(0)
    ctx = x if kv is None else kv # supports both SA and cross
    q = self._split(self.q(x), B)
    k, v = (t for t in self.kv(ctx).chunk(2, dim=-1)) # seperates k from v
    o = F.scaled_dot_product_attention(
      q, k, v, attn_mask=attn_mask, dropout_p=(self.p if self.training else 0.0))
    o = o.transpose(1, 2).reshape(B, -1, self.h * self.d) # undo the _split
    return self.proj(o)


class PatchEmbed(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.proj = nn.Conv2d(in_channels=3,
         out_channels=cfg.vit_width,
          kernel_size=cfg.patch_size,
           stride=cfg.patch_size) # 14x14 patches

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2) #[B,3,H,W] -> [B,N,C] B x 196 x 768 


class ViTBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        w = cfg.vit_width
        self.norm1, self.norm2 = nn.LayerNorm(w), nn.LayerNorm(w)
        self.attn = MHA(dim=w, n_heads=cfg.vit_heads, dropout=cfg.dropout)
        self.mlp = nn.Sequential(nn.Linear(in_features=w, out_features=cfg.mlp_ratio * w),
                                nn.GELU(),
                                nn.Linear(in_features=cfg.mlp_ratio * w, out_features=w))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x)) # pre & post-norm


class VisionTransformer(nn.Module): # [B,3,H,W] -> [B,N+1,W] img embeds
    def __init__(self, cfg):
        super().__init__()
        n = (cfg.image_size // cfg.patch_size) ** 2
        self.patch_embed = PatchEmbed(cfg)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.vit_width))
        self.pos_embed = nn.Parameter(torch.zeros(1, n + 1, cfg.vit_width))
        self.blocks = nn.ModuleList([ViTBlock(cfg) for _ in range(cfg.vit_layers)]) # 12 ViT layers
        self.norm = nn.LayerNorm(cfg.vit_width)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, pixel_values):
        B = pixel_values.size(0)
        x = self.patch_embed(pixel_values)
        x = torch.cat([self.cls_token.expand(B, -1, -1), x], 1) + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)


