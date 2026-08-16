"""
GIT Image Captioning - HF implementation (both from scratch and transfered)
architecture:
    1. Visual Encoder: CLIP ViT-B/16
    2. Text Decoder: BERT-like (random initialization)

The encoder seen in the paper's GIT model (Florence/CoSwin) is not 
publically available but the GIT-base model uses a CLIP ViT-B/16, which
has differences: layerNorm right after patch+position embedding and a 
postnorm at the end, and QuickGELU instead of GELU in MLP.
The decoder is a BERT-like transformer with weights randomly initiallized. 
In addition it uses a prefix-causal mask with LM training objective.

In the Transfered models, we have both Git-B and Git-L since Git-L could
fit in even with hardware limitations.
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field

@dataclass
class VisionConfig:
    image_size: int = 224
    patch_size: int = 16
    num_channels: int = 3

    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12

    mlp_ratio: int = 4


@dataclass
class GITConfig:
    vocab_size: int = 30_522
    max_text_length: int = 1024
    hidden_size: int = 768
    num_layers: int = 6
    num_heads: int = 12
    mlp_ratio: float = 4.0

    pad_token_id: int = 0
    bos_token_id: int = 101
    eos_token_id: int = 102

    vision: VisionConfig = field(default_factory=VisionConfig)

# ===============================
# PRIVATE FUNCTIONS

# <NOTE> add dropout but set to zero
# Mask function
def _attn_mask(img_len, txt_pad_mask):
    """
    creates GIT attention mask

    Args:
        img_len: number of image-prefix tokens
        txt_pad_mask: padding mask for each 
            sample in the batch (T=true token).
    
    Returns: Bool mask of shape [B, 1, L, L] (T=can attend)
    """

    device = txt_pad_mask.device
    batch_size, txt_len = txt_pad_mask.shape # txt_len includes padding part too
    total_seq_len = img_len + txt_len

    # all queries can attend to img
    img_key_mask = torch.ones((total_seq_len, img_len), dtype=torch.bool, device=device)
    # img -> txt = False, txt -> txt = Causal
    txt_key_mask = torch.ones(total_seq_len, txt_len, dtype=torch.bool, device=device).tril_(diagonal=-img_len)
    structural_mask = torch.cat([img_key_mask, txt_key_mask], dim=1) # [L, L]

    # All img token positions are valid
    img_valid = torch.ones((batch_size, img_len), dtype=torch.bool, device=device)
    # valid img tokens + valid txt tokens in the batch
    key_valid = torch.cat([img_valid, txt_pad_mask.bool()], dim=1) # [B, L]

    mask = (structural_mask[None, :, :] & key_valid[:, None, :])
    return mask[:, None, :, :]


def _quick_gelu(x):
    return x * torch.sigmoid(1.702 * x)

# ==================================================
# VISUAL ENCODER (CLIP ViT-B/16)
# ==================================================

class MHA(nn.Module):
    """
    GIT-specific MHA, only does self-attention (no causal)
    since _attn_mask handles that.
    """

    def __init__(self, dim, h, drop=0.0):
        super().__init__()
        assert dim % h == 0

        self.h, self.h_dim = h, dim // h
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)
        self.drop = drop

    def _split(self, x):
        B, T, C = x.shape # number of batch, seq length and channels 
        return x.view(B, T, self.h, self.h_dim).transpose(1, 2) # [B, H, T, h_dim]
    
    def forward(self, x, attn_mask=None):
        B, T, C = x.shape

        q = self._split(self.q(x))
        k, v = self.kv(x).chunk(2, dim=-1) # last dim is channels. split to k and v.
        k = self._split(k)
        v = self._split(v)
        p = self.drop if self.training else 0.0

        out = F.scaled_dot_product_attention(query=q, 
        key=k, 
        value=v, 
        attn_mask=attn_mask, 
        dropout_p=p) # [B, h, T, h_dim]

        out = out.transpose(1, 2).reshape(B, T, C) # concating attn heads back into one
        return self.proj(out) # return projection of the heads


class MLP(nn.Module):
    def __init__(self, dim, ratio, act=F.gelu, drop=0.0):
        super().__init__()

        self.dense_in = nn.Linear(dim, dim * ratio)
        self.dense_out = nn.Linear(dim * ratio, dim)
        self.act = act
        self.dropout = nn.Dropout(drop)

    def forward(self, x):
        h = self.act(self.dense_in(x))
        return self.dropout(self.dense_out(h)) # residual and post-norm NOT internalized


# <NOTE> Not sure if the cfg passes to this correctly. Must check later
class ViTBlock(nn.Module):
    def __init__(self, cfg: VisionConfig):
        super().__init__()

        self.ln1, self.ln2 = nn.LayerNorm(cfg.hidden_size, eps=1e-5), nn.LayerNorm(cfg.hidden_size, eps=1e-5)
        self.attn = MHA(dim=cfg.hidden_size, h=cfg.num_heads, drop=0.0)
        self.mlp = MLP(dim=cfg.hidden_size, ratio=cfg.mlp_ratio, act=_quick_gelu, drop=0.0)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))

class PatchProjection(nn.Module):
    def __init__(self, num_channels, hidden_dim, patch_size):
        super().__init__()

        self.proj = nn.Conv2d(in_channels=num_channels, 
                              out_channels=hidden_dim, 
                              kernel_size=patch_size, 
                              stride=patch_size,
                              bias=False)

    def forward(self, x):
        return self.proj(x)


class VisionTransformer(nn.Module):
    def __init__(self, cfg: GITConfig):
        super().__init__()
        assert cfg.vision.image_size % cfg.vision.patch_size == 0
        num_patches = (cfg.vision.image_size // cfg.vision.patch_size) ** 2

        self.patch_embedding = PatchProjection(cfg.vision.num_channels, 
                                               cfg.vision.hidden_size, 
                                               cfg.vision.patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.vision.hidden_size))
        self.pos_embedding = nn.Parameter(torch.zeros(1, num_patches + 1, cfg.vision.hidden_size))
        self.pre_ln = nn.LayerNorm(cfg.vision.hidden_size, eps=1e-5)
        self.blocks = nn.ModuleList([ViTBlock(cfg.vision) for _ in range(cfg.vision.num_layers)])
        self.post_ln = nn.LayerNorm(cfg.vision.hidden_size, eps=1e-5)

    def forward(self, x): # [B, num_channels, img_size, img_size]
        B = x.shape[0]

        x = self.patch_embedding(x).flatten(2).transpose(1, 2)
        x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1) + self.pos_embedding
        x = self.pre_ln(x)
        for blk in self.blocks:
            x = blk(x)
        return self.post_ln(x)


# ==================================================
# Text Decoder (Bert-like)
# ==================================================

# EMBEDDINGS
class GITEmbeddings(nn.Module):
    def __init__(self, vocab_size, max_len, hidden_dim, pad_id, drop_p=0.0):
        super().__init__()

        self.word_embed = nn.Embedding(vocab_size, hidden_dim, pad_id)
        self.pos_embed = nn.Embedding(max_len, hidden_dim)
        self.pos_ids = nn.Buffer(
            torch.arange(max_len).expand((1, -1)),
            persistent=False
        )
        self.ln = nn.LayerNorm(hidden_dim, eps= 1e-12)
        self.drop = nn.Dropout(drop_p)

    def forward(self, ids):
        seq_len = ids.shape(1)
        x = self.word_embed(ids) + self.pos_embed(self.pos_ids[:, :seq_len])
        return self.drop(self.ln(x))


# <NOTE> 5. Decoder Blocks
# <NOTE> 6. LM head + loss
# <NOTE> 6. Generate func
# why is tie_word_embeddings equal to False?