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
    mlp_ratio: int = 4

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
        self.pos_ids = self.register_buffer(
            "pos_ids", 
            torch.arange(max_len).unsqueeze(0), 
            persistent=False
        )
        self.ln = nn.LayerNorm(hidden_dim, eps= 1e-12)
        self.drop = nn.Dropout(drop_p)

    def forward(self, ids):
        seq_len = ids.shape[1]
        x = self.word_embed(ids) + self.pos_embed(self.pos_ids[:, :seq_len])
        return self.drop(self.ln(x))


class GITLayer(nn.Module):
    def __init__(self, cfg: GITConfig):
        super().__init__()

        self.attention = MHA(cfg.hidden_size, cfg.num_heads)
        self.attn_drop = nn.Dropout(0.0) # since we won't train it, hardcoded to 0.0
        self.post_ln1 = nn.LayerNorm(cfg.hidden_size, eps=1e-12)
        self.ffn = nn.Sequential(
            nn.Linear(cfg.hidden_size, cfg.hidden_size * cfg.mlp_ratio),
            nn.GELU(),
            nn.Linear(cfg.hidden_size * cfg.mlp_ratio, cfg.hidden_size),
            nn.Dropout(0.0)
        )
        self.post_ln2 = nn.LayerNorm(cfg.hidden_size, eps=1e-12)

    def forward(self, x, attn_mask):
        x = self.post_ln1(self.attn_drop(self.attention(x, attn_mask)) + x)
        return self.post_ln2(self.ffn(x) + x)

class GITEncoder(nn.Module):
    def __init__(self, cfg: GITConfig):
        super().__init__()

        self.blocks = nn.ModuleList([GITLayer(cfg) for _ in range(cfg.num_layers)])

    def forward(self, x, attn_mask):
        for blk in self.blocks:
            x = blk(x, attn_mask)
        return x


# specifically uselful when the vision hidden dim doesn't match the text hidden dim.
class GITProjection(nn.Module):
    def __init__(self, vision_dim, txt_dim):
        super().__init__()

        self.linear = nn.Linear(vision_dim, txt_dim)
        self.ln = nn.LayerNorm(txt_dim, eps=1e-05)

    def forward(self, x):
        return self.ln(self.linear(x))


class GITFromScratch(nn.Module):
    def __init__(self, cfg: GITConfig):
        super().__init_()
        self.cfg = cfg

        self.vision = VisionTransformer(cfg.vision)
        self.proj = GITProjection(cfg.vision.hidden_size, cfg.hidden_size)
        self.txt_embed = GITEmbeddings(cfg)
        self.git_encoder = GITEncoder(cfg)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size)

    def forward(self, pixel_values=None, ids=None, pad_mask=None, vision_tokens=None, labels=None):
        # catch if we're given both or none
        # training -> vision_token is None, generating -> pixel_values is None
        assert (pixel_values is None) != (vision_tokens is None), "pass either vission_tokens or pixel_values"

        if pad_mask is None: # if no padding given, make all True mask
            pad_mask = torch.ones_like(ids, dtype=torch.bool)

        if vision_tokens is None:
            vision_tokens = self.proj(self.vision(pixel_values))

        txt_tokens = self.txt_embed(ids)
        x = torch.cat((vision_tokens, txt_tokens), dim=1)

        img_len = vision_tokens.shape[1]
        mask = _attn_mask(img_len, pad_mask)
        hidden = self.git_encoder(x, mask) # prefix gets self-attended [B, seq_len, hid_size]
        
        # drop the img prefix before the vocab proj since not needed for the task
        txt_hidden = hidden[:, img_len:]
        logits = self.lm_head(txt_hidden) # [B, text_seq_len, vocab_size]

        if labels is None: # if not training
            return logits

        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)), # flatten all predicted token positions
            labels[:, 1:].reshape(-1), # same with all labels
            ignore_index=self.cfg.pad_token_id
        )
        return loss, logits

    @torch.inference_mode()
    def generate(self, pixel_values, max_new_tokens=20):
        bos_id = self.cfg.bos_token_id
        B = pixel_values.shape[0]
        # decrease by 1 for the BOS token, which is accounted for with max_new_tokens arg
        max_new_tokens = min(max_new_tokens, self.cfg.max_text_length - 1)
        ids = torch.full((B, 1), bos_id, dtype=torch.long, device=pixel_values.device)
        # 1. seed the sequence with BOS, one per batch item -> ids [B, 1]
        is_done = torch.zeros(B, dtype=torch.bool, device=pixel_values.device)
        # 2. loop up to max_new_tokens:

        # To prevent calculating the same vision tokens on each loop
        vision_tokens = self.proj(self.vision(pixel_values))

        for _ in range(max_new_tokens):
            # <NOTE> We ignore the two-output scenario for forward since we don't train
            # but implement something to handle that later...
            logits = self(ids=ids, vision_tokens=vision_tokens) # -> logits [B, cur_len, V]
            # take the last position -> [B, V]
            next_logits = logits[:, -1, :]
            # pick a token id -> [B]
            next_token = torch.argmax(next_logits, 1)
            # replace with padding if seq has ended already
            next_token = torch.where(is_done, self.cfg.pad_token_id, next_token)
            # append to ids -> [B, cur_len+1]
            ids = torch.cat((ids, next_token[:, None]), dim=1)
            # stop if every sequence has emitted EOS
                # switch is_done to True on any batch that ends with EOS
            is_done |= (next_token == self.cfg.eos_token_id)
                # if all is_done, break out of loop
            if is_done.all():
                break
        # 3. return ids
        return ids


# <NOTE> change VisionTransformer cfg -> cfg.vision in next commit
# why is tie_word_embeddings equal to False?