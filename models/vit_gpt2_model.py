
"""
ViT-GPT2 image captioning, two parallel implementations:

  1. from scratch: ViT-B/16 encoder -> linear cross-attn
    bridge -> GPT-2 decoder with PER-LAYER cross-attn. Not meant
    for training.

  2. transfer learning (actual use): VisionEncoderDecoderModel with pretrained
    ViT + GPT-2. HF injects the cross-attn.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


# ==================================================================
# PART 1 - FROM-SCRATCH IMPLEMENTATION
# ==================================================================


# ViT & GPT2 HYPERPARAMETERS

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

@dataclass
class GPT2Config:
    vocab_size: int = 50257
    n_positions: int = 1024
    dim: int = 768
    depth: int = 12
    heads: int = 12
    mlp_ratio: float = 4.0
    drop: float = 0.1

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


# ViT BLOCK

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


# ViT FULL MODEL

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


# CROSS-ATTN BRIDGE
# ==================================

class CrossAttentionBridge(nn.Module):
    """Projects encoder hidden states into the decoder's cross-attn KV space. Dim:
    768->768. the thing that must be trained when both towers are frozen, rather than a dimensionality fix."""

    def __init__(self, enc_dim, dec_dim):
        super().__init__()
        self.proj = nn.Linear(enc_dim, dec_dim)
        self.norm = nn.LayerNorm(dec_dim)

    def forward(self, enc):
        return self.norm(self.proj(enc))


# GPT-2 DECODER
# ===================================

class GPT2Block(nn.Module):
    """GPT-2 block with cross-attn inserted between masked self-attn and MLP.
    Sublayer order per block: self-attn (causal) -> cross-attn -> MLP, all pre-norm."""

    def __init__(self, cfg: GPT2Config):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.dim)
        self.self_attn = MultiHeadAttention(cfg.dim, cfg.heads, cfg.drop)
        self.ln_cross = nn.LayerNorm(cfg.dim)
        self.cross_attn = MultiHeadAttention(cfg.dim, cfg.heads, cfg.drop)
        self.ln2 = nn.LayerNorm(cfg.dim)
        self.mlp = MLP(cfg.dim, cfg.mlp_ratio, cfg.drop)

    def forward(self, x, memory, mem_mask=None):
        x = x + self.self_attn(self.ln1(x), causal=True)
        x = x + self.cross_attn(self.ln_cross(x), memory, key_padding_mask=mem_mask)
        x = x + self.mlp(self.ln2(x))
        return x


class GPT2Decoder(nn.Module):
    def __init__(self, cfg: GPT2Config):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.wpe = nn.Embedding(cfg.n_positions, cfg.dim)
        self.drop = nn.Dropout(cfg.drop)
        self.blocks = nn.ModuleList([GPT2Block(cfg) for _ in range(cfg.depth)])
        self.ln_f = nn.LayerNorm(cfg.dim)
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight # weight tying

    def forward(self, input_ids, memory, mem_mask=None):
        T = input_ids.shape[1]
        pos = torch.arange(T, device=input_ids.device)
        x = self.drop(self.wte(input_ids) + self.wpe(pos)[None])
        for blk in self.blocks:
            x = blk(x, memory, mem_mask)
        return self.lm_head(self.ln_f(x)) # (B, T, vocab)


# ===================================

class ViTGPT2FromScratch(nn.Module):

    # build the 3 main pieces and run _init on every layer
    def __init__(self, vit_cfg=ViTConfig(), gpt_cfg=GPT2Config()):
        super().__init__()
        self.encoder = VisionTransformer(vit_cfg)
        self.bridge = CrossAttentionBridge(vit_cfg.dim, gpt_cfg.dim)
        self.decoder = GPT2Decoder(gpt_cfg)
        self.apply(self._init)
 

    # if linear layer, fill weight with random num AND set bias to 0
    # if lookup table (Embedding), fill with random num
    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)
 
    def forward(self, pixel_values, input_ids, labels=None):
        """labels: copy of input_ids with pad positions set to -100. The next-token
        shift is applied internally (teacher forcing)."""
        memory = self.bridge(self.encoder(pixel_values)) # run img thru enc and bridge
        logits = self.decoder(input_ids, memory) # run caption through dec along with memory(img features)
        loss = None
        if labels is not None: # calculate loss only on training (true captions)
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )
        return {"logits": logits, "loss": loss}
 
    @torch.no_grad()
    def generate(self, pixel_values, bos_id, eos_id, max_len=30,
                 temperature=1.0, top_k=None):
        """Un-cached decode (O(T^2) — fine for caption lengths). Memory is encoded once."""
        self.eval()
        memory = self.bridge(self.encoder(pixel_values)) # img -> enc -> bridge
        B = pixel_values.size(0) # no. of imgs in batch
        ids = torch.full((B, 1), bos_id, dtype=torch.long, device=pixel_values.device) # captions made of bos
        done = torch.zeros(B, dtype=torch.bool, device=pixel_values.device) # done flag per caption

        # Word generation
        for _ in range(max_len):
            logits = self.decoder(ids, memory)[:, -1]
            if temperature > 0: # random-sampling mode (best word has highest chance)
                logits = logits / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits = logits.masked_fill(logits < v[:, [-1]], float("-inf"))
                nxt = torch.multinomial(F.softmax(logits, dim=-1), 1)
            else:
                nxt = logits.argmax(-1, keepdim=True) # greedy mode
            nxt = nxt.masked_fill(done.unsqueeze(1), eos_id)
            ids = torch.cat([ids, nxt], dim=1)
            done |= nxt.squeeze(1) == eos_id
            if done.all():
                break
        return ids

# ===========================================================================
# PART 2 - HF MODEL
# ===========================================================================

from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
 
 
def build_vit_gpt2_pretrained(ckpt="nlpconnect/vit-gpt2-image-captioning", device="cpu"):
    """Encoder, decoder, AND cross-attention all pretrained. Use for inference and as
    the ViT-GPT2 entry in the three-way comparison."""
    model = VisionEncoderDecoderModel.from_pretrained(ckpt).to(device).eval()
    image_processor = ViTImageProcessor.from_pretrained(ckpt)
    tokenizer = AutoTokenizer.from_pretrained(ckpt)
    return model, image_processor, tokenizer
 
 
def build_vit_gpt2_for_finetune(
    encoder_ckpt="google/vit-base-patch16-224-in21k",
    decoder_ckpt="gpt2",
    device="cpu",
    freeze_encoder=True,
):
    """A fresh encoder-decoder from two pretrained towers. HF injects
    randomly-initialized cross-attention into GPT-2; both towers are 768-dim so no
    enc->dec projection needed. With freeze_encoder=True only
    cross-attn + GPT-2 train"""
    # Only need to train GPT2 and cross-attn layers
    
    model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(encoder_ckpt, decoder_ckpt)
    tokenizer = AutoTokenizer.from_pretrained(decoder_ckpt)
    image_processor = ViTImageProcessor.from_pretrained(encoder_ckpt)
 
    # GPT-2 ships without pad token
    tokenizer.pad_token = tokenizer.eos_token
    model.config.decoder_start_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
 
    if freeze_encoder:
        for p in model.encoder.parameters():
            p.requires_grad_(False)
 
    return model.to(device), image_processor, tokenizer
 
 
@torch.no_grad()
def caption(model, image_processor, tokenizer, images, device="cpu",
            max_length=30, num_beams=4, **gen):
    pixel_values = image_processor(images=images, return_tensors="pt").pixel_values.to(device)
    out = model.generate(pixel_values, max_length=max_length, num_beams=num_beams, **gen)
    return tokenizer.batch_decode(out, skip_special_tokens=True)