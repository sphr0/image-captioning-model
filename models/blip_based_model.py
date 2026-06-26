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

# ==========================
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
    vocab_size: int = 30524 # 30522+[DEC]+[ENC]
    text_width: int = 768
    text_layers: int = 12
    text_heads: int = 12
    max_text_len: int = 40
    # heads / misc
    embed_dim: int = 256 # ITC projection dim
    mlp_ratio: int = 4
    dropout: float = 0.1
    label_smoothing: float = 0.1
    pad_token_id: int = 0
    cls_token_id: int = 101
    eos_token_id: int = 102 # SEP
    bos_token_id: int = 30522 # decoder BOS
    enc_token_id: int = 30523 # ITM task token


# ==========================
# PRIVATE FUNCTIONS

def _sa_mask(pad, mode, L, device):
  # if given padding, we only add dims for head and query to pad, else we make the whole thing
  key = (torch.ones(1, 1, 1, L, dtype=torch.bool, device=device) if pad is None else pad.bool()[:, None, None, :])
  
  if mode == "multimodal_dec":
    causal = torch.ones(L, L, device=device).tril().bool()[None, None] # [1, 1, L, L]
    return key & causal # both cuasal and padding in a bool tensor
  return key

def _cross_mask(img_mask): #
  return None if img_mask is None else img_mask.bool()[:, None, None, :]

# ===================================================
# VISION TOWER
# ===================================================

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


# =====================================================
# TEXT TOWER
# =====================================================

class BertEmbeddings(nn.Module):
  def __init__(self, cfg):
    super().__init__()
    self.word = nn.Embedding(cfg.vocab_size,
                             cfg.text_width,
                             padding_idx=cfg.pad_token_id) # initial padding grad = 0 and no learning
    self.pos = nn.Embedding(cfg.max_text_len, 
                            cfg.text_width)
    self.norm = nn.LayerNorm(cfg.text_width)
    self.drop = nn.Dropout(cfg.dropout)
    self.register_buffer("pos_ids", # prevents re-creating the same pos_ids on each cycle
                         torch.arange(cfg.max_text_len)[None],
                         persistent=False)
  
  def forward(self, ids):
    L = ids.size(1)
    x = self.word(ids) + self.pos(self.pos_ids[:, :L])
    return self.drop(self.norm(x))


class BertLayer(nn.Module):
  """SA(bi or causal) -> cross attn -> FFN.
  """
  def __init__(self, cfg):
    super().__init__()
    w = cfg.text_width
    self.sa_bi = MHA(w, cfg.text_heads, cfg.dropout)
    self.sa_caus = MHA(w, cfg.text_heads, cfg.dropout)
    self.norm_sa = nn.LayerNorm(w)
    self.cross = MHA(w, cfg.text_heads, cfg.dropout)
    self.norm_ca = nn.LayerNorm(w)
    self.ffn = nn.Sequential(
      nn.Linear(w, cfg.mlp_ratio * w),
      nn.GELU(),
      nn.Linear(cfg.mlp_ratio * w, w))
    self.norm_ffn = nn.LayerNorm(w)

  def forward(self, x, mode, sa_mask=None, img=None, img_mask=None):
    sa = self.sa_caus if mode == "multimodal_dec" else self.sa_bi
    x = self.norm_sa(x + sa(x, attn_mask=sa_mask))
    if mode in ("multimodal_dec", "multimodal_enc"):
      x = self.norm_ca(x + self.cross(x, kv=img, attn_mask=img_mask)) # no ca needed for text
    return self.norm_ffn(x + self.ffn(x))


class TextTransformer(nn.Module):
  def __init__(self, cfg):
    super().__init__()
    self.embeddings = BertEmbeddings(cfg)
    self.layers = nn.ModuleList(
      [BertLayer(cfg) for _ in range(cfg.text_layers)])
    
  def forward(self, ids, mode, attn_mask=None, image_embeds=None, image_mask=None):
    x = self.embeddings(ids)
    sa = _sa_mask(pad=attn_mask, mode=mode, L=x.size(1), device=x.device)
    ca = _cross_mask(img_mask=image_mask)
    for layer in self.layers:
      x = layer(x, mode, sa_mask=sa, img=image_embeds, img_mask=ca)
    return x

# ================================
# SUB-MODULES INIT WEIGHTS

def _init_bert_weights(m):
  if isinstance(m, nn.Linear): # set linear weights to have mean=0 and std=0.02. bias also 0
    nn.init.normal_(m.weight, std=0.02)
    if m.bias is not None:
      nn.init.zeros_(m.bias)
  elif isinstance(m, nn.Embedding): # same thing with embedding layers
    nn.init.normal_(m.weight, std=0.02) # overwrites padding_idx. must be zeroed again.
    if m.padding_idx is not None:
      with torch.no_grad():
        m.weight[m.padding_idx].zero_() # zero out [pad] embedding
  elif isinstance(m, nn.LayerNorm): # y = (x-mean) / std*[w=1] + [b=0] -> y=normalized(x)
    nn.init.zeros_(m.bias)
    nn.init.ones_(m.weight)

# ================================
# FULL MODEL CLASS

class BLIPFromScratch(nn.Module):
  def __init__(self, cfg:BLIPConfig=BLIPConfig()):
    self.cfg = cfg
    # MAIN TOWERS
    self.visual = VisionTransformer(cfg=cfg)
    self.text = TextTransformer(cfg=cfg)
    # ITC
    self.vision_proj = nn.Linear(cfg.vit_width, cfg.embed_dim)
    self.text_proj = nn.Linear(cfg.text_width, cfg.embed_dim) # map both to embed dim
    self.temp = nn.Parameter(torch.tensor(0.07)) # CLIP's temp. model fine-tunes it.
    # ITM (match, no-match)
    self.itm_head = nn.Linear(cfg.text_width, 2)
    # LM (weight-tied to word embeddings)
    self.lm_head = nn.Linear(cfg.text_width, cfg.vocab_size, bias=False)
    self.apply(_init_bert_weights) # applies to all submodules not just LM
    self.lm_head.weight = self.text.embeddings.word.weight

  # ========< ENCODERS >========
  def encode_image(self, pixel_val):
    embeds = self.visual(pixel_val) # [B, N+1, W]
    feat = F.normalize(self.vision_proj(embeds[:, 0]), dim=-1) # only cls tokens for ITC
    return embeds, feat

  def encode_text(self, input_ids, attn_mask): # for ITC (unimodal)
    h = self.text(input_ids, mode='text', attn_mask=attn_mask) # this is unused by this architecture
    feat = F.normalize(self.text_proj(h[:, 0]), dim=-1)
    return h, feat

  # ========< OBJECTIVE LOSSES >========
  def loss_itc(self, img_feat, txt_feat):
    logits = img_feat @ txt_feat.t() / self.temp.clamp(min=1e-3) # similarity / temp
    # we divide by temp to help values spread out after softmax.
    # clamp prevents values getting too close to 0.
    # REMINDER: normalized vectors only need the dot product operator to calculate cosine similarity
    labels = torch.arange(logits.size(0), device=logits.device)
    return (F.cross_entropy(logits, labels) +
          F.cross_entropy(logits.t(), labels)) / 2 # bidirectional -> we take avg

  @torch.no_grad() # since we're just sampling (no inference mode though)
  def _hard_negatives(self, img_feat, txt_feat):
    sim = img_feat @ txt_feat.t() # similarity tbale [B, B]
    eye = torch.eye(sim.size(0), dtype=torch.bool, device=sim.device) # to mask correct pairs
    w_i2t = sim.masked_fill(eye, -1e4).softmax(1) # high-similarity wrong text probabilities
    w_t2i = sim.t().masked_fill(eye, -1e4).softmax(1)
    neg_txt = torch.multinomial(w_i2t, 1).squeeze(1) # random sample with a bias towards highest
    neg_img = torch.multinomial(w_t2i, 1).squeeze(1)
    return neg_img, neg_txt

  def loss_itm(self, img_embeds, img_feat, txt_feat, input_ids, attn_mask):
    B, dev = img_embeds.size(0), img_embeds.device
    img_mask = torch.ones(img_embeds.shape[:2], device=dev) # no mask
    ids = input_ids.clone()
    ids[:, 0] = self.cfg.enc_token_id
    neg_img, neg_txt = self._hard_negatives(img_feat, txt_feat)
    ids_all = torch.cat([ids, ids, ids[neg_txt]], 0)
    am_all = torch.cat([attn_mask, attn_mask, attn_mask[neg_img]], 0)
    img_all = torch.cat([img_embeds, img_embeds[neg_img], img_embeds], 0)
    imsk_all = torch.cat([img_feat, img_feat[neg_img], img_feat], 0)
    h = self.text(ids=ids_all,
                  mode="multimodal_enc",
                  attn_mask=am_all,
                  image_embed=img_all,
                  image_mask=imsk_all)
    logits = self.itm_head(h[:, 0])
    labels = torch.cat([torch.ones(B, device=dev),
                        torch.zeros(2 * B, device=dev)]).long()
    return F.cross_entropy(logits, labels)

  def loss_lm(self, img_embeds, input_ids, attn_mask):
    dev = img_embeds.device
    img_mask = torch.ones(img_embeds.shape[:2], device=dev)
    ids = input_ids.clone()
    ids[:, 0] = self.cfg.bos_token_id
    h = self.text(ids=ids,
                  mode="multimodal_dec",
                  attn_mask=attn_mask,
                  image_embeds=img_embeds,
                  image_mask=img_mask)
    logits = self.lm_head(h)
    labels = ids.masked_fill(attn_mask == 0, -100)
    return F.cross_entropy( # predict token t from before t
      logits[:, :-1].reshape(-1, logits.size(-1)),
      labels[:, 1:].reshape(-1),
      ignore_index=-100,
      label_smoothing=self.cfg.label_smoothing)

# PRE-TRAINING FORWARD (ITC + ITM + LM)
  def forward(self, pixel_val, input_ids, attn_mask):
    img_embeds, img_feat = self.encode_image(pixel_val)
    _, txt_feat = self.encode_text(input_ids, attn_mask) # returns unimodal h, which we dispose of.
    l_itc = self.loss_itc(img_feat=img_feat, txt_feat=txt_feat)
    l_itm = self.loss_itm(img_embeds, img_feat, txt_feat, input_ids, attn_mask)
    l_lm = self.loss_lm(img_embeds, input_ids, attn_mask)
    return l_itc + l_itm + l_lm, {
      "itc": l_itc.item(),
      "itm": l_itm.item(),
      "lm": l_lm.item()}

# CAPTIONING INTERFACE (LM decoder)
  @torch.no_grad()
  def generate(self, pixel_val, max_len=30, sample=False, top_k=0, temperature=1.0):
    # sample: False-> pick the best token(greedy) | True: sample from probability distribution
    # top_k: only top k tokens allowed to be in prob distribution (used when sample=True)
    was_training = self.training # save the state model is in so we turn it back to how it was
    self.eval() # disables Dropout layers
    img_embeds = self.visual(pixel_val)
    img_mask = torch.ones(img_embeds.shape[:2], device=img_embeds.device)
    B = img_embeds.size(0)
    # initial token sequence [B, 1] with starting token of [BOS]
    ids = torch.full((B, 1), self.cfg.bos_token_id,
                      dtype=torch.long,
                      device=img_embeds.device) # will be filled on generation loop
    done = torch.zeros(B, dtype=torch.bool, device=img_embeds.device) # tracks if seq is done
    for _ in range(max_len): # generation loop
      h = self.text(ids=ids,
                    mode="multimodal_dec",
                    attn_mask=None,
                    image_embeds=img_embeds,
                    image_mask=img_mask) # [B,current_seq_length] -> [B,current...,text_width(768)]
      # <NOTE> no KV cache, so the whole seq is processed on each step. insignificant for short caption
      logits = self.lm_head(h[:, -1]) # [B, txt_width] -> [B, vocab_size]
      if sample:
        if top_k > 0:
          kth = logits.topk(top_k, 1).values[:, -1, None] # keep only the top k likely tokens
          logits = logits.masked_fill(logits < kth, float("-inf")) # turn the rest to -inf
        nxt = torch.multinomial((logits / temperature).softmax(-1), 1).squeeze(1) # [B, 1] -> [B]
      else: # greedy decoding
        nxt = logits.argmax(-1)
      nxt = nxt.masked_fill(done, self.cfg.pad_token_id) # replace token with padding if seq aleady done
      ids = torch.cat([ids, nxt[:, None]], -1) # append generated tokens
      done |= (nxt == self.cfg.eos_token_id) # mark seq as done if it ended with [EOS]
      if done.all():
        break
    if was_training: # set back to the state it was
      self.train()
    return ids


# ========================================================================
# TRANSFER LEARNING MODEL
# ========================================================================

from transformers import BlipForConditionalGeneration, BlipProcessor

CHECKPOINT = "Salesforce/blip-image-captioning-base"
## defaults for comparing all 3 proposed models
GEN_DEFAULTS = dict(num_beams=3, max_new_tokens=30, do_sample=False)

def load_blip(device, dtype=torch.float16):
  model = BlipForConditionalGeneration.from_pretrained(CHECKPOINT)
  model = model.to(device, dtype=dtype).eval()
  processor = BlipProcessor.from_pretrained(CHECKPOINT) # tokenizes inputs
  return model, processor

class BLIPCaptioner:
  name = "blip-base"

  def __init__(self, device=None, dtype=torch.float16):
      self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
      self.dtype = dtype
      self.model, self.processor = load_blip(self.device, dtype)

  @torch.no_grad()
  def caption(self, images, prompt=None, **gen_kwargs):
      if not isinstance(images, (list, tuple)):
          images = [images]
      text = [prompt] * len(images) if prompt else None
      inputs = self.processor(images=images, text=text, return_tensors="pt", padding=True)
      inputs = {k: v.to(self.device) for k, v in inputs.items()}
      inputs["pixel_values"] = inputs["pixel_values"].to(self.dtype)
      out = self.model.generate(**inputs, **{**GEN_DEFAULTS, **gen_kwargs})
      # <NOTE> with a prompt, the decoded string includes the prompt prefix
      return [c.strip() for c in self.processor.batch_decode(out, skip_special_tokens=True)]