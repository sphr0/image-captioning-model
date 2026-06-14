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

