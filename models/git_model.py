"""
GIT Image Captioning - HF implementation (both from scratch and transfered)
architecture:
    1. Visual Encoder: CLIP ViT-B/16
    2. Text Decoder: BERT-like (random initialization)

The encoder seen in the paper's GIT model (Florence/CoSwin) is not 
publically available but the GIT-base model uses a CLIP ViT-B/16, which
has differences: layerNorm right after patch+position embedding, QuickGELU
instead of GELU in MLP, and inside ViT block it uses postnorm instead
of pre-norm.
The decoder is a BERT-like transformer with weights randomly initiallized. 
In addition it uses a prefix-causal mask with LM training objective.

In the Transfered models, we have both Git-B and Git-L since Git-L could
fit in even with hardware limitations.
"""

from huggingface_hub.utils._http import default_client_factory
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field

@dataclass
class VisionConfig:
    image_size: int = 224
    patch_size: int = 16

    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12

    mlp_ratio: float = 4.0


@dataclass
class GITConfig:
    vocab_size: int = 30_522
    max_text_length: int = 1_024
    hidden_size: int = 768
    num_layers: int = 6
    num_heads: int = 12
    mlp_ratio: float = 4.0

    pad_token_id: int = 0
    bos_token_id: int = 101
    eos_token_id: int = 102

    vision: VisionConfig = field(default_factory=VisionConfig)

# ==================================================
# VISUAL ENCODER (CLIP ViT-B/16)
# ==================================================

# <NOTE> 1. MHA + test
# <NOTE> 2. projection and embedding modules
# <NOTE> 4. vision tower

# ==================================================
# Text Decoder (Bert-like)
# ==================================================

# <NOTE> 3. decoder block

# why is tie_word_embeddings equal to False?

# <NOTE> 5. assemble forward
# <NOTE> 6. greedy generate
# <NOTE> 7. smoke test
