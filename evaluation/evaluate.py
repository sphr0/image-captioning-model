"""
docstring goes here.
"""

# ======================
# IMPORTS

import json
from collections import defaultdict

METRICS = ["Bleu_1", "Bleu4", "METEOR", "ROUGE_L", "CIDEr"]

# =======================
# Loader Functions

def load_references(ann_path):
    """turn captions_val2017.json to python defaultdict format"""
    with open(ann_path) as f:
        data = json.load(f)
    refs = defaultdict(list)
    for a in data["annotations"]:
        refs[int(a["image_id"])].append([a["caption"]])
    return dict(refs)
