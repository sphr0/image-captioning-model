"""
evaluate.py
COCO caption metrics + significance testing + diagnostics

"""

# ======================
# IMPORTS

import json
from collections import defaultdict

METRICS = ["Bleu_1", "Bleu4", "METEOR", "ROUGE_L", "CIDEr"]

# =======================
# Loader Functions

def load_references(ann_path):
    """turn captions_val2017.json to python dict"""
    with open(ann_path) as f:
        data = json.load(f)
    refs = defaultdict(list)
    for a in data["annotations"]:
        refs[int(a["image_id"])].append(a["caption"])
    return dict(refs)

def load_predictions(path):
    """turn predictions json to python dict"""
    with open(path) as f:
        data = json.load(f)
    preds = {}
    for p in data:
        iid = int(p["image_id"])
        if iid in preds: # Catch if there are any duplicates
            raise ValueError(f"Duplicate image_id {iid} in {path}")
        preds[iid] = p["caption"].pop().strip()
    return preds