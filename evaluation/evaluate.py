"""
evaluate.py
COCO caption metrics + significance testing + diagnostics

"""

# ======================
# IMPORTS

import json
from collections import defaultdict, Counter
import numpy as np

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider

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

# =====================================
# SCORING

def score(gts_tok, res_tok):
    """Returns corpus_scores and per_image_scores keyed by metric then image_id"""
    ids = list(gts_tok.keys())
    corpus, per_image = {}, {}

    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr")
    ]

    for scorer, name in scorers:
        s, ss = scorer.compute_score(gts_tok, res_tok)
        if isinstance(name, list): # for BLEU
            for n, sc, per_im in zip(name, s, ss):
                corpus[n] = float(sc)
                # dict of img ids mapped to their score in py float
                per_image[n] = dict(zip(ids, [float(x) for x in per_im]))
        else: # for non-BLEU
            corpus[name] = float(s)
            per_image[name] = dict(zip(ids, [float(x) for x in ss]))
    
    return corpus, per_image

# =============================
# DIAGNOSTICS

def diagnostics(res_tok, gts_tok):
    caps = [res_tok[i][0] for i in res_tok] # list of just the caption
    toks = [c.split() for c in caps] # list(list(word strings))
    lens = np.array([len(t) for t in toks])

    ref_lens = np.array([
        np.mean([len(r.split()) for r in gts_tok[i]]) for i in gts_tok
    ])

    # def distinct(n):
    # return n, len_mean, 
    # len_std, len_p5 & p95, 
    # ref_len_mean, vocab_size, 
    # dup_caption_rate, 
    # exact_match_rate, 