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
    """
    RETURNS:
        n: number of evaluated caps
        len_mean: avg num of tokens in caps
        len_std: std through cap lengths
        len_p5: bottom-5% cap length
        len_p95: top-5% cap length
        ref_len_mean: avg length of ground-truth caps
        vocab_size: vocab size of all generated caps
        distinct-1 & distinct-2: distinct measurements
        dup_caption_rate: rate of exact generated caps
         for different images
        exact_ref_match_rate: rate of exact matches in
         generated and reference caps
    """
    caps = [res_tok[i][0] for i in res_tok] # list of just the caption
    toks = [c.split() for c in caps] # list(list(word strings))
    lens = np.array([len(t) for t in toks])

    ref_lens = np.array([
        np.mean([len(r.split()) for r in gts_tok[i]]) for i in gts_tok
    ])

    def distinct(n): # lexical diversity measure
        """returns distinct-n of toks"""
        grams = Counter()
        # for each list of words, make up n-long tuples as keys for Counter and update it.
        for t in toks:
            grams.update(tuple(t[k: k+n]) for k in range(len(t) - n + 1))
        total = sum(grams.values()) # total n-gram occurances, len(grams) = no. of unique n-grams
        return len(grams) / total if total else 0.0

    # rate of exact matches in generated captions and ground-truth captions
    # <NOTE> do we need the [0]? Must test later
    exact_rate = sum(
        1 for i in res_tok if res_tok[i][0] in set(gts_tok[i])
    ) / len(res_tok)

    return {
        "n": len(caps),
        "len_mean": float(lens.mean()),
        "len_std": float(lens.std()),
        "len_p5": float(np.percentile(lens, 5)),
        "len_p95": float(np.percentile(lens, 95)),
        "ref_len_mean": float(ref_lens.mean()),
        "vocab_size": len({w for t in toks for w in t}),
        "distinct_1": distinct(1),
        "distinct_2": distinct(2),
        "dup_caption_rate": 1 - len(set(caps)) / len(caps),
        "exact_ref_match_rate": exact_rate
    }