"""
evaluate.py
COCO caption metrics + significance testing + diagnostics

"""

# ======================
# IMPORTS

import json
from collections import defaultdict, Counter
import numpy as np
from pathlib import Path

from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider

METRICS = ["Bleu_1", "Bleu_4", "METEOR", "ROUGE_L", "CIDEr"]

# =======================
# LOADER FUNC

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
        preds[iid] = p["caption"][0]
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


def bootstrap_ci(per_image, ids, n_boot=2000, alpha=0.05, seed=42):
    """returns scores mean, percentile low, percentile high
    from a bootstrap sample set"""
    rng = np.random.default_rng(seed)
    v = np.array([per_image[i] for i in ids])
    idx = rng.integers(0, len(v), (n_boot, len(v)))
    mean = v[idx].mean(axis=1)
    lo, hi = np.percentile(mean, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(v.mean()), float(lo), float(hi)


def paired_bootstrap(a, b, ids, n_boot=2000, seed=42):
    """returns avg score delta between a & b and 
    avg bootstrap difference equal or greater than 0."""
    rng = np.random.default_rng(seed)
    d = np.array([a[i] - b[i] for i in ids]) # delta of model a & b per-img scores
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    avg_d = d[idx].mean(axis=1)
    return float(d.mean()), float((avg_d <= 0).mean())


# =======================
# REPORTS

def final_report(annotations, preds, out="evaluation/results/eval.json", n_boot=500):
    """
    
    EXAMPLE:
        annotations =
            "data/captions.json"
        preds =
            ["vit_gpt2=preds/vit_gpt2.json",
             "blip=preds/blip.json",
             "git=preds/git.json"]
        out = "results/eval.json"
        n_boot = 1000
    """
    refs_all = load_references(annotations)
    models = dict(p.split("=", 1) for p in preds)
    loaded = {k: load_predictions(v) for k, v in models.items()}
    common = set.intersection(*[set(p) for p in loaded.values()])
    common &= set(refs_all)

    # catch if no intersection OR not all intersections
    if not common:
        raise ValueError("No overlapping image_ids across models + annotations.")
    for k, p in loaded.items():
        if len(p) != len(common):
            print(f"  [warn] {k}: {len(p)} preds -> {len(common)} after intersection")
    ids = set(common)
    print(f"Evaluating {len(ids)} images across {len(loaded)} models\n")

    tok = PTBTokenizer()
    gts_tok = tok.tokenize({i:[{"caption": c} for c in refs_all[i]] for i in ids})
    results = {}

    for name, preds in loaded.items():
        print(f"==========< {name} >==========")
        res_tok = tok.tokenize({i:[{"caption": preds[i]}] for i in ids})
        corpus, per_image = score(gts_tok, res_tok)
        results[name] = {
            "corpus": corpus,
            "per_image": per_image,
            "diagnostics": diagnostics(res_tok, gts_tok)}

    # ================
    # REPORTS
    print("\n" + "-" * 78)
    print(f"{'model':<14}" + "".join(f"{m:>13}" for m in METRICS))
    print("-" * 78)
    for name, r in results.items():
        row = f"{name:<14}"
        for m in METRICS:
            mean, lo, hi = bootstrap_ci(r["per_image"][m], ids, n_boot)
            row += f"{mean:>13.4f}"
            r.setdefault("ci", {})[m] = [lo, hi]
        print(row)
    print("=" * 78)

    print("\n95% CI (paired bootstrap over images):")
    for name, r in results.items():
        print(f"  {name:<14}" + "  ".join(
            f"{m} [{r['ci'][m][0]:.3f}, {r['ci'][m][1]:.3f}]" for m in ["Bleu_4", "CIDEr"]))
    
    print("\nDiagnostics:")
    keys = ["len_mean", "ref_len_mean", "vocab_size", "distinct_2",
            "dup_caption_rate", "exact_ref_match_rate"]
    print(f"{'model':<14}" + "".join(f"{k:>22}" for k in keys))
    for name, r in results.items():
        d = r["diagnostics"]
        print(f"{name:<14}" + "".join(f"{d[k]:>22.4f}" for k in keys))

    print("\nPairwise (CIDEr), paired bootstrap:")
    names = list(results)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            delta, p = paired_bootstrap(
                results[a]["per_image"]["CIDEr"],
                results[b]["per_image"]["CIDEr"], ids, n_boot)
            verdict = "significant" if p < 0.05 or p > 0.95 else "NOT significant"
            print(f"  {a} - {b}: {delta:+.4f}  p={p:.4f}  ({verdict})")

    output_path = Path(out)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for r in results.values():
        r.pop("per_image")

    with open(out, "w") as f:
        json.dump({"n_images": len(ids), "models": results}, f, indent=2)
    print(f"\n{out} has been written.")


# <NOTE> must change subset number of data from 1k to 5k
# <NOTE> vit_gpt2 preds broken: 
# `vit_gpt_preds_universal_defaults.json` has different format

# final_report(annotations="evaluation/captions_val2017_subset1000.json",
#              preds=["vit_gpt2=evaluation/predictions/vit_gpt_preds_universal_defaults.json",
#                     "blip=evaluation/predictions/blip_based_preds_universal_defaults.json",
#                     "git=evaluation/predictions/git_preds_universal_defaults.json"])


