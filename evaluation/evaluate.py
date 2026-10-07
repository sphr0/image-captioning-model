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

try:
    import torch
    import torch.nn.functional as F
    import open_clip
    from PIL import Image
    _CLIP_OK = True
except ImportError:
    _CLIP_OK = False

from coco_synonyms import COCO_SYNONYMS

METRICS = ["Bleu_1", "Bleu_4", "METEOR", "ROUGE_L", "CIDEr"]
CLIP_METRICS = ["CLIPScore", "RefCLIPScore"]

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
        preds[iid] = " ".join(p["caption"][0].split()) # removes the extra space and any \n that may exist
    return preds

def load_img_paths(ann_path, imgs_dir):
    """Build {image_id: Path} using coco-format json and img filenames"""
    with open(ann_path) as f:
        data = json.load(f)
    img_folder = Path(imgs_dir)
    return {
        int(im["id"]): img_folder / im["file_name"] for im in data["images"]
    }

def load_instance_objects(instance_path):
    """Turn instances_val2017 json to {image_id: set(coco_category_name)}.
    Used for CHAIR as the ground-truth bounding-box annotation"""
    with open(instance_path) as f:
        data = json.load(f)
    cat_map = {cat["id"]: cat["name"] for cat in data["categories"]}
    objs = defaultdict(set)
        # map image_id to category names
    for ann in data["annotations"]:
        objs[int(ann["image_id"])].add(cat_map[ann["category_id"]])
    return dict(objs)

# =====================================
# N-GRAM SCORING

def score(gts_tok, res_tok, scorers_list):
    """Returns corpus_scores and per_image_scores keyed by metric then image_id"""
    assert list(gts_tok.keys()) == list(res_tok.keys()), (
        "gts_tok and res_tok key order is mismatched.")

    ids = list(gts_tok.keys())
    corpus, per_image = {}, {}

    for scorer, name in scorers_list:
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
    Corpus-level statistics that give deeper insight on metric results.
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


# ==================================
# CLIPScore + RefCLIPScore

def open_clip_scores(preds,
                refs, 
                img_paths, 
                ids,
                device='cuda', 
                model_name='ViT-L-14', 
                pretrained='openai', 
                batch_size=64):
    """compute CLIPScore and RefCLIPScore for every img in ids.
    Returns ({image_id: clip_scores}, {image_id: refclip_scores})"""

    precision = "fp16" if "cuda" in str(device) else "fp32"
    model, _, preprocess = open_clip.create_model_and_transforms(model_name=model_name,
                                                                 pretrained=pretrained, 
                                                                 device=device, 
                                                                 precision=precision)
    model.eval()
    tokenizer = open_clip.get_tokenizer(model_name=model_name)
    fp_dtype = torch.float16 if precision == "fp16" else torch.float32

    # pre-encoding all texts
    all_ref_txts, all_ref_ids = [], []
    for iid in ids:
        for cap in refs[iid]:
            all_ref_ids.append(iid)
            all_ref_txts.append(cap)

    ref_embeds_flat = []
    with torch.no_grad():
        for i in range(0, len(all_ref_txts), 256):
            tok = tokenizer(all_ref_txts[i:i+256]).to(device)
            emb = F.normalize(model.encode_text(tok), dim=-1)
            ref_embeds_flat.append(emb.cpu().float())
    ref_embeds_flat = torch.cat(ref_embeds_flat, dim=0)

    ref_embeddings = {}
    ptr = 0
    for iid in ids:
        n = len(refs[iid])
        ref_embeddings[iid] = ref_embeds_flat[ptr:ptr+n]
        ptr += n

    # main loop
    clip_scores, refclip_scores = {}, {}
    with torch.no_grad():
        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i:i+batch_size]

            imgs = torch.stack([ # img embeds
                preprocess(Image.open(img_paths[iid]).convert("RGB")) for iid in batch_ids
            ]).to(device, dtype=fp_dtype)
            f_img = F.normalize(model.encode_image(imgs), dim=-1) # [B, D]

            cands = [preds[iid] for iid in batch_ids] # candidate txt embeds
            f_cand = F.normalize(
                model.encode_text(tokenizer(cands).to(device)), dim=-1) # [B, D]

            # CLIPScore = 2.5 x max(cos(img, cand), 0)
            cos_ic = (f_img * f_cand).sum(-1).clamp(min=0) * 2.5 # [B]
            cos_cr = torch.zeros(len(batch_ids)) # RefCLIPScore
            for j, iid in enumerate(batch_ids):
                f_refs = ref_embeddings[iid].to(device, dtype=fp_dtype)
                cos_cr[j] = ((f_cand[j:j+1] * f_refs).sum(-1).clamp(min=0).max() * 2.5).cpu()

            a = cos_ic.float().cpu()
            b = cos_cr
            denom = a + b
            harmonic_mean = torch.where(denom > 0,
                                 2 * a * b / denom,
                                 torch.zeros_like(denom))
            
            for j, iid in enumerate(batch_ids):
                clip_scores[iid] = float(a[j])
                refclip_scores[iid] = float(harmonic_mean[j])
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return clip_scores, refclip_scores


# ====================================
# CHAIR

def _build_reverse_lookup():
    """Invert COCO_SYNONYMS to {word: category}"""
    word_to_cat = {form: cat for cat, forms in COCO_SYNONYMS.items() for form in forms}
    # to serve longer phrases first to CHAIR, we sort by word length  in descending order
    # The lambda returns a tuple. If first vaue is equal, it will sort them alphabetically
    phrases = sorted(word_to_cat, key=lambda p: (-len(p.split()), p))
    return word_to_cat, phrases


def _mentioned_objects(cap_str, word_to_cat, phrases):
    """Extract the set of COCO categories in one tokenized caption"""
    words, found, i = cap_str.split(), set(), 0
    while i < len(words):
        matched = False
        for phrase in phrases:
            ptoks = phrase.split()
            end = i + len(ptoks)
            if words[i:end] == ptoks:
                found.add(word_to_cat[phrase])
                i, matched = end, True
                break
        if not matched:
            i +=1
    return found


def chair(res_tok, image_objects):
    """
    Compute CHAIR_i and CHAIR_s. CHAIR_i shows how badly each cap
    hallucinates. CHAIR_s shows how often does hallucination occur
    in the corpus.
    returns:
        {CHAIR_i, CHAIR_s, per_image}
    """
    word_to_cat, phrases = _build_reverse_lookup()
    per_image_chair = {}
    chair_i_vals = []
    hallu_cap_count = 0

    for iid, cap_list in res_tok.items():
        cap = cap_list[0]
        mentioned_objs = _mentioned_objects(cap, word_to_cat, phrases)
        gt_objs = image_objects.get(iid, set()) # return set if no match
        hallucinated = mentioned_objs - gt_objs

        ci = len(hallucinated) / len(mentioned_objs) if mentioned_objs else 0.0
        chair_i_vals.append(ci)
        per_image_chair[iid] = {
             "chair_i": ci,
             "mentioned": sorted(mentioned_objs),
             "hallucinated": sorted(hallucinated)
        }
        if hallucinated:
            hallu_cap_count += 1

    return {
        "CHAIR_i": float(np.mean(chair_i_vals)) if chair_i_vals else 0.0,
        "CHAIR_s": hallu_cap_count / len(res_tok) if res_tok else 0.0,
        "per_image": per_image_chair
    }



# ==================================
# BOOTSTRAP STATS

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

def final_report(annotations, instances, images, clip_model, preds, out="evaluation/results/eval.json", n_boot=500):
    """
    
    EXAMPLE:
        annotations="data/subsets/captions_val2017_subset5000.json",
        instances="data/coco/annotations/instances_val2017.json",
        images="data/coco/val2017",
        clip_model="ViT-L-14",
        preds=["vit_gpt2=evaluation/predictions/vit_gpt_preds_universal_defaults_5000.json",
               "blip=evaluation/predictions/blip_based_preds_universal_defaults_5000.json",
               "git=evaluation/predictions/git_preds_universal_defaults_5000.json"],
        out="evaluation/results/eval_5000.json",
        n_boot=2000
    """

    do_clip = images is not None
    do_chair = instances is not None

    if do_clip and not _CLIP_OK:
        print("[warn] images given but open-clip-torch/torch not installed. \nSkipping CLIPScore...")
        do_clip = False

    if do_chair:
        image_objects = load_instance_objects(instance_path=instances)
    if do_clip:
        image_paths = load_img_paths(ann_path=annotations, imgs_dir=images)

    refs_all = load_references(annotations)
    models = dict(p.split("=", 1) for p in preds)
    loaded = {k: load_predictions(v) for k, v in models.items()}
    common = set.intersection(*[set(p) for p in loaded.values()]) & set(refs_all)

    # catch if no intersection OR not all intersections
    if not common:
        raise ValueError("No overlapping image_ids across models + annotations.")
    for k, p in loaded.items():
        if len(p) != len(common):
            print(f"  [warn] {k}: {len(p)} preds -> {len(common)} after intersection")
    ids = sorted(common)
    print(f"Evaluating {len(ids)} images across {len(loaded)} models\n")

    tok = PTBTokenizer()
    gts_tok = tok.tokenize({i:[{"caption": c} for c in refs_all[i]] for i in ids})

    scorers_list = [
    (Bleu(4),   ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
    (Meteor(),  "METEOR"),
    (Rouge(),   "ROUGE_L"),
    (Cider(),   "CIDEr")]

    results = {}
    res_toks = {}
    
    for name, preds in loaded.items():
        print(f"==========< {name} >==========")
        res_tok = tok.tokenize({i:[{"caption": preds[i]}] for i in ids})
        res_toks[name] = res_tok
        corpus, per_image = score(gts_tok, res_tok, scorers_list)
        results[name] = {
            "corpus": corpus,
            "per_image": per_image,
            "diagnostics": diagnostics(res_tok, gts_tok)}

    # ================
    # CHAIR
    if do_chair:
        print("\nRunning CHAIR")
        for name in results:
            ch = chair(res_toks[name], image_objects)
            results[name]["chair"] = ch
            print(f"  {name:<14}  CHAIR_i={ch['CHAIR_i']:.4f}  "
                  f"CHAIR_s={ch['CHAIR_s']:.4f}")

    # ================
    # CLIPScore & RefCLIPScore
    if do_clip:
        device = "cuda" if (_CLIP_OK and torch.cuda.is_available()) else "cpu"
        print(f"\nRunning CLIPScore ({clip_model}, {device})...")
        for name, preds in loaded.items():
            print(f"  {name}...", end=" ", flush=True)
            cs, rcs = open_clip_scores(preds=preds,
                                       refs=refs_all,
                                       img_paths=image_paths, 
                                       ids=ids,
                                       device=device,
                                       model_name=clip_model)
            results[name]["per_image"]["CLIPScore"] = cs
            results[name]["per_image"]["RefCLIPScore"] = rcs
            mean_cs = np.mean(list(cs.values()))
            mean_rcs = np.mean(list(rcs.values()))
            print(f"CLIPScore={mean_cs:.4f}  RefCLIPScore={mean_rcs:.4f}")

    # ================
    # REPORTS
    active_clip = CLIP_METRICS if do_clip else []
    W = 14 + 13 * (len(METRICS) + len(active_clip))

    print("\n" + "-" * W)
    print(f"{'model':<14}" + "".join(f"{m:>13}" for m in METRICS))
    print("-" * W)
    for name, r in results.items():
        row = f"{name:<14}"
        for m in METRICS:
            row += f"{r['corpus'][m]:>13.4f}"
        for m in active_clip:
            mean_val = np.mean(list(r["per_image"][m].values()))
            row += f"{mean_val:>13.4f}"
            
        print(row)
    print("=" * W)
    print("  BLEU/METEOR: corpus-level.  "
          "CIDEr/ROUGE_L/CLIPScore: per-image mean.")

    # ======
    # CI block
    ci_metrics = ["Bleu_4", "CIDEr"] + active_clip
    print(f"\n95% CI (per-image bootstrap):  {', '.join(ci_metrics)}")
    for name, r in results.items():
        parts = []
        for m in ci_metrics:
            _, lo, hi = bootstrap_ci(r["per_image"][m], ids, n_boot)
            parts.append(f"{m} [{lo:.3f},{hi:.3f}]")
        print(f"  {name:<14} " + "  ".join(parts))
    
    print("\nDiagnostics:")
    keys = ["len_mean", "ref_len_mean", "vocab_size", "distinct_2",
            "dup_caption_rate", "exact_ref_match_rate"]
    print(f"{'model':<14}" + "".join(f"{k:>22}" for k in keys))
    for name, r in results.items():
        d = r["diagnostics"]
        print(f"{name:<14}" + "".join(f"{d[k]:>22.4f}" for k in keys))

    print("\n=====<Pairwise Bootstrap>=====")
    for label, metric in [("CIDEr", "CIDEr")] + (
        [("CLIPScore", "CLIPScore")] if do_clip else []):
        print(f"\nPairwise {label}, paired bootstrap:")
        names = list(results)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                delta, p = paired_bootstrap(
                    results[a]["per_image"][metric],
                    results[b]["per_image"][metric], ids, n_boot)
                verdict = "significant" if p < 0.05 or p > 0.95 else "NOT significant"
                print(f"  {a} - {b}: {delta:+.4f}  p={p:.4f}  ({verdict})")

    if do_chair:
        print(f"\nCHAIR:\n{'model':<14}{'CHAIR_i':>12}{'CHAIR_s':>12}")
        for name, r in results.items():
            print(f"{name:<14}{ch['CHAIR_i']:>12.4f}{ch['CHAIR_s']:>12.4f}")


    # Saving
    output_path = Path(out)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # saving per_image outputs as well
    per_img_out = "".join(out.split(".")[0]) + "_per_image.json"
    per_img_reports = {
        name: {
            m: r["per_image"][m]
            for m in list(METRICS) + active_clip
            if m in r["per_image"]
        }
        for name, r in results.items()
    }
    with open(per_img_out, "w") as f:
        json.dump({"n_images": len(ids), "models": per_img_reports}, f, indent=2)
    print(f"\n {per_img_out} has been written.")

    # summary JSON = corpus scores + diagnostics + CHAIR aggregates (no per-image)
    summary = {}
    for name, r in results.items():
        entry = {"corpus": r["corpus"], "diagnostics": r["diagnostics"]}
        if do_clip:
            entry["clip_mean"] = {
                m: float(np.mean(list(r["per_image"][m].values())))
                for m in CLIP_METRICS
            }
        if do_chair:
            # strip the bulky per_image dict from the summary file
            entry["chair"] = {k: v for k, v in r["chair"].items()
                              if k != "per_image"}
        summary[name] = entry

    with open(out, "w") as f:
        json.dump({"n_images": len(ids), "models": summary}, f, indent=2)
    print(f"\n{out} has been written.")


# final_report(annotations="data/subsets/captions_val2017_subset5000.json",
#              instances="data/coco/annotations/instances_val2017.json",
#              images="data/coco/val2017",
#              clip_model="ViT-L-14",
#              preds=["vit_gpt2=evaluation/predictions/vit_gpt_preds_universal_defaults_5000.json",
#                     "blip=evaluation/predictions/blip_based_preds_universal_defaults_5000.json",
#                     "git=evaluation/predictions/git_preds_universal_defaults_5000.json"],
#                     out="evaluation/results/eval_5000.json",
#                     n_boot=2000)