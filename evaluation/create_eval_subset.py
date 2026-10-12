import json
import random
from collections import Counter

COCO_PATH = "data/coco/annotations/captions_val2017.json"
KARPATHY_PATH = "data/coco/dataset_coco.json"
OUT_DIR = "data/subsets/val2017_subset1000.json"

KARPATHY_EXPECTED = {"train": 82783, "test": 5000, "restval": 30504, "val": 5000,}

def _clean_ids(karpathy_path):
    """given the karpathy json path, returns the 
    clean version of coco, suitable for testing 
    on kaparthy-trained models."""
    # load karpathy onto a var
    with open(karpathy_path) as f:
        karpathy = json.load(f)

    # count karpathy to match KARPATHY_EXPECTED, if not equal throw valerror
    counts = dict(Counter(im["split"] for im in karpathy["images"]))
    if counts != KARPATHY_EXPECTED:
        raise ValueError(f"Karpathy split count doesn't match the expected numbers. \
            \nexpected: {KARPATHY_EXPECTED}\nreceived: {counts}")

    # take the part of the karpathy ids that include an overlap with coco_val2017 (len=10k)
    return {im["cocoid"] for im in karpathy["images"] if im["split"] in ("val", "test")}


def create_subset(dataset_path=COCO_PATH,
                  karpathy_path=KARPATHY_PATH,
                  seed=42,
                  n=None,
                  out_dir=OUT_DIR):
    """
    given the coco captions_val2017.json file, or a json using its
    structure, returns a subset of n images that do NOT overlap 
    with any training split (including karpathy train & restval). 
    optionally creates a new json file containing the samples

    if given n=None, it will return the cleaned version of the 
    entire dataset.
    """

    with open(dataset_path) as f:
        coco = json.load(f)

    clean_ids = _clean_ids(karpathy_path)

    # sort coco data by img ID number (only clean ids)
    # we sort so the randomization works properly
    data_pool = sorted((img for img in coco["images"] if img["id"] in clean_ids),
    key=lambda img: img["id"])

    if n is None:
        n = len(data_pool)
    if n > len(data_pool):
        raise ValueError(f"The arguement given for n cannot be above {len(data_pool)}.")


    rng = random.Random(seed)
    rng.shuffle(data_pool)
    selected_imgs = sorted(data_pool[:n], key=lambda img: img["id"])
    selected_ids = [img["id"] for img in selected_imgs]

    selected_annots = [a for a in coco["annotations"] if a["image_id"] in selected_ids]

    subset = {
    "info": coco.get("info"),
    "licences": coco.get("licences"),
    "images": selected_imgs,
    "annotations": selected_annots
    }

    if out_dir:
        with open(out_dir, 'w') as f:
            json.dump(subset, f, indent=2)
    
    return subset

create_subset(dataset_path=COCO_PATH,
              karpathy_path=KARPATHY_PATH,
              seed=42,
              n=None,
              out_dir="data/subsets/captions_val2017_CLEAN.json")