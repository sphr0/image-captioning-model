import json
import random

dataset_path = "data/coco/annotations/captions_val2017.json"
json_address = "evaluation/captions_train2017_subset1000.json"

def create_subset(dataset_path=dataset_path,
                  seed=42,
                  n=1000,
                  create_json=True,
                  json_address=json_address):
    """
    given the coco captions_val2017.json file, or a json using its
     structure, returns a subset of n samples and optionally creates
     a new json file containing the samples

    """

    rng = random.seed(seed)
    with open(dataset_path) as f:
        coco = json.load(f)

    selected_imgs = random.sample(coco["images"], n)
    selected_ids = [img["id"] for img in selected_imgs]

    selected_annots = [annot for annot in coco["annotations"] if annot["image_id"] in selected_ids]

    subset = {
    "info": coco.get("info"),
    "licences": coco.get("licences"),
    "images": selected_imgs,
    "annotations": selected_annots
    }

    if create_json and json_address:
        with open(json_address, 'w') as f:
            json.dump(subset, f, indent=2)
    
    return subset

# create_subset(dataset_path=dataset_path,
#                   seed=42,
#                   n=5000,
#                   create_json=True,
#                   json_address="data/subsets/captions_val2017_subset5000.json")