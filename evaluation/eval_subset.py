import json
from collections import defaultdict
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset

img_dir = "data/coco/val2017"
json_path = "evaluation/captions_train2017_subset1000.json"

class EvalDataset(Dataset):
    def __init__(self, img_dir=img_dir, json_path=json_path):
        self.img_dir = Path(img_dir)

        with open(json_path, 'r') as f:
            data = json.load(f)

        # self.image
        self.images = data["images"]

        # self.references
        self.references = defaultdict(list)
        for ann in data["annotations"]:
            self.references[ann["image_id"]].append(ann["caption"])
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        id = image["id"]
        file_name = image["file_name"]
        image_path = self.img_dir / file_name
        image = Image.open(image_path).convert('RGB')

        return {
            "image": image,
            "image_id": id,
            "file_name": file_name,
            "references": self.references[id]
        }