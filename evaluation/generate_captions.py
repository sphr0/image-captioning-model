import json
from pathlib import Path

def generate_captions(captioner, dataset, output_path):
    preds = []

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for i, sample in enumerate(dataset):
        caption = captioner.caption(sample["image"])

        preds.append({
            "image_id": sample["image_id"],
            "caption": caption
        })

        if (i + 1) % 100 == 0:
            print(f'{i+1} / {len(dataset)} captions generated')

    with open(output_path, 'w') as f:
        json.dump(preds, f, indent=2)


    # Sanity Checks
    n_preds = len(preds)
    n_data = len(dataset)
    assert n_preds == n_data, (
            f"Expected {n_data} number of predictions, got {n_preds}.")

    pred_ids = [pred["image_id"] for pred in preds]

    assert len(pred_ids) == len(set(pred_ids)), (
            "Duplicate image ids found in predictions.")

    return preds