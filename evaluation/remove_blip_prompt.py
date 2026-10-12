import json

def clean_blip_caption(caption_path="evaluation/predictions/blip_based_preds_universal_defaults_CLEAN.json", 
                       prefix="a picture of ", 
                       out_path="evaluation/predictions/blip_based_preds_universal_defaults_CLEAN.json"):
    """remove the prmopt given to BLIP's model.
    Generating captions with no prompt to BLIP's 
    coco-finetuned model will result in it adding 
    "there are" or "there is a" at the start of 
    all captions. To prevent that we add the prompt 
    of "a picture of" and later remove it."""
    if caption.lower().startswith(prefix):
        caption = caption[len(prefix):]

    with open(out_path) as f:
        json.dump()
    return caption.strip()


import json
from pathlib import Path


def clean_captions(caption_path="evaluation/predictions/blip_based_preds_universal_defaults_CLEAN.json", 
                   prefix="a picture of ", 
                   output_path="evaluation/predictions/blip_based_preds_universal_defaults_CLEAN.json"):
    """remove the prmopt given to BLIP's model.
    Generating captions with no prompt to BLIP's 
    coco-finetuned model will result in it adding 
    "there are" or "there is a" at the start of 
    all captions. To prevent that we add the prompt 
    of "a picture of" and later remove it."""
    with open(caption_path, "r", encoding="utf-8") as f:
        preds = json.load(f)

    prefix = prefix.lower()

    for prediction in preds:
        for i, caption in enumerate(prediction["caption"]):
            if caption.lower().startswith(prefix):
                prediction["caption"][i] = caption[len(prefix):].strip()


    with open(output_path, "w") as f:
        json.dump(preds, f, indent=2)

# clean_captions()