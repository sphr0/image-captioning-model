#!/usr/bin/env python3
"""
app.py - Gradio front-end for the BLIP-base captioner (the eval winner).

    python app.py                                  # Salesforce/blip-image-captioning-base
    python app.py --model ./checkpoints/blip_best  # any save_pretrained() folder
    python app.py --dtype fp32 --port 7861         # override precision / port
    python app.py --share                          # temporary public *.gradio.live link

Prompt handling
---------------
BLIP's caption checkpoint was fine-tuned with the decoder prefix "a picture of ",
so the decoder expects to *continue* that text. Leaving the prompt empty therefore
feeds "a picture of " and strips it from the result. A user prompt is fed the same
way but kept in the output, because there the prompt is part of the sentence the
user asked for ("a black and white photo of" -> "a black and white photo of a dog").

Stripping is done on token ids, not on the decoded string. The tokenizer is
uncased and re-spaces punctuation ("A Photo, of" decodes to "a photo, of"), so a
string match against what the user typed can silently fail. Comparing ids sidesteps
that, and the prefix check means a mismatch is detected instead of chopping words.

Requires: torch, transformers>=4.56 (for `dtype=`), gradio>=6.
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import gradio as gr
import torch
from PIL import Image, ImageOps
from transformers import BlipForConditionalGeneration, BlipProcessor

DEFAULT_MODEL = "Salesforce/blip-image-captioning-base"
DEFAULT_PROMPT = "a picture of "

# Decoding defaults. Set these to whatever your eval used so the demo produces the
# captions you actually scored. num_beams=3 / min 5 tokens / ~20 tokens is the
# original BLIP COCO setting.
DEFAULTS = {"num_beams": 3, "max_new_tokens": 20, "min_new_tokens": 5,
            "repetition_penalty": 1.0, "top_p": 0.9, "temperature": 1.0}


@dataclass
class Caption:
    text: str            # what the user sees
    raw: str             # full decoder output, prompt included
    prompt: str          # prompt actually fed to the decoder
    stripped: bool       # whether the prompt was removed from `text`
    prompt_tokens: int
    new_tokens: int
    latency_ms: float
    peak_vram_mb: float | None


class Captioner:
    """Loads BLIP once and captions one image per call."""

    def __init__(self, model_id: str, device: str, dtype: torch.dtype):
        self.device, self.dtype = torch.device(device), dtype
        self.processor = BlipProcessor.from_pretrained(model_id)
        self.model = (BlipForConditionalGeneration
                      .from_pretrained(model_id, dtype=dtype)
                      .to(self.device)
                      .eval())
        self.tok = self.processor.tokenizer
        self.special = set(self.tok.all_special_ids)  # [CLS]/[DEC], [SEP], [PAD], [UNK], ...
        self.default_ids = self._ids(DEFAULT_PROMPT)
        n_params = sum(p.numel() for p in self.model.parameters()) / 1e6
        print(f"loaded {model_id}: {n_params:.0f}M params, {dtype} on {self.device}")

    # ------------------------------------------------------------------ helpers
    def _ids(self, text: str) -> list[int]:
        """Prompt ids exactly as they will appear in the output (no special tokens)."""
        ids = self.tok(text, add_special_tokens=False).input_ids
        return [t for t in ids if t not in self.special]

    @staticmethod
    def _prep(image: Image.Image) -> Image.Image:
        # Gradio already does both of these; repeated here so Captioner is safe to
        # call from scripts too. RGBA/L/P images would otherwise break the processor.
        return ImageOps.exif_transpose(image).convert("RGB")

    # --------------------------------------------------------------- inference
    @torch.inference_mode()
    def __call__(self, image: Image.Image, prompt: str | None = None, **gen) -> Caption:
        user_prompt = (prompt or "").strip()
        text = user_prompt or DEFAULT_PROMPT
        prompt_ids = self._ids(text)
        strip = not user_prompt or prompt_ids == self.default_ids

        enc = self.processor(images=self._prep(image), text=text, return_tensors="pt")
        # Only pixel_values gets the model dtype; ids and masks must stay int64.
        pixel_values = enc.pixel_values.to(self.device, self.dtype)
        input_ids = enc.input_ids.to(self.device)
        attention_mask = enc.attention_mask.to(self.device)

        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        t0 = time.perf_counter()
        # BLIP's generate() swaps [CLS] for [DEC] (bos) and drops the trailing [SEP],
        # so the output is: [DEC] + prompt ids + new ids + [SEP]. Note it edits
        # input_ids in place, so never reuse a tensor across calls.
        out = self.model.generate(pixel_values=pixel_values, input_ids=input_ids,
                                  attention_mask=attention_mask, **gen)[0].tolist()
        latency_ms = (time.perf_counter() - t0) * 1000  # .tolist() already synced the GPU
        peak = (torch.cuda.max_memory_allocated(self.device) / 2**20
                if self.device.type == "cuda" else None)

        ids = [t for t in out if t not in self.special]
        has_prefix = ids[:len(prompt_ids)] == prompt_ids
        if not has_prefix:  # should not happen; better to show everything than cut words
            print(f"[warn] output does not start with the prompt ids: {out}")
        new_ids = ids[len(prompt_ids):] if has_prefix else ids

        raw = self.tok.decode(ids, skip_special_tokens=True).strip()
        new_text = self.tok.decode(new_ids, skip_special_tokens=True).strip()
        if strip and has_prefix:
            text_out = new_text
        elif has_prefix:
            # Re-attach the prompt as typed, not its decoded form ("black-and-white"
            # would come back as "black - and - white", "Paris" as "paris").
            joiner = " " if new_text[:1].isalnum() else ""
            text_out = user_prompt + joiner + new_text
        else:
            text_out = raw
        return Caption(text=text_out[:1].upper() + text_out[1:], raw=raw, prompt=text,
                       stripped=strip and has_prefix, prompt_tokens=len(prompt_ids),
                       new_tokens=len(new_ids), latency_ms=latency_ms, peak_vram_mb=peak)


def gen_kwargs(method: str, num_beams: int, max_new: int, rep_pen: float,
               top_p: float, temperature: float) -> dict:
    """max_new_tokens rather than max_length: max_length counts the prompt too, so a
    long custom prompt would silently shorten the caption."""
    kw = {"max_new_tokens": int(max_new), "min_new_tokens": DEFAULTS["min_new_tokens"],
          "repetition_penalty": float(rep_pen)}
    if method == "Sampling":
        kw |= {"do_sample": True, "num_beams": 1, "top_p": float(top_p),
               "temperature": float(temperature)}
    else:
        kw |= {"do_sample": False, "num_beams": int(num_beams)}
    return kw


# ---------------------------------------------------------------------------- UI
def build_ui(cap: Captioner) -> gr.Blocks:
    def run(image, prompt, method, num_beams, max_new, rep_pen, top_p, temperature):
        if image is None:
            raise gr.Error("Upload an image first.")
        c = cap(image, prompt, **gen_kwargs(method, num_beams, max_new, rep_pen, top_p, temperature))
        details = {
            "prompt fed to decoder": c.prompt,
            "prompt stripped from caption": c.stripped,
            "raw decoder output": c.raw,
            "prompt tokens": c.prompt_tokens,
            "generated tokens": c.new_tokens,
            "latency (ms)": round(c.latency_ms, 1),
            "peak VRAM (MB)": None if c.peak_vram_mb is None else round(c.peak_vram_mb),
            "device / dtype": f"{cap.device} / {str(cap.dtype).removeprefix('torch.')}",
        }
        return c.text, details

    with gr.Blocks(title="BLIP captioner") as demo:
        gr.Markdown("## BLIP-base image captioner\n"
                    "Upload an image. Leave the prompt empty for a plain caption, or start "
                    "the sentence yourself and BLIP will finish it.")
        with gr.Row():
            with gr.Column():
                image = gr.Image(type="pil", image_mode="RGB", label="Image", height=360)
                prompt = gr.Textbox(label="Prompt (optional)", placeholder=DEFAULT_PROMPT.strip(),
                                    max_length=120,
                                    info=f'Empty = "{DEFAULT_PROMPT.strip()}", which is removed '
                                         "from the result. Your own prompt is kept in it.")
                with gr.Accordion("Decoding", open=False):
                    method = gr.Radio(["Beam search", "Sampling"], value="Beam search", label="Method",
                                      info="Beam search: stable, what the eval scores. "
                                           "Sampling: varied, re-run for alternatives.")
                    num_beams = gr.Slider(1, 8, value=DEFAULTS["num_beams"], step=1,
                                          label="Beams (beam search)")
                    max_new = gr.Slider(5, 60, value=DEFAULTS["max_new_tokens"], step=1,
                                        label="Max new tokens")
                    rep_pen = gr.Slider(1.0, 2.0, value=DEFAULTS["repetition_penalty"], step=0.05,
                                        label="Repetition penalty",
                                        info="Raise to ~1.3 if captions loop ('a dog and a dog and...')")
                    top_p = gr.Slider(0.1, 1.0, value=DEFAULTS["top_p"], step=0.05,
                                      label="Top-p (sampling)")
                    temperature = gr.Slider(0.1, 2.0, value=DEFAULTS["temperature"], step=0.05,
                                            label="Temperature (sampling)")
                button = gr.Button("Caption", variant="primary")
            with gr.Column():
                caption = gr.Textbox(label="Caption", lines=3, interactive=False, buttons=["copy"])
                with gr.Accordion("Under the hood", open=False):
                    details = gr.JSON(label="Generation details")

        inputs = [image, prompt, method, num_beams, max_new, rep_pen, top_p, temperature]
        # One public endpoint (/caption) for scripts and gradio_client; the Enter-key
        # and auto-on-upload triggers run the same function without a second endpoint.
        button.click(run, inputs, [caption, details], api_name="caption")
        for trigger in (prompt.submit, image.upload):
            trigger(run, inputs, [caption, details], api_visibility="private")
    return demo


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL, help="hub id or local save_pretrained() folder")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", choices=["auto", "fp32", "fp16", "bf16"], default="auto",
                    help="auto = fp16 on GPU, fp32 on CPU")
    ap.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to reach it from your LAN")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true")
    args = ap.parse_args()

    dtypes = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    if args.dtype == "auto":
        dtype = torch.float16 if args.device.startswith("cuda") else torch.float32
    else:
        dtype = dtypes[args.dtype]

    cap = Captioner(args.model, args.device, dtype)
    # Warm-up: the first CUDA call pays for context setup and kernel selection.
    cap(Image.new("RGB", (384, 384), "gray"), max_new_tokens=3)

    # One GPU, one model: run one request at a time and queue the rest.
    demo = build_ui(cap).queue(default_concurrency_limit=1, max_size=32)
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()