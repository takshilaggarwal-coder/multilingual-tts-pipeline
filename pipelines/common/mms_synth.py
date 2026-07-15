#!/usr/bin/env python
"""Meta MMS-TTS synthesis (facebook/mms-tts-*), the always-works VITS floor.

Single-pass VITS, no cloning, tiny + faster-than-realtime on CPU. Used as the
per-language control/floor (mms-tts-ara, mms-tts-hin). License: CC-BY-NC-4.0
(non-commercial) — disclosed. Runs via transformers in the mlx venv (torch present).

    python mms_synth.py --model-id facebook/mms-tts-ara --texts eval/texts/ar.tsv \
        --outdir outputs/ar/mms --lang ar
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "pipelines" / "common"))
from synth_utils import TimingRecorder, load_texts, run_synth_loop  # noqa: E402


def build_synth_fn(model, tokenizer):
    def synth_fn(text):
        t0 = time.perf_counter()
        inputs = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            out = model(**inputs).waveform  # (1, samples)
        first_chunk_s = time.perf_counter() - t0  # batch model: first audio == full clip
        wav = out.squeeze().cpu().numpy().astype(np.float32)
        return wav, model.config.sampling_rate, first_chunk_s

    return synth_fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--texts", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--lang", required=True)
    args = ap.parse_args()

    from transformers import AutoTokenizer, VitsModel

    t0 = time.perf_counter()
    model = VitsModel.from_pretrained(args.model_id)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model.eval()
    cold_load_s = time.perf_counter() - t0
    if getattr(tokenizer, "is_uroman", False):
        print(f"[warn] {args.model_id} tokenizer expects uroman-romanized input")

    rows = load_texts(args.texts)
    recorder = TimingRecorder(
        model_name=f"MMS-TTS ({args.lang})",
        model_ref=args.model_id,
        device="cpu",
        config={"sampling_rate": model.config.sampling_rate},
        cold_load_s=cold_load_s,
    )
    run_synth_loop(rows, args.outdir, build_synth_fn(model, tokenizer), recorder, redo_first=True)
    print(f"[mms] {len(rows)} utterances -> {args.outdir} (cold_load {cold_load_s:.1f}s)")


if __name__ == "__main__":
    main()
