#!/usr/bin/env python
"""Chatterbox Multilingual (Resemble AI) via MLX 4-bit — the cloning candidate.

One 607 MB Apache-2.0 checkpoint (mlx-community/chatterbox-4bit) covering 23
languages incl. English, Arabic, Hindi, with zero-shot voice cloning from a
reference clip + its transcript. Measured on this M2/8GB: warm RTF ~0.3,
peak ~2 GB — the only 8GB-viable model spanning all three languages w/ cloning.

    python chatterbox_synth.py --texts eval/texts/ar.tsv --outdir outputs/ar/chatterbox \
        --lang-code ar --ref-audio references/ar/ar_ref_main.wav --ref-text "<transcript>"

Notes:
- ref_text must transcribe ref_audio (Chatterbox conditions on both).
- Reference should be same-language for the similarity metric; cross-lingual
  cloning works but degrades SIM (worth measuring separately).
- Base Chatterbox outputs carry Resemble's PerTh perceptual watermark (disclosed).
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "pipelines" / "common"))
from synth_utils import TimingRecorder, load_texts, run_synth_loop  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--texts", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--repo", default="mlx-community/chatterbox-4bit")
    ap.add_argument("--lang-code", required=True)
    ap.add_argument("--ref-audio", required=True)
    ap.add_argument("--ref-text", required=True)
    args = ap.parse_args()

    from mlx_audio.tts.utils import load_model

    t0 = time.perf_counter()
    model = load_model(args.repo)
    cold_load_s = time.perf_counter() - t0

    def synth_fn(text):
        first_chunk_s = None
        t0 = time.perf_counter()
        chunks, sr = [], 24000
        for seg in model.generate(text=text, lang_code=args.lang_code,
                                  ref_audio=args.ref_audio, ref_text=args.ref_text):
            if first_chunk_s is None:
                first_chunk_s = time.perf_counter() - t0
            chunks.append(np.asarray(seg.audio, dtype=np.float32).squeeze())
            sr = seg.sample_rate
        wav = np.concatenate(chunks) if chunks else np.zeros(1, dtype=np.float32)
        return wav, sr, first_chunk_s

    rows = load_texts(args.texts)
    recorder = TimingRecorder(
        model_name=f"Chatterbox-ML 4bit ({args.lang_code})",
        model_ref=args.repo,
        device="mlx-metal",
        config={"lang_code": args.lang_code, "ref_audio": args.ref_audio,
                "quant": "4bit"},
        cold_load_s=cold_load_s,
    )
    run_synth_loop(rows, args.outdir, synth_fn, recorder, redo_first=True)
    print(f"[chatterbox] {len(rows)} utts -> {args.outdir} (cold_load {cold_load_s:.1f}s)")


if __name__ == "__main__":
    main()
