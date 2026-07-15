#!/usr/bin/env python
"""Kokoro-82M synthesis pipeline (MLX, Apple Silicon).

Kokoro is an 82M-param, Apache-2.0 model with a fixed voice bank (no cloning).
It runs natively on MLX and is the fastest natural-sounding option on this M2.
This same script serves English (lang_code 'a', voice af_heart) and Hindi
(lang_code 'h', voices hf_alpha/hf_beta/hm_omega/hm_psi) — only the flags change.

Contract (see pipelines/common/synth_utils.py):
    python kokoro_synth.py --texts eval/texts/en.tsv --outdir outputs/en/kokoro \
        --voice af_heart --lang-code a

Writes one WAV per row + timings.json with warm, measured numbers. Timing is
end-to-end wall-clock around the synthesis call (the honest number a caller
would see), with the first row re-run warm to drop graph-compilation cost.
`first_chunk_s` is time to the first yielded audio segment (streaming latency
proxy); for a single-sentence utterance it ~= full-clip latency.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "pipelines" / "common"))

from synth_utils import TimingRecorder, load_texts, run_synth_loop  # noqa: E402


def build_synth_fn(model, voice, lang_code, speed):
    def synth_fn(text):
        first_chunk_s = None
        t0 = time.perf_counter()
        chunks = []
        sr = 24000
        for seg in model.generate(text=text, voice=voice, speed=speed, lang_code=lang_code):
            if first_chunk_s is None:
                first_chunk_s = time.perf_counter() - t0
            chunks.append(np.asarray(seg.audio, dtype=np.float32).squeeze())
            sr = seg.sample_rate
        waveform = np.concatenate(chunks) if chunks else np.zeros(1, dtype=np.float32)
        return waveform, sr, first_chunk_s

    return synth_fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--texts", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--repo", default="prince-canuma/Kokoro-82M")
    ap.add_argument("--voice", default="af_heart")
    ap.add_argument("--lang-code", default="a")
    ap.add_argument("--speed", type=float, default=1.0)
    args = ap.parse_args()

    from mlx_audio.tts.utils import load_model

    t0 = time.perf_counter()
    model = load_model(args.repo)
    cold_load_s = time.perf_counter() - t0

    rows = load_texts(args.texts)
    recorder = TimingRecorder(
        model_name=f"Kokoro-82M ({args.voice})",
        model_ref=args.repo,
        device="mlx-metal",
        config={"voice": args.voice, "lang_code": args.lang_code, "speed": args.speed},
        cold_load_s=cold_load_s,
    )
    synth_fn = build_synth_fn(model, args.voice, args.lang_code, args.speed)
    run_synth_loop(rows, args.outdir, synth_fn, recorder, redo_first=True)
    print(f"[kokoro] {len(rows)} utterances -> {args.outdir}  (cold_load {cold_load_s:.1f}s)")


if __name__ == "__main__":
    main()
