#!/usr/bin/env python
"""Batch speaker-similarity for a cloning pipeline's output directory.

    python run_sim.py --outdir ../outputs/ar/chatterbox --ref ../references/ar/ar_ref_main.wav \
        [--ref2 ../references/ar/ar_ref_02.wav] [--other ../references/en/en_ref_main.wav]

Scores ECAPA cosine(generated, reference) for every {id}.wav and writes
{outdir}/sim.json. Calibration context (cosines are embedding-specific, the 0.75
target is meaningless without it):
  --ref2   second clip of the SAME speaker  -> real-vs-real ceiling
  --other  clip of a DIFFERENT speaker      -> real-vs-other floor
"""
import argparse
import json
import statistics as st
from pathlib import Path

from mos import cosine_similarity


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--ref2", default=None, help="same-speaker clip for ceiling")
    ap.add_argument("--other", default=None, help="different-speaker clip for floor")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    wavs = sorted(p for p in outdir.glob("*.wav"))
    scores = []
    for wav in wavs:
        c = cosine_similarity(wav, args.ref)
        scores.append({"id": wav.stem, "cosine": c})
        print(f"  {wav.stem:6} cos={c:.4f}")

    result = {
        "reference": args.ref,
        "embedding": "speechbrain/spkrec-ecapa-voxceleb",
        "mean_cosine": round(st.mean(s["cosine"] for s in scores), 4) if scores else None,
        "min_cosine": min((s["cosine"] for s in scores), default=None),
        "n": len(scores),
        "utterances": scores,
    }
    if args.ref2:
        result["real_vs_real"] = cosine_similarity(args.ref, args.ref2)
        print(f"  [ceiling] real-vs-real       = {result['real_vs_real']:.4f}")
    if args.other:
        result["real_vs_other"] = cosine_similarity(args.ref, args.other)
        print(f"  [floor]   real-vs-other-spkr = {result['real_vs_other']:.4f}")

    out = Path(args.out) if args.out else outdir / "sim.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\nmean cosine={result['mean_cosine']}  -> {out}")


if __name__ == "__main__":
    main()
