#!/usr/bin/env python
"""Batch predicted-MOS for one pipeline's output directory.

    python run_mos.py --outdir ../outputs/en/kokoro --texts texts/en.tsv [--anchor <real.wav>]

Scores every {id}.wav with Distill-MOS + UTMOS and writes {outdir}/mos.json
(per-utterance + mean). If --anchor (a real-speech clip in the same language) is
given, it is scored too and reported as the real-speech ceiling for context —
important for Arabic/Hindi where the predictors are out-of-domain.
"""
import argparse
import csv
import json
import statistics as st
from pathlib import Path

from mos import predicted_mos


def load_texts(tsv_path):
    with open(tsv_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def _mean(vals):
    vals = [v for v in vals if v is not None]
    return round(st.mean(vals), 3) if vals else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--texts", required=True)
    ap.add_argument("--anchor", default=None, help="real-speech clip (same lang) for ceiling")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    rows = load_texts(args.texts)
    scores = []
    for row in rows:
        wav = outdir / f"{row['id']}.wav"
        if not wav.exists():
            continue
        s = predicted_mos(wav)
        s["id"] = row["id"]
        scores.append(s)
        print(f"  {row['id']:6} distill={s.get('distill_mos')}  utmos={s.get('utmos')}")

    result = {
        "outdir": str(outdir),
        "mean_distill_mos": _mean([s.get("distill_mos") for s in scores]),
        "mean_utmos": _mean([s.get("utmos") for s in scores]),
        "n": len(scores),
        "utterances": scores,
    }
    if args.anchor:
        result["real_anchor"] = {"path": args.anchor, **predicted_mos(Path(args.anchor))}
        print(f"  [anchor] {result['real_anchor']}")

    out = Path(args.out) if args.out else outdir / "mos.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\nmean Distill-MOS={result['mean_distill_mos']}  mean UTMOS={result['mean_utmos']}  -> {out}")


if __name__ == "__main__":
    main()
