#!/usr/bin/env python
"""Batch round-trip WER for one pipeline's output directory.

    python run_wer.py --outdir ../outputs/en/kokoro --texts texts/en.tsv --lang en

Transcribes every {id}.wav with faster-whisper (forced language), normalizes both
sides with the language-appropriate normalizer, and writes {outdir}/wer.json with
per-utterance detail + the error-weighted corpus WER. Run from the eval/ dir (so
`normalizers` and `asr_wer` import cleanly) in the eval venv.
"""
import argparse
import csv
import json
from pathlib import Path

from asr_wer import corpus_wer, score_utterance


def load_texts(tsv_path):
    with open(tsv_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True, help="dir with {id}.wav files")
    ap.add_argument("--texts", required=True)
    ap.add_argument("--lang", required=True, choices=["en", "ar", "hi"])
    ap.add_argument("--model", default="large-v3")
    ap.add_argument("--out", default=None, help="defaults to {outdir}/wer.json")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    rows = load_texts(args.texts)
    scores = []
    for row in rows:
        wav = outdir / f"{row['id']}.wav"
        if not wav.exists():
            print(f"  [skip] {row['id']}: no wav")
            continue
        s = score_utterance(row["text"], wav, args.lang, args.model)
        s["id"] = row["id"]
        s["type"] = row.get("type")
        scores.append(s)
        print(f"  {row['id']:6} WER={s['wer']:.3f}  ({s['substitutions']}S {s['deletions']}D {s['insertions']}I / {s['ref_words']}w)")

    result = {
        "lang": args.lang,
        "asr_model": args.model,
        "corpus_wer": corpus_wer(scores),
        "n_utterances": len(scores),
        "utterances": scores,
    }
    out = Path(args.out) if args.out else outdir / "wer.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\ncorpus WER ({args.lang}, {len(scores)} utts): {result['corpus_wer']:.4f}  -> {out}")


if __name__ == "__main__":
    main()
