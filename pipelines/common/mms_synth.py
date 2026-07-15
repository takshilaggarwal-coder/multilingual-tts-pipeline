#!/usr/bin/env python
"""Meta MMS-TTS synthesis (facebook/mms-tts-*), the always-works VITS floor.

Single-pass VITS, no cloning, tiny + faster-than-realtime on CPU. Used as the
per-language control/floor (mms-tts-ara, mms-tts-hin). License: CC-BY-NC-4.0
(non-commercial) — disclosed. Runs via transformers in the mlx venv (torch present).

    python mms_synth.py --model-id facebook/mms-tts-ara --texts eval/texts/ar.tsv \
        --outdir outputs/ar/mms --lang ar

--verbalize-digits: pre-verbalize digits with num2words before synthesis.
Found the hard way: mms-tts-ara's char-level VITS tokenizer silently DROPS
out-of-vocab characters, so raw digits ("1249") are simply never spoken
(outputs/ar/mms wer.json, ar_04/ar_11 deletion-dominated). Verbalizing in the
text frontend is the fix; both runs are kept for the before/after comparison.
"""
import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

_AR_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
_DEV_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
_NUM = re.compile(r"\d+")


def verbalize_digits(text, lang):
    from num2words import num2words
    text = text.translate(_AR_DIGITS).translate(_DEV_DIGITS)

    def repl(m):
        try:
            return " " + num2words(int(m.group()), lang=lang) + " "
        except (NotImplementedError, OverflowError, ValueError):
            return m.group()

    return re.sub(r"\s+", " ", _NUM.sub(repl, text)).strip()

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
    ap.add_argument("--verbalize-digits", action="store_true",
                    help="num2words digit verbalization in the text frontend")
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
    if args.verbalize_digits:
        for row in rows:
            row["text"] = verbalize_digits(row["text"], args.lang)
    recorder = TimingRecorder(
        model_name=f"MMS-TTS ({args.lang})" + (" +digitverb" if args.verbalize_digits else ""),
        model_ref=args.model_id,
        device="cpu",
        config={"sampling_rate": model.config.sampling_rate,
                "verbalize_digits": args.verbalize_digits},
        cold_load_s=cold_load_s,
    )
    run_synth_loop(rows, args.outdir, build_synth_fn(model, tokenizer), recorder, redo_first=True)
    print(f"[mms] {len(rows)} utterances -> {args.outdir} (cold_load {cold_load_s:.1f}s)")


if __name__ == "__main__":
    main()
