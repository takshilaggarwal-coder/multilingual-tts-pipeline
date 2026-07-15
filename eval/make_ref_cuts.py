#!/usr/bin/env python
"""Prepare cloning-ready reference cuts + transcripts.

Chatterbox and F5-family models want a short reference (~10 s) plus its exact
transcript. The AR/HI corpus clips run 16-29 s, so: cut each <lang>_ref_main.wav
at the quietest 100 ms window in the 8-12 s range (never mid-word), save as
<lang>_ref_10s.wav, then transcribe THE CUT with faster-whisper large-v3
(forced language) so transcript and audio match by construction. EN ref is
already 10.0 s with a corpus transcript, but is transcribed the same way for
consistency. Results -> references/transcripts.json (used by pipeline scripts).

    python make_ref_cuts.py     # run from eval/ in the eval venv
"""
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from asr_wer import transcribe  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
REFS = REPO / "references"
CUT_RANGE_S = (8.0, 12.0)
WIN_S = 0.1


def quiet_cut_point(wav, sr):
    lo, hi = int(CUT_RANGE_S[0] * sr), min(int(CUT_RANGE_S[1] * sr), len(wav))
    if hi - lo < int(WIN_S * sr):
        return len(wav)
    win = int(WIN_S * sr)
    seg = wav[lo:hi]
    # RMS per sliding window (hop = win/2), pick the quietest
    hops = range(0, len(seg) - win, win // 2)
    rms = [(np.sqrt((seg[i:i + win] ** 2).mean()), i) for i in hops]
    _, best = min(rms)
    return lo + best + win // 2


def main():
    out = {}
    for lang in ("en", "ar", "hi"):
        main_wav = REFS / lang / f"{lang}_ref_main.wav"
        wav, sr = sf.read(main_wav, dtype="float32")
        dur = len(wav) / sr
        if dur > 12.5:
            cut = quiet_cut_point(wav, sr)
            cut_path = REFS / lang / f"{lang}_ref_10s.wav"
            sf.write(cut_path, wav[:cut], sr)
            print(f"[{lang}] cut {dur:.1f}s -> {cut/sr:.1f}s at quiet point")
        else:
            cut_path = main_wav
            print(f"[{lang}] main ref already {dur:.1f}s, no cut")
        text = transcribe(cut_path, lang, "large-v3")
        out[lang] = {"ref_audio": str(cut_path.relative_to(REPO)), "ref_text": text,
                     "sr": sr, "duration_s": round(len(sf.read(cut_path)[0]) / sr, 2)}
        print(f"     transcript: {text}")

    with open(REFS / "transcripts.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n-> {REFS/'transcripts.json'}")


if __name__ == "__main__":
    main()
