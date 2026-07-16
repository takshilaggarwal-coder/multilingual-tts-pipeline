#!/usr/bin/env python
"""Extra evaluation metrics beyond the section-3 six.

The brief invites "metrics you think matter" (prosody, names/numbers, GPU cost).
These are the ones that actually changed how I read the results, computed from
audio + the existing wer.json (no new models, so they're cheap to re-run):

  1. Hard-token accuracy  -- WER restricted to the rows that carry names, numbers,
     currency, digits and acronyms. Corpus WER hides these; a model can sound
     lovely and still misread "$1,249.99" or a flight number, which is exactly
     what breaks a real voice agent.
  2. Expressiveness (prosody)  -- pitch (F0) variation, and the ratio of F0 spread
     on the emotion row vs the neutral row. A flat reader and an expressive one
     can tie on MOS/WER but feel completely different; this separates them.
  3. Audio hygiene  -- % clipped samples and edge silence. Catches the ugly
     artifacts (clicks, dead air, cut-offs) that a MOS average smooths over.

Run from eval/ in the eval venv:  python extra_metrics.py
"""
import json
import statistics as st
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "outputs"
HARD_TYPES = {"numbers", "currency", "names", "digits", "acronym"}
F0_MIN, F0_MAX = 80.0, 400.0  # plausible voiced range for adult speech


def _load(path):
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return data[:, 0], sr  # first channel


def f0_stats(wav, sr):
    """Voiced-frame F0 mean/std/range via torchaudio's detector."""
    x = torch.from_numpy(wav).unsqueeze(0)
    try:
        f0 = torchaudio.functional.detect_pitch_frequency(x, sr).squeeze().numpy()
    except Exception:
        return {}
    voiced = f0[(f0 >= F0_MIN) & (f0 <= F0_MAX)]
    if voiced.size < 3:
        return {"f0_voiced_frac": round(float(voiced.size / max(f0.size, 1)), 3)}
    return {
        "f0_mean": round(float(voiced.mean()), 1),
        "f0_std": round(float(voiced.std()), 1),
        "f0_range": round(float(voiced.max() - voiced.min()), 1),
        "f0_voiced_frac": round(float(voiced.size / f0.size), 3),
    }


def hygiene(wav, sr):
    peak = float(np.abs(wav).max()) if wav.size else 0.0
    clipped = float(np.mean(np.abs(wav) >= 0.999)) if wav.size else 0.0
    # edge silence: seconds below -40 dBFS at head/tail
    thr = 10 ** (-40 / 20)
    above = np.where(np.abs(wav) > thr)[0]
    lead = round(float(above[0] / sr), 2) if above.size else round(len(wav) / sr, 2)
    tail = round(float((len(wav) - above[-1]) / sr), 2) if above.size else 0.0
    return {"peak": round(peak, 3), "clipped_frac": round(clipped, 5),
            "lead_sil_s": lead, "tail_sil_s": tail}


def hard_token_wer(wer_path):
    if not wer_path.exists():
        return None
    w = json.load(open(wer_path, encoding="utf-8"))
    hard = [u for u in w["utterances"] if u.get("type") in HARD_TYPES]
    if not hard:
        return None
    errs = sum(u["substitutions"] + u["deletions"] + u["insertions"] for u in hard)
    words = sum(u["ref_words"] for u in hard)
    return round(errs / words, 4) if words else None


def analyze(model_dir, texts_by_id):
    rows = {}
    for wav in sorted(model_dir.glob("*.wav")):
        uid = wav.stem
        wav_data, sr = _load(wav)
        rows[uid] = {"type": texts_by_id.get(uid, "?"), **f0_stats(wav_data, sr),
                     **hygiene(wav_data, sr)}
    # expressiveness: F0 spread on the emotion row vs the neutral row
    emo = next((r for r in rows.values() if r["type"] == "emotion"), None)
    neu = next((r for r in rows.values() if r["type"] == "neutral"), None)
    expr = None
    if emo and neu and emo.get("f0_std") and neu.get("f0_std"):
        expr = round(emo["f0_std"] / neu["f0_std"], 2)
    f0_stds = [r["f0_std"] for r in rows.values() if r.get("f0_std")]
    return {
        "hard_token_wer": hard_token_wer(model_dir / "wer.json"),
        "expressiveness_emo_over_neutral": expr,
        "mean_f0_std": round(st.mean(f0_stds), 1) if f0_stds else None,
        "max_clipped_frac": round(max((r["clipped_frac"] for r in rows.values()), default=0), 5),
        "per_utt": rows,
    }


def main():
    import csv
    texts_by_id = {}
    for lang in ("en", "ar", "hi"):
        with open(REPO / "eval" / "texts" / f"{lang}.tsv", encoding="utf-8") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                texts_by_id[row["id"]] = row["type"]

    summary, results = [], {}
    for lang_dir in sorted(OUT.glob("*")):
        if not lang_dir.is_dir() or lang_dir.name.startswith("_"):
            continue
        for md in sorted(lang_dir.glob("*")):
            if not (md / "timings.json").exists():
                continue
            key = f"{lang_dir.name}/{md.name}"
            a = analyze(md, texts_by_id)
            results[key] = a
            summary.append((key, a))
            print(f"{key:18} hard-WER={a['hard_token_wer']}  "
                  f"expressiveness={a['expressiveness_emo_over_neutral']}  "
                  f"meanF0std={a['mean_f0_std']}  maxclip={a['max_clipped_frac']}")

    (REPO / "results" / "extra_metrics.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False))
    # markdown table
    lines = ["| System | Hard-token WER | Expressiveness (emo/neutral F0) | Mean F0 std (Hz) | Max clipped |",
             "|---|---|---|---|---|"]
    for key, a in summary:
        lines.append(f"| {key} | {a['hard_token_wer']} | "
                     f"{a['expressiveness_emo_over_neutral']} | {a['mean_f0_std']} | "
                     f"{a['max_clipped_frac']} |")
    note = ("\n*Hard-token WER* = WER on the names/numbers/currency/digits/acronym rows only "
            "(the tokens a voice agent must not fumble). *Expressiveness* = ratio of pitch "
            "spread on the emotion row to the neutral row; ~1.0 means a flat reader, higher "
            "means the model modulates for affect. *Max clipped* = worst per-clip fraction of "
            "samples at full scale (artefact check).\n")
    (REPO / "results" / "extra_metrics.md").write_text("\n".join(lines) + "\n" + note)
    print(f"\n-> results/extra_metrics.md, results/extra_metrics.json")


if __name__ == "__main__":
    main()
