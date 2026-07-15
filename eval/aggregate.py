#!/usr/bin/env python
"""Aggregate every pipeline's metrics into one results table.

Scans outputs/<lang>/<model>/ for timings.json (latency/RTF), wer.json (round-trip
WER), mos.json (predicted MOS), and sim.json (speaker similarity, cloning only),
and emits results/results.json + results/results.md keyed to the section-3 targets.
Missing files are shown as "-" so a partial run still tabulates. Run from repo root:

    python eval/aggregate.py
"""
import json
import statistics as st
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "outputs"
RES = REPO / "results"

TARGETS = {
    "mos": "≥4.0", "sim": "≥0.75", "latency_full_s": "<2.0",
    "first_chunk_s": "<0.5", "rtf": "≤0.5", "wer": "≤0.10",
}


def _load(p):
    return json.load(open(p, encoding="utf-8")) if p.exists() else None


def collect():
    rows = []
    for lang_dir in sorted(OUT.glob("*")):
        if not lang_dir.is_dir():
            continue
        lang = lang_dir.name
        for model_dir in sorted(lang_dir.glob("*")):
            if not model_dir.is_dir():
                continue
            timings = _load(model_dir / "timings.json")
            wer = _load(model_dir / "wer.json")
            mos = _load(model_dir / "mos.json")
            sim = _load(model_dir / "sim.json")
            row = {"lang": lang, "model": model_dir.name}
            if timings:
                us = timings["utterances"]
                rtfs = [u["rtf"] for u in us if u.get("rtf")]
                first = next((u for u in us if u["id"].endswith("_01")), us[0])
                row["rtf_median"] = round(st.median(rtfs), 3) if rtfs else None
                row["latency_full_s"] = first.get("gen_time_s")
                row["first_chunk_s"] = first.get("first_chunk_s")
                row["cold_load_s"] = timings.get("cold_load_s")
                row["engine"] = timings.get("model")
            if wer:
                row["wer"] = wer.get("corpus_wer")
            if mos:
                row["distill_mos"] = mos.get("mean_distill_mos")
                row["utmos"] = mos.get("mean_utmos")
                if mos.get("real_anchor"):
                    row["anchor_distill"] = mos["real_anchor"].get("distill_mos")
            if sim:
                row["speaker_cosine"] = sim.get("mean_cosine")
                row["sim_ceiling"] = sim.get("real_vs_real")
                row["sim_floor"] = sim.get("real_vs_other")
            rows.append(row)
    return rows


def fmt(v, nd=3):
    return "-" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def to_markdown(rows):
    cols = [
        ("lang", "Lang"), ("model", "Model"),
        ("distill_mos", "Distill-MOS"), ("utmos", "UTMOS"),
        ("speaker_cosine", "Spk cos"),
        ("latency_full_s", "Lat full (s)"), ("first_chunk_s", "1st chunk (s)"),
        ("rtf_median", "RTF med"), ("wer", "WER"),
    ]
    lines = ["| " + " | ".join(h for _, h in cols) + " |",
             "|" + "|".join("---" for _ in cols) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(fmt(r.get(k), 3) for k, _ in cols) + " |")
    tgt = ("\n**Targets:** MOS ≥4.0 · speaker cosine ≥0.75 · latency <2 s full / <0.5 s "
           "first-chunk · RTF ≤0.5 · round-trip WER ≤0.10. Predicted MOS (Distill/UTMOS) "
           "is an English-trained proxy — human MOS is the graded metric; for AR/HI compare "
           "against the per-language real-speech anchor, not the absolute number.")
    return "\n".join(lines) + "\n" + tgt + "\n"


def main():
    RES.mkdir(exist_ok=True)
    rows = collect()
    (RES / "results.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    (RES / "results.md").write_text(to_markdown(rows))
    print(to_markdown(rows))
    print(f"\n{len(rows)} pipeline(s) -> {RES/'results.json'}, {RES/'results.md'}")


if __name__ == "__main__":
    main()
