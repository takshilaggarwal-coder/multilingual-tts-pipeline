"""Shared helpers for pipeline synth scripts.

Every pipeline script is standalone (runs in its own venv) and follows the contract:

    python <model>_synth.py --texts eval/texts/<lang>.tsv --outdir outputs/<lang>/<model> [--ref <wav>]

It must write one WAV per text row plus a `timings.json` with real measured numbers.
Timing is measured *warm*: the model is loaded once (cold load reported separately),
then each row is synthesized in a loop. The first row is re-run at the end and only
the re-run kept, so no row carries first-call compilation overhead.
"""
import csv
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf


def load_texts(tsv_path):
    with open(tsv_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def machine_info():
    try:
        chip = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
    except Exception:
        chip = platform.processor()
    return {
        "chip": chip,
        "platform": platform.platform(),
        "python": sys.version.split()[0],
    }


def save_wav(path, waveform, sample_rate):
    waveform = np.asarray(waveform, dtype=np.float32).squeeze()
    peak = np.abs(waveform).max()
    if peak > 1.0:  # some models emit slightly clipped float audio
        waveform = waveform / peak
    sf.write(str(path), waveform, sample_rate)
    return len(waveform) / sample_rate


class TimingRecorder:
    """Collects per-utterance timing; call `record` around each synthesis."""

    def __init__(self, model_name, model_ref, device, config, cold_load_s):
        self.meta = {
            "model": model_name,
            "model_ref": model_ref,
            "device": device,
            "config": config,
            "cold_load_s": round(cold_load_s, 3),
            "machine": machine_info(),
            "utterances": [],
        }

    def record(self, row_id, text, gen_time_s, audio_len_s, first_chunk_s=None):
        self.meta["utterances"].append({
            "id": row_id,
            "text": text,
            "gen_time_s": round(gen_time_s, 4),
            "audio_len_s": round(audio_len_s, 3),
            "rtf": round(gen_time_s / audio_len_s, 4) if audio_len_s > 0 else None,
            "first_chunk_s": round(first_chunk_s, 4) if first_chunk_s is not None else None,
        })

    def write(self, outdir):
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        with open(outdir / "timings.json", "w", encoding="utf-8") as f:
            json.dump(self.meta, f, indent=2, ensure_ascii=False)


def run_synth_loop(rows, outdir, synth_fn, recorder, redo_first=True):
    """synth_fn(text) -> (waveform, sample_rate, first_chunk_s | None)"""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    def synth_one(row):
        t0 = time.perf_counter()
        waveform, sr, first_chunk_s = synth_fn(row["text"])
        gen_time = time.perf_counter() - t0
        audio_len = save_wav(outdir / f"{row['id']}.wav", waveform, sr)
        return gen_time, audio_len, first_chunk_s

    for row in rows:
        gen_time, audio_len, first_chunk_s = synth_one(row)
        recorder.record(row["id"], row["text"], gen_time, audio_len, first_chunk_s)

    if redo_first and rows:
        # the first utterance often pays one-time compilation cost; re-run it warm
        # and keep only the warm measurement
        row = rows[0]
        gen_time, audio_len, first_chunk_s = synth_one(row)
        recorder.meta["utterances"] = [
            u for u in recorder.meta["utterances"] if u["id"] != row["id"]
        ]
        recorder.record(row["id"], row["text"], gen_time, audio_len, first_chunk_s)
    recorder.write(outdir)
