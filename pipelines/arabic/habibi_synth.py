#!/usr/bin/env python
"""Habibi-TTS MSA (SWivid/Habibi-TTS, Specialized/MSA) — Arabic quality contender.

F5-TTS v1 fine-tune purpose-built for Arabic (arXiv 2601.13802). The MSA
checkpoint is Apache-2.0 (Unified/SAU/UAE variants are CC-BY-NC-SA and unused
here). Zero-shot cloning from (ref_audio, ref_text); no tashkeel required.
Flow-matching inference is compute-heavy on M2 — this is an [offline-batch]
pipeline on this hardware; latency/RTF reported as measured, not spun.

    python habibi_synth.py --texts eval/texts/ar.tsv --outdir outputs/ar/habibi \
        --ref-audio references/ar/ar_ref_10s.wav --ref-text "<transcript>" [--nfe 32]
"""
import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "pipelines" / "common"))
from synth_utils import TimingRecorder, load_texts, run_synth_loop  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--texts", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--ref-audio", required=True)
    ap.add_argument("--ref-text", required=True)
    ap.add_argument("--nfe", type=int, default=32)
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    # torchaudio 2.11 routes .load through torchcodec, which needs ffmpeg shared
    # libs (absent on this box). f5_tts calls torchaudio.load(ref_audio); shim it
    # to soundfile so no ffmpeg/torchcodec is required.
    import soundfile as sf
    import torch
    import torchaudio

    def _sf_load(path, *a, **k):
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
        return torch.from_numpy(data.T).contiguous(), sr

    torchaudio.load = _sf_load

    from huggingface_hub import hf_hub_download
    from f5_tts.api import F5TTS

    t0 = time.perf_counter()
    ckpt = hf_hub_download("SWivid/Habibi-TTS", "Specialized/MSA/model_200000.safetensors")
    vocab = hf_hub_download("SWivid/Habibi-TTS", "Specialized/MSA/vocab.txt")
    tts = F5TTS(model="F5TTS_v1_Base", ckpt_file=ckpt, vocab_file=vocab, device=args.device)
    cold_load_s = time.perf_counter() - t0

    def synth_fn(text):
        t0 = time.perf_counter()
        wav, sr, _spec = tts.infer(
            ref_file=args.ref_audio, ref_text=args.ref_text, gen_text=text,
            nfe_step=args.nfe, remove_silence=False, show_info=lambda *a, **k: None,
        )
        first_chunk_s = time.perf_counter() - t0  # batch model: first audio == full clip
        return wav, sr, first_chunk_s

    rows = load_texts(args.texts)
    recorder = TimingRecorder(
        model_name="Habibi-TTS MSA (F5v1)",
        model_ref="SWivid/Habibi-TTS Specialized/MSA model_200000",
        device=args.device,
        config={"nfe_step": args.nfe, "ref_audio": args.ref_audio, "license": "Apache-2.0 (MSA ckpt)"},
        cold_load_s=cold_load_s,
    )
    run_synth_loop(rows, args.outdir, synth_fn, recorder, redo_first=True)
    print(f"[habibi] {len(rows)} utts -> {args.outdir} (cold_load {cold_load_s:.1f}s)")


if __name__ == "__main__":
    main()
