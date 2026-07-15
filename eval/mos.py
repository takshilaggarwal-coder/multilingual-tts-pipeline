"""Predicted MOS + speaker similarity (open-source, CPU on M2).

Predicted MOS is an AUTOMATED PROXY, not a substitute for the human panel. Two
predictors are reported:
  - Distill-MOS (MIT, xls-r-sqa backbone) — primary, blind, CPU-fast.
  - UTMOS22-strong (tarepan/SpeechMOS via torch.hub) — the score TTS papers report,
    included for literature comparability.
Both are English-trained. For Arabic/Hindi we ALSO score the real reference clips
with the same predictors and report the TTS score as a delta below that per-language
real-speech anchor; absolute cross-language MOS is not claimed valid.

Speaker similarity uses SpeechBrain ECAPA-TDNN (spkrec-ecapa-voxceleb) cosine. The
0.75 target is embedding-specific; we also report the real-vs-real ceiling and
real-vs-different-speaker floor per language so the number is interpretable.
All models run once and are cached in-process.
"""
import numpy as np
import soundfile as sf
import torch
import torchaudio

_CACHE = {}
TARGET_SR = 16000


def _load_wav_16k(path):
    # soundfile, not torchaudio.load: torchaudio 2.11 delegates to torchcodec/ffmpeg
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)  # (samples, ch)
    wav = torch.from_numpy(data.T)  # (ch, samples)
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    if sr != TARGET_SR:
        wav = torchaudio.functional.resample(wav, sr, TARGET_SR)
    return wav.contiguous()  # (1, samples) float32


def _distillmos():
    if "distill" not in _CACHE:
        import distillmos
        m = distillmos.ConvTransformerSQAModel(load_weights=True)
        m.eval()
        _CACHE["distill"] = m
    return _CACHE["distill"]


def _utmos():
    if "utmos" not in _CACHE:
        _CACHE["utmos"] = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True)
    return _CACHE["utmos"]


def _ecapa():
    if "ecapa" not in _CACHE:
        from pathlib import Path

        from speechbrain.inference.speaker import EncoderClassifier
        # repo-anchored savedir (not CWD-relative) so the 80MB cache always lands
        # under envs/ (git-ignored) regardless of which dir the eval is launched from
        savedir = Path(__file__).resolve().parents[1] / "envs" / "eval" / ".sb_ecapa"
        _CACHE["ecapa"] = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=str(savedir),
            run_opts={"device": "cpu"},
        )
    return _CACHE["ecapa"]


@torch.no_grad()
def predicted_mos(wav_path):
    wav = _load_wav_16k(wav_path)
    out = {}
    try:
        out["distill_mos"] = round(float(_distillmos()(wav).squeeze().item()), 3)
    except Exception as e:
        out["distill_mos"] = None
        out["distill_error"] = str(e)[:120]
    try:
        out["utmos"] = round(float(_utmos()(wav, TARGET_SR).squeeze().item()), 3)
    except Exception as e:
        out["utmos"] = None
        out["utmos_error"] = str(e)[:120]
    return out


@torch.no_grad()
def speaker_embedding(wav_path):
    wav = _load_wav_16k(wav_path)
    emb = _ecapa().encode_batch(wav).squeeze()
    return emb / emb.norm()


def cosine_similarity(wav_a, wav_b):
    ea, eb = speaker_embedding(wav_a), speaker_embedding(wav_b)
    return round(float(torch.dot(ea, eb).item()), 4)
