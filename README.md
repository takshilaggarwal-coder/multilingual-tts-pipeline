# Multilingual Voice AI Pipeline — English · Arabic · Hindi

Take-home case study (Track A: code). Three working open-source TTS pipelines — one per
language — benchmarked for naturalness, speaker similarity (cloning), latency, real-time
factor, and round-trip intelligibility, **all runs executed for real on the hardware below**.

> **Recommended setup (one paragraph).** Per-language router, all open-source, split by
> speed vs naturalness. Speed path (EN/HI): **Kokoro-82M** (Apache-2.0, MLX) — RTF ~0.09,
> 0.3–0.5 s latency, 1.3 % English WER, but human listeners rate its prosody ~2.8/5.
> Naturalness/cloning path: **Chatterbox-Multilingual 4-bit (MLX)** — the Hindi clone of my
> own voice was the human panel's best TTS (**MOS 4.47**, A/B "same speaker", cosine 0.78)
> at RTF ~1.1. **Arabic:** **Habibi-TTS MSA** (Apache-2.0) wins accuracy (WER 9.4 %, cosine
> 0.78) but is offline-only (RTF ~5); Chatterbox is the practical middle; MMS-TTS the fast
> floor. Cloning references: my own voice (EN/HI) + the CC BY 4.0 Arabic Speech Corpus
> narrator (AR). Evaluation fully open-source: faster-whisper large-v3 (+ Hindi-finetuned
> cross-check), Distill-MOS + UTMOS as labeled proxies, SpeechBrain ECAPA cosine, and a
> 3-listener blinded P.808-style panel (hidden real anchor scored 4.67 — panel sane).
> Exact numbers: `results/results.md` + `results/human_mos.json`.

## Hardware (all numbers measured here unless labeled otherwise)

MacBook Pro, **Apple M2, 8 GB unified RAM**, macOS 15 (Darwin 25.2), no CUDA GPU.
The 8 GB budget is the defining constraint: PyTorch cloning models swap-death at this size,
so synthesis runs MLX (Apple-native) wherever possible — see `notes/LANDSCAPE.md`.

## Layout

```
pipelines/           one runnable script per model family (shared contract)
  english/kokoro_synth.py        Kokoro-82M via MLX (EN + HI, --lang-code)
  common/chatterbox_synth.py     Chatterbox-Multilingual 4-bit via MLX (EN/AR/HI, cloning)
  common/mms_synth.py            Meta MMS-TTS VITS floor (AR/HI) + digit-verbalization fix
  arabic/habibi_synth.py         Habibi-TTS MSA (F5-TTS v1 fine-tune, cloning) [offline tier]
eval/
  texts/{en,ar,hi}.tsv           12 typed eval sentences per language (latency row = *_01)
  normalizers.py                 per-language WER text normalization (disclosed, in-repo)
  asr_wer.py, run_wer.py         round-trip WER (faster-whisper, forced language)
  mos.py, run_mos.py             Distill-MOS + UTMOS predicted naturalness (proxy)
  run_sim.py                     ECAPA speaker cosine + ceiling/floor calibration
  make_ref_cuts.py               ~10 s cloning refs + exact transcripts
  aggregate.py                   all metrics -> results/results.{md,json}
  listening_test/                blinded human-MOS kit generator + scorer
references/                      CC BY 4.0 reference speakers + LICENSES.md (provenance)
outputs/<lang>/<model>/          generated WAVs + timings.json + wer.json + mos.json (+ sim.json)
results/                         aggregated tables
notes/                           LANDSCAPE.md (model survey), BENCHMARK_PLAN.md, WORKLOG.md
```

## Reproduce

```bash
# 0) toolchain: uv + python 3.11 (no brew needed)
uv venv --python 3.11 envs/mlx  && uv pip install --python envs/mlx/bin/python \
    mlx-audio "misaki[en]" num2words torch transformers \
    https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
uv venv --python 3.11 envs/eval && uv pip install --python envs/eval/bin/python \
    faster-whisper jiwer num2words soundfile torch torchaudio speechbrain distillmos whisper-normalizer
uv venv --python 3.11 envs/f5   && uv pip install --python envs/f5/bin/python f5-tts

# 1) synthesis (each writes WAVs + timings.json)
envs/mlx/bin/python pipelines/english/kokoro_synth.py  --texts eval/texts/en.tsv --outdir outputs/en/kokoro --voice af_heart --lang-code a
envs/mlx/bin/python pipelines/english/kokoro_synth.py  --texts eval/texts/hi.tsv --outdir outputs/hi/kokoro --voice hf_alpha --lang-code h
envs/mlx/bin/python pipelines/common/mms_synth.py      --model-id facebook/mms-tts-ara --texts eval/texts/ar.tsv --outdir outputs/ar/mms_vd --lang ar --verbalize-digits
envs/eval/bin/python eval/make_ref_cuts.py             # cloning refs + transcripts (once)
for L in en ar hi; do envs/mlx/bin/python pipelines/common/chatterbox_synth.py \
    --texts eval/texts/$L.tsv --outdir outputs/$L/chatterbox --lang-code $L \
    --ref-audio "$(python3 -c "import json;print(json.load(open('references/transcripts.json'))['$L']['ref_audio'])")" \
    --ref-text  "$(python3 -c "import json;print(json.load(open('references/transcripts.json'))['$L']['ref_text'])")" ; done
envs/f5/bin/python pipelines/arabic/habibi_synth.py    --texts eval/texts/ar.tsv --outdir outputs/ar/habibi \
    --ref-audio references/ar/ar_ref_10s.wav --ref-text "$(python3 -c "import json;print(json.load(open('references/transcripts.json'))['ar']['ref_text'])")"

# 2) evaluation (from eval/)
cd eval
for D in ../outputs/en/kokoro ../outputs/en/chatterbox; do ../envs/eval/bin/python run_wer.py --outdir $D --texts texts/en.tsv --lang en; done
# ... same pattern for ar/hi dirs; then:
../envs/eval/bin/python run_mos.py --outdir <dir> --texts texts/<lang>.tsv
../envs/eval/bin/python run_sim.py --outdir <cloning dir> --ref ../references/<lang>/<ref> --ref2 ... --other ...
../envs/eval/bin/python aggregate.py          # -> results/results.md

# 3) human listening test
python3 eval/listening_test/make_kit.py        # blinded kit; send kit/ minus key.json
python3 eval/listening_test/score_kit.py <filled sheets>
```

## Results

See **`results/results.md`** (regenerate with `eval/aggregate.py`). Human-MOS collection
status and per-metric analysis live in the write-up (`notes/` + submission doc).

## Disclosure

- Core generation: open-source models only (Kokoro Apache-2.0; Chatterbox MIT upstream /
  Apache-2.0 4-bit MLX repo; Habibi MSA checkpoint Apache-2.0; MMS CC-BY-NC-4.0, used only
  as a disclosed floor/control). **Watermark:** upstream PyTorch Chatterbox applies Resemble's
  PerTh watermark by default, but the 4-bit MLX path used here does not (verified — no `perth`
  install, no watermark code in mlx-audio's chatterbox module), so the delivered clips are
  unwatermarked. A production deployment on the PyTorch path would watermark, and that must be
  disclosed. Full component license table: `notes/LIMITATIONS_AND_DISCLOSURE.md`.
- Evaluation: fully open-source (no closed APIs anywhere).
- Every benchmark number was produced by real runs on the hardware named above. The
  honest, dated log of what was done — including dead ends — is in `notes/WORKLOG.md`.
