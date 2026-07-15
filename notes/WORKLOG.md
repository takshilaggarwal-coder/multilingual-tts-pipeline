# Worklog

Honest running log of what was actually done, including dead ends. Newest at the bottom.

## 2026-07-15

- Received brief. Decisions: Track A (repo), all runs local on MacBook Pro M2 / 8 GB RAM / macOS
  (no CUDA available), openly licensed reference voice for cloning.
- Machine recon: system Python 3.9 only, no brew/ffmpeg. Installed `uv` + Python 3.11.
  8 GB unified RAM flagged as the defining constraint for model choice.
- Kicked off a structured research sweep (web) of the current open-source TTS landscape for
  EN/AR/HI: per-language candidates, licenses, Apple-Silicon feasibility, eval tooling
  (predicted MOS, speaker-embedding similarity, per-language ASR + WER normalization),
  with adversarial fact-checking of language-support and license claims.
- Authored eval text sets (12 sentences × 3 languages): latency sentence (~10 words), numbers,
  currency, names, question, long compound, emotion, tech instructions, Hinglish code-mix (hi),
  greeting w/ religious formula (ar), tongue twister (en).
- Repo scaffolded; base eval environment being set up.
- Eval env (`envs/eval`, Python 3.11 via uv) confirmed working for the ASR+WER path:
  faster-whisper 1.2.1, jiwer, num2words, soundfile, onnxruntime all import. Still need
  torch + speechbrain for predicted-MOS and speaker-similarity.
- **Session interrupted by an API session limit mid-research.** On resume: recovered all 5
  cached research results from the workflow journal (English/Arabic/Indic/eval/runtimes)
  and persisted them as durable repo docs so the paid-for research isn't lost:
  `notes/LANDSCAPE.md` (per-language model survey + licenses + Mac feasibility),
  `notes/BENCHMARK_PLAN.md` (shortlist, eval stack, execution order, risks),
  `notes/research_results.json` (raw structured output w/ source URLs).
- Headline finding: on 8 GB, **MLX is the only safe runtime for cloning TTS** (PyTorch
  swap-deaths at ~10 GB; 8-bit MLX ~6 GB, RTF 0.54). Plan: mlx-audio + quantized weights,
  one model resident at a time. Per-language router — EN Kokoro/Chatterbox, AR
  ArTST/Habibi/Chatterbox, HI Kokoro-hi/IndicF5/Chatterbox; mms-tts as floor per language.
- Committed this milestone. Next: EN Kokoro end-to-end (real audio + latency/RTF/WER).
