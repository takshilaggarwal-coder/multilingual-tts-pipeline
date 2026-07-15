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

### Build phase (2026-07-15, all runs on the M2/8 GB)

- **Toolchain:** `envs/mlx` (mlx-audio 0.4.5, mlx 0.32) for MLX synthesis; `envs/eval`
  extended with torch/torchaudio/speechbrain/distillmos for MOS + speaker sim. Kokoro
  needs `misaki[en]` (+ spaCy `en_core_web_sm` installed as a wheel) for English G2P;
  Hindi uses espeak G2P (bundled via espeakng-loader).
- **English — Kokoro-82M (MLX, af_heart):** 12 clips. RTF median **0.095**, latency
  **0.29 s** full / 0.28 s first-chunk, round-trip WER **1.3 %** (only miss: en_03
  date/time). Predicted MOS Distill **4.63** / UTMOS **4.51**. 4/6 metrics real & passing;
  speaker-sim N/A (fixed voice), human MOS pending.
- **Hindi — Kokoro-82M (hf_alpha):** 12 clips. RTF median **0.092**, latency **0.48 s**.
  Round-trip WER **11.1 %** — but that is dominated by ASR: Whisper's Hindi floor on clean
  human speech is ~19 %, and the worst rows are the designed failure modes (hi_10 Hinglish
  0.41, hi_03 numbers 0.32, hi_11 digits 0.19). Need IndicConformer cross-check + a real
  Hindi anchor to separate TTS intelligibility from ASR error. Predicted MOS 4.64 / 4.26.
- **Arabic — MMS-TTS (mms-tts-ara, VITS floor):** 12 clips. RTF median **0.222**, latency
  **1.22 s**, 16 kHz. Predicted MOS Distill 4.45 / **UTMOS 3.27** — the low UTMOS flags MMS
  as the least-natural of the fixed-voice set (expected; it is the floor). WER running.
- Built `run_wer.py`, `mos.py`/`run_mos.py`, `aggregate.py`. Whisper large-v3 int8 CPU
  WER runs ~2–3 min per 12-clip set → run in background to dodge the 2-min foreground cap.
- **Fast-path (fixed-voice) tier now benchmarked for all three languages.** Latency + RTF
  pass everywhere. Next phases need human input: reference voice for cloning (own vs
  openly-licensed) → speaker-similarity + cloning models (Chatterbox/Habibi/IndicF5);
  and the human-MOS listener panel (esp. native Arabic). Checked in with the user here.
- **Arabic MMS round-trip WER: 30.2 % — fails the 10 % bar, and the transcripts show why.**
  Digit rows are deletion-dominated: ar_04's verbalized "1249" and ar_11's "542"/"12" are
  simply never spoken. MMS-tts-ara's char-level VITS tokenizer silently drops Arabic-numeral
  digits (out-of-vocab chars are discarded). ar_05 also shows proper-name mangling
  (الزهراني → مزهران). Action: add digit pre-verbalization (num2words) to the MMS text
  frontend and re-run — an honest found-it/fixed-it improvement. The naturalness ceiling of
  the floor model stands regardless (UTMOS 3.27).
- Check-in questions to the user failed to deliver (UI stream aborted) and the follow-up
  said "go on" — proceeding per the brief's make-a-call rule with the stated defaults:
  openly licensed reference clips for cloning (candidate voice swappable later) and a
  prepared listening kit for a human panel run by the candidate. Logged in ASSUMPTIONS.md.
