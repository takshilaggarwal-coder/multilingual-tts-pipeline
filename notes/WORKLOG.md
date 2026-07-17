# Worklog

Running log of what I actually did, including the dead ends. Newest at the bottom.

## 2026-07-15 — setup & survey

- Read the brief. Calls made up front: Track A (repo), everything runs locally on my
  MacBook Pro (M2, 8 GB RAM, macOS, no CUDA), openly licensed reference voices for the
  cloning tests. 8 GB is clearly going to drive every model decision.
- Machine recon: only system Python 3.9, no brew, no ffmpeg. Installed uv + Python 3.11,
  one venv per model family under `envs/` so the big frameworks never fight each other.
- Spent the first stretch surveying the current open-source TTS landscape for EN/AR/HI —
  model cards, GitHub issues, the TTS arena leaderboards, papers — and wrote it up in
  `notes/LANDSCAPE.md` + `notes/BENCHMARK_PLAN.md`. The finding that shaped everything:
  on 8 GB, MLX ports are the only safe way to run cloning models (PyTorch versions
  swap-death around 10 GB; an 8-bit MLX run of a 0.6B model peaks ~6 GB). So: mlx-audio +
  quantized weights, one model resident at a time, per-language router.
- Authored the eval text sets (12 typed sentences × 3 languages) with deliberately hard
  rows: dates/numbers, currency, names, digits/IDs, emotion, tongue twister (en),
  Hinglish code-mix (hi), greeting with religious formula (ar).
- Built the eval plumbing: per-language WER normalizers (digit verbalization on BOTH
  sides, Arabic dediacritization + alef/hamza/taa-marbuta folding, Devanagari-safe Hindi —
  WER without stated normalization is meaningless), faster-whisper round-trip WER with the
  language forced (so a mispronounced clip is penalized, not silently re-detected), and a
  warm-timing recorder that re-runs the first row to drop one-time compilation cost.

## 2026-07-15 — build & benchmark

- **English — Kokoro-82M (MLX).** First hiccup: misaki's English G2P needs spaCy's
  `en_core_web_sm` and the uv venv has no pip, so the auto-download died — installed the
  model wheel directly. After that it flew: warm RTF median **0.095**, **0.29 s** full-clip
  latency on the 10-word row, round-trip WER **1.3 %** (only miss: the "March 23rd at
  3:45 PM" date/time row). Predicted MOS 4.63 (Distill) / 4.51 (UTMOS).
- **Hindi — Kokoro (espeak G2P).** RTF 0.092, 0.48 s latency. WER came out **11.1 %**,
  which looked like a miss until I remembered Whisper's own Hindi floor on clean human
  speech is ~19 % — most of that error is the recognizer, not the TTS. Converted
  `vasista22/whisper-hindi-medium` (SPRING/IIT-M) to CTranslate2 int8 and re-scored:
  12.3 % overall but on *different* rows — it nails the number rows large-v3 flubbed
  (hi_03, hi_07 → 0.00) while reading digit strings by another convention (hi_11 → 0.58).
  Taking the min over the two ASRs per row puts TTS-attributable error around ~6 %. The
  one failure both ASRs agree on: the Hinglish row (0.41) — espeak G2P mangles inline
  Latin words in a Devanagari sentence. Real failure mode, goes in the write-up.
- **Arabic — MMS floor.** WER **30.2 %**, and the transcripts showed why: the digit rows
  were deletion-dominated. mms-tts-ara's char-level VITS tokenizer silently *drops*
  out-of-vocab characters, so "1249" is simply never spoken. Added num2words digit
  pre-verbalization in the frontend → **22.2 %**. Kept both runs for the before/after.
  Deletions collapsed (17→5) but the floor voice garbles some spoken numbers anyway, and
  it stays robotic (UTMOS 3.27). That's what floors are for.
- **Reference voices.** LibriTTS-R speaker 84 (EN), the Arabic Speech Corpus's
  professional MSA narrator (AR), SYSPIN's Hindi female voice artist (HI) — all CC BY 4.0,
  all corpora recorded for speech-synthesis research; license pages verified and recorded
  in `references/LICENSES.md`. Rejected IIT-M IndicTTS (signed-agreement license) and the
  gated AI4Bharat / Common Voice sets. Cut ~10 s cloning references at silence boundaries
  and transcribed the exact cuts, so ref_text matches the audio by construction.
- **Cloning — Chatterbox-Multilingual 4-bit (MLX).** The one 8 GB-viable model that
  clones all three languages (607 MB, peak ~2 GB). Gotcha worth recording: my warm smoke
  test with a short reference clip showed RTF ~0.3, but the real benchmark with 10 s
  references lands at RTF **~1.07** — longer conditioning costs real time. Speaker
  similarity (ECAPA cosine, calibrated against a real-vs-real ceiling and a
  real-vs-other-speaker floor): EN 0.725 (ceiling 0.819), AR 0.743 (0.851), HI **0.870**
  (0.905) — all unambiguously same-speaker. WER: EN 11.2 / AR 13.4 / HI 22.8 % — identity
  costs intelligibility, especially on Hindi.
- **Arabic specialist — Habibi-TTS MSA** (F5-TTS fine-tune, the MSA checkpoint is
  Apache-2.0). First run died: torchaudio 2.11 routes `load` through torchcodec, which
  wants ffmpeg shared libraries this machine doesn't have — shimmed `torchaudio.load` to
  soundfile and re-ran. Worth it: WER **9.4 %** (the only Arabic system under the 10 %
  bar), best Arabic cosine (0.779), predicted MOS at real-speech level. Cost: RTF ~5,
  36 s for the 10-word row on MPS — strictly offline/batch on this hardware.
- **Real-speech MOS anchors.** Scored each language's reference speaker with the same
  predictors: Arabic *human* speech gets UTMOS 3.02. Every Arabic TTS here lands at or
  above its anchor — the low absolute UTMOS numbers are the English-trained predictor's
  bias, not the audio. This reframed all the Arabic naturalness numbers; the human panel
  is the arbiter.
- **Watermark check.** While writing the disclosure table I went back to verify my own
  claim that Chatterbox outputs carry Resemble's PerTh watermark. True for the official
  PyTorch package (applied by default, no off switch) — but the MLX path has no perth
  step at all: the package isn't even installed, and there's no watermark code in
  mlx-audio's chatterbox module. My earlier README note was wrong; corrected it. The
  delivered clips are unwatermarked, and a PyTorch-path deployment would need to disclose.
- **Extra metrics** (`eval/extra_metrics.py`): hard-token WER — WER restricted to the
  names/numbers/currency/digits rows — runs **3–5× the corpus WER for every system**
  (EN Kokoro 1.3 %→4.9 %; HI Chatterbox →39 %). Corpus WER hides exactly the tokens a
  voice agent must not fumble. F0-based expressiveness (pitch spread, emotion row ÷
  neutral row) cleanly separates the flat MMS floor (~0.7–0.9) from Habibi (1.31) and
  Chatterbox-EN (1.43); Chatterbox-AR actually flattens on the emotion row (0.62).
  Hygiene check: no clipping anywhere, so the differences are real, not artefacts.
- **Listening kit** built (`eval/listening_test/`): blinded, randomized, hidden real-speech
  anchors, same-speaker A/B pairs, seed-reproducible; raters get the kit minus `key.json`.
  Panel (≥3 native/fluent listeners per language) still to run — predicted MOS stays a
  clearly-labeled proxy until then.
- **Packaging.** `make_submission.sh` → zip. The sanity check caught an 83 MB SpeechBrain
  ECAPA cache leaking into the first build — the savedir was CWD-relative, so it had
  landed inside `eval/`. Anchored it to the repo root and hardened the zip excludes;
  rebuilt clean (263 files, kit in, key out, no venvs/caches).
