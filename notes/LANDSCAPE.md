# Open-source TTS landscape for EN / AR / HI on Apple M2 (8 GB), mid-2026

This is the survey that drives model selection. It comes from a structured web sweep
(5 parallel researchers: English, Arabic, Indic, eval tooling, Apple-Silicon runtimes)
run on 2026-07-15, with license and language-support claims flagged for verification.
Claims here are **verified empirically as each model is actually installed and run** —
see `results/` and the per-pipeline notes for what held up. Where a claim has not yet
been re-checked on this machine, it is marked _(unverified)_.

## The defining constraint: 8 GB unified memory

The single most important finding: **on an M2 with 8 GB, MLX is effectively the only
safe runtime for cloning-capable TTS.** A documented 8 GB run of a 0.6B model showed
PyTorch peaking ~10 GB (swap death) versus ~6 GB peak and RTF 0.54 under 8-bit MLX.
So the architecture is:

- **`mlx-audio` (Blaizzy, MIT)** as the primary TTS runtime, with quantized
  `mlx-community` weights, **one model resident at a time**, generation serialized
  (MLX Metal is reported to crash on concurrent ops).
- PyTorch models (XTTS, IndicF5, Indic-Parler, CosyVoice) are treated as **offline
  batch-only** on this Mac — usable to *generate* clips for the benchmark, but their
  Mac latency/RTF is not representative and is labelled as such.
- Anything ≥3B or CUDA-oriented (Veena, Step-Audio-EditX, Higgs Audio v3, Dia-1.6B,
  snorTTS) is out of scope locally and noted as a **Colab T4 stretch** candidate.

## English

| Model | License | Clone | Mac (M2 8 GB) fit | Notes |
|---|---|---|---|---|
| **Kokoro-82M** | Apache-2.0 | ✗ | **Best-in-class** — MLX/ONNX/CoreML ports, RTF ≪ 1, ~90 ms TTFA class | 82M params. No cloning, fixed voice bank. The comfortable naturalness+speed floor. |
| **Chatterbox / -Turbo / -Multilingual** | MIT (PerTh watermark) | ✓ | **Feasible via MLX 4-bit** (`mlx-community/chatterbox-4bit`, 607 MB) | Multilingual V3 covers 23 langs incl. AR+HI. PyTorch-on-Mac path has MPS bugs + memory leak (issue #218) → use MLX. |
| **NeuTTS Air** (Neuphonic) | Apache-2.0 | ✓ | **Excellent** — GGUF/llama.cpp, RTF < 0.5 on CPU | Built for exactly this hardware class. English-primary. 3–15 s reference clone. |
| **Qwen3-TTS-0.6B** (Alibaba) | Apache-2.0 | ✓ | Strong via 8-bit MLX (RTF 0.54, ~6 GB peak on 8 GB M1 — documented) | Best-documented 8 GB cloning datapoint. AR/HI unverified in base. |
| **CSM-1B** (Sesame) | Apache-2.0 | ✓ | "nearly real-time on M2 Air" (quantized) | English only. Cloning via conversation context. |
| **F5-TTS** | code MIT, **weights CC-BY-NC-4.0** | ✓ | `f5-tts-mlx`, ~RTF 1 on M2 | Non-commercial weights. EN/ZH base. Basis for Habibi (AR) + IndicF5 (HI). |
| **Fish-Speech / OpenAudio S1-mini** | **CC-BY-NC-SA-4.0** | ✓ | MPS-supported, ~real-time or slower | Non-commercial. AR tier-2. |

Arena context (Artificial Analysis speech arena, open-weights tier, Jul 2026):
Step-Audio-EditX (ELO 1118, CUDA-only), Fish S2 Pro (1110, non-commercial), **Kokoro-82M
(1060)**, Maya1 (1053), **Chatterbox (1011–1123)**. No open model cracks the closed top-10;
gap ~80 ELO.

## Arabic (MSA + dialect bonus)

The "top natural models skip Arabic" claim is **now partly outdated**:

| Model | License | Clone | Mac fit | Notes |
|---|---|---|---|---|
| **Habibi-TTS** (F5 team, Jan 2026, arXiv 2601.13802) | code MIT; MSA/EGY/IRQ/ALG/MAR **Apache-2.0**; Unified/SAU/UAE CC-BY-NC-SA | ✓ | F5 MPS path, ~RTF 1–3 on M2, ~2–3 GB | Purpose-built Arabic, 11 dialects, **no tashkeel required**, zero-shot clone. Primary AR candidate _(claims to beat ElevenLabs v3-alpha on dialect WER/SIM — unverified)_. |
| **Chatterbox-Multilingual** | MIT | ✓ | Borderline (MLX 4-bit) | AR shipped Sep 2025, cross-lingual clone. Best permissively-licensed multi-lang fallback. |
| **ArTST v2** (MBZUAI) | MIT | ✗ | **Excellent** — SpeechT5-size, realtime on M2 | MSA + Classical + 17 dialects. Lowest-risk local, no cloning. |
| **OuteTTS-1.0-1B** | CC-BY-NC-SA + Llama-3.2 license | ✓ | **Best 8 GB fit** — GGUF/llama.cpp, near-realtime | AR in high-data tier, auto word-alignment (raw text in). Non-commercial. |
| **XTTS-v2** (Coqui) | **CPML (non-commercial; Coqui defunct)** | ✓ | CPU-only (MPS broken #3649), RTF several× | 17 langs incl. AR+HI. Offline-only on Mac. |
| **facebook/mms-tts-ara** | **CC-BY-NC-4.0** | ✗ | Trivial — VITS, faster-than-realtime | Zero-risk "always works" floor. |

Diacritization: Habibi/Chatterbox/XTTS do **not** require tashkeel; classic VITS quality
improves with it. Tools if needed: CAMeL Tools, Farasa, Mishkal, `catt`.

## Hindi (+ Indic bonus)

| Model | License | Clone | Mac fit | Notes |
|---|---|---|---|---|
| **Kokoro-82M** (hi voices: hf_alpha/beta, hm_omega/psi) | Apache-2.0 | ✗ | **Comfortable** — realtime on CPU/MLX | Only fully-comfortable local Hindi model. Fixed voices, no clone. |
| **IndicF5** (AI4Bharat) | MIT | ✓ | Fits RAM, **slow off-CUDA** (F5 flow-matching, MPS partial → RTF ≫ 1) | Best open Hindi **cloning** quality. **Garbles raw Latin-script Hinglish → transliterate English words to Devanagari first.** 11 Indic langs. |
| **Indic Parler-TTS** (AI4Bharat) | Apache-2.0 | ✗ | Loads in fp16, ~10–15 s/sentence off-GPU | 21 Indic langs (Tamil/Telugu/Bengali bonus). Hindi MOS-like ~84.8. Offline-only. |
| **Chatterbox-Multilingual** | MIT | ✓ | Borderline (MLX 4-bit) | HI + clone; the one model spanning EN+AR+HI. |
| **facebook/mms-tts-hin** | **CC-BY-NC-4.0** | ✗ | Trivial — VITS | Zero-risk floor; siblings for 1100+ langs. |
| Veena / snorTTS-Indic | Apache-2.0 | ✓/✗ | **CUDA-only** (3–4B) | Colab T4 stretch only. |

## Evaluation stack (all open-source, CPU on M2)

- **Predicted MOS:** `Distill-MOS` (MIT, CPU-fast) as primary; `UTMOS22-strong` via
  `tarepan/SpeechMOS` (torch.hub) for literature comparability; optional NISQA
  (weights CC-BY-NC-SA). **All English-trained** → for AR/HI report deltas vs. a
  real-speech anchor and disclose that absolute cross-language MOS is not valid
  (IndicMOS showed open predictors degrade on Indian languages). Predicted MOS is a
  proxy, **not** a substitute for the human panel.
- **Speaker similarity:** SpeechBrain **ECAPA-TDNN** (`spkrec-ecapa-voxceleb`,
  Apache-2.0). Note: cosine thresholds like 0.75 are **not** comparable across embedding
  models; calibrate a same-speaker band per language (real-vs-real ceiling, real-vs-other
  floor). `microsoft/wavlm-base-plus-sv` as optional second opinion.
- **Round-trip WER ASR:** `faster-whisper large-v3` (CT2, int8, forced language) as
  primary for all three; **IndicConformer/IndicWhisper** cross-check for Hindi (vanilla
  Whisper Hindi WER ~19 % would otherwise dominate the metric). whisper.cpp+Metal for speed.
- **WER normalization:** English → Whisper `EnglishTextNormalizer`; Arabic → CAMeL-style
  dediacritization + alef/hamza/taa-marbuta folding; Hindi → indic-nlp + Devanagari-safe
  normalizer. **Never** run Whisper `BasicTextNormalizer` on Devanagari (strips matras).
- **Human MOS:** ITU-T P.808-style ACR, 3–5 listeners/language, blinded + randomized,
  real-speech + anchor clips hidden in the set.

## Sources
Full source URLs per candidate are preserved in
`notes/research_results.json` (the raw structured research output). Key anchors:
Artificial Analysis / HF TTS-Arena leaderboards; model HF cards + GitHub issues cited
inline above (Chatterbox #218, Coqui #3649, IndicF5 Hinglish behaviour, Qwen3-TTS 8 GB MLX
report); arXiv 2601.13802 (Habibi-TTS).
