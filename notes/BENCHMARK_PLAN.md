# Benchmark plan & model shortlist

Derived from `LANDSCAPE.md`. This is the decision doc: what we benchmark, in what order,
against the section-3 targets (MOS ≥ 4.0; speaker cosine ≥ 0.75 or clearly same-speaker;
latency < 500 ms first chunk / < 2 s full clip; RTF ≤ 0.5; round-trip WER ≤ 10 %).

## Strategy: per-language router, MLX-first

No single open model wins all three languages on 8 GB. We benchmark a **fast fixed-voice
model** and a **cloning model** per language, then recommend a per-language winner. Every
number is measured on the M2/8 GB unless labelled `[T4-stretch]` or `[offline-only]`.

## Per-language shortlist (what actually gets run here)

| Lang | Primary (fast, ship-it) | Cloning candidate | Challengers | Floor / control |
|---|---|---|---|---|
| **EN** | **Kokoro-82M** (MLX) | **Chatterbox-Multilingual 4-bit** (MLX) | NeuTTS Air (GGUF), Qwen3-TTS-0.6B (MLX) | piper/mms-eng |
| **AR** | **ArTST v2** (MSA, no clone) or Kokoro-ar (n/a) | **Habibi-TTS** (Apache MSA ckpt); Chatterbox-ML | XTTS-v2 `[offline]` | **mms-tts-ara** (VITS) |
| **HI** | **Kokoro-82M hi voices** (MLX) | **IndicF5** `[offline, slow]`; Chatterbox-ML | Indic-Parler `[offline]` | **mms-tts-hin** (VITS) |

Notes:
- **Chatterbox-Multilingual 4-bit MLX is the linchpin** — if it runs on 8 GB it gives
  cloning for **all three** languages from one model, which is the cleanest story. Validate
  its memory + per-language quality first.
- Kokoro has no cloning, so it competes on **naturalness + latency + WER**, not speaker
  similarity. That is fine — the brief asks for the most human + fastest; a fixed-voice
  model can win those two even if it can't clone.
- For cloning speaker-similarity, the reference clip's language should match the target
  language (cross-lingual clone degrades SIM).

## Eval stack (locked)

| Metric | Tool | Install target |
|---|---|---|
| Predicted MOS | Distill-MOS (primary) + UTMOS22 (torch.hub) | `envs/eval` (+torch CPU) |
| Speaker sim | SpeechBrain ECAPA `spkrec-ecapa-voxceleb`, cosine | `envs/eval` (+torch CPU) |
| Round-trip WER | faster-whisper `large-v3` int8, forced lang | `envs/eval` ✓ (installed) |
| WER (HI cross-check) | IndicConformer / IndicWhisper | `envs/eval` (later) |
| Latency / RTF | in-pipeline `TimingRecorder`, warm | per-pipeline venv ✓ |
| Human MOS | P.808-style ACR kit, ≥3 listeners/lang | `eval/listening_test/` (generator) |

- **Latency:** warm (model resident) synthesis-call → first chunk (streaming) or full
  waveform (batch); cold model-load reported separately. Measured on the `*_01` ~10-word row.
- **Speaker-sim calibration:** report real-vs-real (ceiling) and real-vs-different-speaker
  (floor) cosines per embedding+language so the 0.75 target is interpreted correctly.

## Execution order (each step commits real artifacts)

1. **[done]** scaffold, eval text sets, normalizers, ASR-WER + timing harness, research.
2. **EN Kokoro end-to-end** — install `mlx-audio`, generate 12 clips, measure latency/RTF,
   run faster-whisper WER. First real audio + 3/6 metrics. ← next
3. **Eval torch stack** — Distill-MOS + UTMOS + ECAPA into `envs/eval`; add predicted-MOS
   + speaker-sim to the harness; back-fill EN Kokoro's MOS.
4. **Reference clips** — acquire openly-licensed EN/AR/HI speaker samples (CC0/CC-BY),
   log provenance in `references/LICENSES.md`. (Option to swap in candidate's own voice.)
5. **EN cloning** — Chatterbox-ML 4-bit MLX; clone ref; full 6 metrics; A/B vs Kokoro.
6. **HI** — Kokoro-hi (fast path) + IndicF5 (cloning, offline) + mms-tts-hin floor.
7. **AR** — mms-tts-ara floor + ArTST + Habibi/Chatterbox; watch tashkeel + digit reading.
8. **Full results tables, listening kit, README + comparison write-up, failure modes.**

## Risks to validate before burning hours (cheapest first)

1. **`mlx-community/chatterbox-4bit` actually fits 8 GB and clones AR+HI acceptably.**
   Biggest single unknown; the whole "one cloning model" story hinges on it. Test memory
   with `/usr/bin/time -l` and listen to AR/HI output early.
2. **IndicF5 RTF on M2 may be ≫ 1** (F5 flow-matching, MPS partial). Likely `[offline-only]`;
   don't promise Hindi cloning latency locally — measure and label.
3. **Whisper Hindi WER floor (~19 %)** can dominate round-trip WER and mask TTS quality →
   need IndicConformer cross-check before declaring a Hindi WER failure.
4. **Arabic digit/number reading** (`ar_04`, `ar_11`) and **Hinglish** (`hi_10`) are the
   likely failure modes — these rows are in the eval set on purpose.
5. **torch + MLX coexistence / disk**: separate venvs per pipeline; only one model loaded
   at a time; keep HF cache on the 193 GB volume.
6. **Human MOS needs real listeners** (esp. native Arabic) — surface early so recruitment
   runs in parallel; predicted MOS is not a substitute.

## Disclosure
Core generation = open-source models only. Entire eval stack = open-source. Non-commercial
weights (F5/XTTS/Fish/OuteTTS/mms/NISQA) are flagged per model and are acceptable for a
job-application demo **with disclosure**; permissive (Apache/MIT) models are preferred where
quality is comparable. Claude Code (AI assistant) used for research/code/automation per the
brief; all runs executed for real on the M2.
