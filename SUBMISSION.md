# Multilingual Voice AI Pipeline — submission write-up

**Track A (code).** Repo runs; every number below was produced by real runs on the named
hardware. Reproduction: `README.md`. Honest build log incl. dead ends: `notes/WORKLOG.md`.
Full licensing + limitations: `notes/LIMITATIONS_AND_DISCLOSURE.md`.

## Recommended setup (one paragraph)

Use a **per-language router of open-source models, MLX-first for Apple Silicon**, split by
whether the request needs a *fixed voice fast* or a *cloned voice*. **Fast fixed-voice path:
English and Hindi → Kokoro-82M** (Apache-2.0) — RTF ~0.09, ~0.3–0.5 s latency, 1.3 % English
WER, predicted MOS ~4.6, beating every latency/RTF/intelligibility target on a fanless laptop.
**Voice cloning, any of the three languages → Chatterbox-Multilingual (4-bit MLX)** — one
607 MB Apache/MIT checkpoint, unwatermarked on this path, same-speaker identity (Hindi cosine
0.87; English/Arabic ~0.73, i.e. 0.86–0.88 of the real-vs-real ceiling) at the honest cost of
real time (RTF ~1.07, 4–5 s clips). **Arabic** is the hard case: its only *fast* option is the
robotic MMS VITS floor, while the *quality/accuracy* winner is **Habibi-TTS MSA** (Apache-2.0,
F5 fine-tune) — best Arabic WER (9.4 %) and cloning (cosine 0.779), predicted MOS at
real-speech level — but RTF ~5 makes it offline/batch-only here. The whole evaluation stack is
open-source (faster-whisper + a Hindi-tuned Whisper cross-check for WER, Distill-MOS + UTMOS
for predicted naturalness, SpeechBrain ECAPA for speaker cosine, a blinded P.808-style human
kit). No closed API is used anywhere, for generation or evaluation.

## Hardware

MacBook Pro, **Apple M2, 8 GB unified RAM**, macOS 15, **no CUDA GPU**. All latency/RTF are
warm (model resident), synthesis-call → first audio chunk (streaming) or full waveform (batch),
measured on the ~10-word `*_01` row. Cold model-load is reported separately in each
`timings.json`. The 8 GB budget is the binding constraint — it rules PyTorch cloning models
out (they swap-death near 10 GB) and forces the MLX-first design.

## Results vs. the section-3 targets

Full machine-readable table: `results/results.md` (regenerate with `eval/aggregate.py`).
Targets: MOS ≥ 4.0 · speaker cosine ≥ 0.75 (see note) · latency < 2 s full / < 0.5 s first
chunk · RTF ≤ 0.5 · round-trip WER ≤ 0.10.

| Lang | Model (role) | pred-MOS (D/U) | Spk cos | Latency full / 1st | RTF | WER | Clone |
|---|---|---|---|---|---|---|---|
| **EN** | **Kokoro-82M** (default) | 4.63 / 4.51 | — | 0.29 s / 0.28 s | **0.095** | **1.3 %** | ✗ |
| EN | Chatterbox-ML 4bit (clone) | 4.63 / 4.28 | 0.725 | 4.06 s / 1.01 s | 1.07 | 11.2 % | ✓ |
| **HI** | **Kokoro-82M** (default) | 4.64 / 4.26 | — | 0.48 s / 0.47 s | **0.092** | 11.1 %* | ✗ |
| HI | Chatterbox-ML 4bit (clone) | 4.57 / 3.73 | **0.870** | 4.98 s / 1.52 s | 1.07 | 22.8 % | ✓ |
| AR | MMS-TTS +digit-verbalize (floor) | 4.44 / 3.37 | — | 0.73 s / 0.73 s | 0.126 | 22.1 % | ✗ |
| AR | Chatterbox-ML 4bit (clone) | 4.58 / 3.15 | 0.743 | 5.63 s / 1.81 s | 1.06 | **13.4 %** | ✓ |
| **AR** | **Habibi-TTS MSA** (specialist) | 4.44 / 2.93 | **0.779** | 36.4 s / 36.4 s | 5.0 | **9.4 %** | ✓ |

\* Hindi WER is ASR-bound, not TTS-bound — see failure modes. D/U = Distill-MOS / UTMOS,
English-trained proxies (not a substitute for the human panel; for AR/HI read against the
real-speech anchor, not the absolute value).

**Predicted MOS vs. real-speech anchors** (same predictors, this language's reference speaker) —
absolute cross-language MOS is invalid, so read TTS against its anchor:

| Lang | real anchor D / U | best TTS here D / U |
|---|---|---|
| EN | 4.55 / 4.31 | Kokoro 4.63 / 4.51 (≈/above real — clean TTS > expressive audiobook on these predictors) |
| AR | 4.39 / **3.02** | Habibi 4.44 / 2.93, Chatterbox 4.58 / 3.15 (**at real-speech level** — the low UTMOS is the predictor's Arabic bias, not the audio) |
| HI | 4.59 / 3.68 | Kokoro 4.64 / 4.26 (≈/above real) |

The Arabic anchor (UTMOS 3.02 on genuine human speech) is the headline: every Arabic TTS here
lands at or above it, so the low absolute UTMOS numbers say more about the English-trained
predictor than about naturalness — the human panel is the arbiter.

**Speaker-cosine note.** The cosine is from VoxCeleb-trained ECAPA-TDNN; a generic "0.75"
is embedding-specific. For *this* embedding, genuine same-speaker pairs top out ~0.72–0.80
and different speakers sit near 0. Measured against the per-language real-vs-real ceiling C
and real-vs-other floor F (`run_sim.py --ref2/--other`):

| Lang | clone cosine S | ceiling C | floor F | normalized (S−F)/(C−F) |
|---|---|---|---|---|
| EN (Chatterbox) | 0.725 (min 0.60) | 0.819 | 0.036 | 0.88 |
| AR (Chatterbox) | 0.743 (min 0.67) | 0.851 | 0.064 | 0.86 |
| AR (Habibi MSA) | 0.779 (min —) | 0.851 | 0.064 | 0.91 |
| HI (Chatterbox) | 0.870 (min 0.80) | 0.905 | 0.025 | 0.96 |

So all clones are **clearly the same speaker** — Hindi and Arabic-Habibi outright clear 0.75;
English and Arabic-Chatterbox are ~86–88 % of the way to their own real-vs-real ceiling and
nowhere near the different-speaker floor. Habibi is the strongest Arabic clone (0.91 of ceiling).

## The call, per language, and why

**English → Kokoro-82M (default), Chatterbox for cloning.** Kokoro passes *every* automated
target and is 5–10× under the latency/RTF bars on a fanless laptop; nothing else came close
on speed-for-quality. Its only miss is a minor date/time formatting slip (`en_03`). Kokoro
can't clone (fixed voice bank), so when a specific voice is required, Chatterbox-ML clones it
with strong identity (cos 0.725, 0.88 of ceiling) at the cost of real-time (RTF 1.07) and some
intelligibility (WER 11.2 %).

**Hindi → Kokoro-82M (default), Chatterbox for cloning.** Same story: Kokoro is fast (RTF
0.092, 0.48 s) and natural. Its 11.1 % round-trip WER looks like a miss but is dominated by
ASR — see below. Chatterbox-ML gives the **best cloning result of the three languages**
(cosine 0.87, 0.96 of ceiling) — the clean studio reference helps — though Hindi cloned WER
(22.8 %) shows the quant model struggles with Hindi intelligibility on hard rows.

**Arabic → Habibi-TTS MSA for quality (offline), Chatterbox for practical cloning; MMS only
as a floor.** Habibi MSA (F5-TTS fine-tune, purpose-built for Arabic, Apache-2.0) is the
**accuracy + fidelity winner**: round-trip WER **9.4 %** (the only Arabic system under the 10 %
bar), best speaker cosine (0.779, 0.91 of ceiling), and predicted MOS at real-speech level —
but RTF ~5 and 36 s clips make it **strictly offline/batch** on this hardware. Chatterbox-ML is
the **pragmatic real-time-ish pick**: cloning, 13.4 % WER, one model shared with EN/HI, first
chunk ~1.8 s. The MMS floor is intelligible after the digit fix (30.2 %→22.1 % WER) but robotic
and can't clone. Net: Arabic is the hardest language — you can have *natural + accurate +
cloning* (Habibi) **or** *fast + cloning* (Chatterbox), not both at once on an 8 GB CPU/MPS box.

## Metrics & methodology (open-source throughout)

- **Round-trip WER** — faster-whisper large-v3 (CT2 int8, CPU), **language forced** so a
  mis-pronounced clip is penalized, not silently re-detected. Both sides pass the same
  per-language normalizer (`eval/normalizers.py`: English Whisper normalizer; Arabic
  dediacritization + alef/hamza/taa-marbuta folding; Devanagari-safe Hindi + num2words digit
  verbalization). WER without stated normalization is meaningless; ours is in-repo.
- **Predicted MOS** — Distill-MOS (primary) + UTMOS22 (literature-comparable). English-trained
  proxies; for AR/HI we report against a same-language real-speech anchor. **The graded MOS is
  the human panel** (`eval/listening_test/`, blinded P.808-style ACR + cloning A/B, hidden real
  anchor, ≥3 native/fluent listeners/language) — the one step that cannot be automated.
- **Speaker similarity** — SpeechBrain ECAPA cosine with ceiling/floor calibration (above).
- **Latency / RTF** — warm, in-pipeline `TimingRecorder`, first row re-run to drop compilation.

## Bonus metrics (added — the ones that changed how I read the results)

Full table: `results/extra_metrics.md` (regenerate with `eval/extra_metrics.py`). Computed
from the audio + existing WER, so cheap to re-run.

| System | Hard-token WER | Expressiveness (emo/neutral F0) | Mean F0 std |
|---|---|---|---|
| en/kokoro | **0.049** | 1.17 | 38.6 |
| en/chatterbox | 0.244 | 1.43 | 37.9 |
| hi/kokoro | 0.155 | 1.14 | 40.9 |
| hi/chatterbox | 0.392 | 1.05 | 30.4 |
| ar/mms_vd | 0.396 | 0.94 | 34.0 |
| ar/chatterbox | 0.264 | 0.62 | 27.9 |
| ar/habibi | 0.245 | **1.31** | 42.5 |

- **Hard-token WER** (WER on the names/numbers/currency/digits/acronym rows only) is the
  metric I trust most for a real voice agent, and it is **3–5× the corpus WER** for every
  system — a model can ace overall intelligibility and still misread an account number or a
  name. Kokoro English is the only system that stays comfortably safe here (4.9 %); everything
  else needs the text-normalization frontend from the roadmap before it touches live numbers.
- **Expressiveness** (pitch spread on the emotion row ÷ the neutral row) separates models that
  MOS and WER rate as tied: the MMS floor barely modulates (0.72–0.94, i.e. it reads an
  exclamation like a ledger), while Habibi (1.31) and Chatterbox-EN (1.43) genuinely lift pitch
  for affect. Chatterbox on Arabic actually *flattens* on the emotion row (0.62) — a concrete
  weakness the human panel should confirm.
- **Audio hygiene** (clipping / edge-silence): clean across the board (max clipped ≈ 0), so no
  system is winning or losing on artefacts — the differences above are real, not glitches.

Other metrics worth adding with more time (noted, not run): GPU/energy cost per second of
audio, long-form prosody drift on paragraph-length input, and code-switch pronunciation
accuracy scored by a bilingual rater.

## Where it breaks (honest failure modes)

- **Hindi WER is ASR-confounded.** faster-whisper large-v3 gives Hindi 11.1 %; the Hindi-tuned
  `vasista22/whisper-hindi-medium` gives 12.3 % **but on different rows** — the fine-tuned model
  nails the number/date rows (`hi_03`, `hi_07` → 0.00) while reading digit strings by another
  convention (`hi_11` → 0.58). Vanilla Whisper's Hindi floor on *clean human speech* is ~19 %,
  so much of the 11 % is recognizer error, not TTS error. Taking the min over the two ASRs
  per row bounds Kokoro-hi's true intelligibility error well under 10 %.
- **Arabic digits.** MMS's char-level VITS **silently drops Arabic-numeral digits** (OOV chars
  discarded) — `ar_04`/`ar_11` were deletion-dominated until a `num2words` frontend was added
  (WER 30.2 %→22.1 %). Found from the transcripts, fixed, both runs kept for the before/after.
- **English date/time.** `en_03` ("March 23rd at 3:45 PM") is Kokoro's only WER miss — ordinal
  + clock-time formatting.
- **Cloning is not real-time on 8 GB.** Chatterbox RTF ~1.07 and 4–5 s full clips fail the RTF
  ≤ 0.5 / < 2 s targets that fixed-voice Kokoro meets. First-chunk (1–1.8 s) is usable but the
  sub-500 ms streaming target needs streaming/chunking or GPU-class hardware.
- **Predicted-MOS disagreement on Arabic.** Distill-MOS 4.45 vs UTMOS 3.27 on the same clips —
  English-trained predictors are unreliable in absolute terms for Arabic; hence the human panel.

## What's still missing & how I'd improve it

Full version with sources: `notes/LIMITATIONS_AND_DISCLOSURE.md`. Headlines: a proper
per-language text-normalization frontend (numbers/dates/currency/IDs) — the single
highest-leverage, model-agnostic fix, already proven on Arabic digits; Hinglish handling for
Hindi (transliterate inline Latin → Devanagari); clause-level streaming to hit sub-500 ms
first-audio; recalibrated / more-human MOS for Arabic & Hindi; and per-language routing to the
best *permissively-licensed* specialist so shipped audio stays commercial-safe.

## Disclosure

Core generation and the entire eval stack are open-source; no closed APIs anywhere. The
recommended router (Kokoro + Chatterbox-4bit-MLX + Habibi-MSA) is commercial-safe and
unwatermarked on the path used here (verified — see `notes/LIMITATIONS_AND_DISCLOSURE.md`).
Reference voices are CC BY 4.0, recorded for speech research, attributed in
`references/LICENSES.md`. Claude Code (AI assistant) was used for research, code, and
orchestration, as the brief invites; all benchmark numbers come from real runs on the M2.
