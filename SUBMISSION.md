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
607 MB Apache/MIT checkpoint, unwatermarked on this path. Cloning **my own voice** (recorded
on a phone for this task): Hindi cosine 0.78 and Arabic (corpus narrator) 0.74 are clearly
same-speaker, but English lands at 0.48 — a real miss that exposes how sensitive zero-shot
cloning is to reference channel quality (a studio-mic reference had scored 0.73 on the same
pipeline; see failure modes). Cost of cloning is real time: RTF ~1.1, 4–5 s clips. **Arabic** is the hard case: its only *fast* option is the
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
| EN | Chatterbox-ML 4bit (clone, my voice) | 3.96 / 3.55 | 0.480 | 3.87 s / 0.95 s | 1.13 | 17.1 % | ✓ |
| **HI** | **Kokoro-82M** (default) | 4.64 / 4.26 | — | 0.48 s / 0.47 s | **0.092** | 11.1 %* | ✗ |
| HI | Chatterbox-ML 4bit (clone, my voice) | 4.01 / 3.17 | **0.778** | 5.17 s / 1.48 s | 1.10 | 16.1 % | ✓ |
| AR | MMS-TTS +digit-verbalize (floor) | 4.44 / 3.37 | — | 0.73 s / 0.73 s | 0.126 | 22.1 % | ✗ |
| AR | Chatterbox-ML 4bit (clone) | 4.58 / 3.15 | 0.743 | 5.63 s / 1.81 s | 1.06 | **13.4 %** | ✓ |
| **AR** | **Habibi-TTS MSA** (specialist) | 4.44 / 2.93 | **0.779** | 36.4 s / 36.4 s | 5.0 | **9.4 %** | ✓ |

\* Hindi WER is ASR-bound, not TTS-bound — see failure modes. D/U = Distill-MOS / UTMOS,
English-trained proxies (not a substitute for the human panel; for AR/HI read against the
real-speech anchor, not the absolute value). Cloning references: English and Hindi are **my
own voice** (phone-mic, recorded for this task); Arabic is the CC BY 4.0 Arabic Speech Corpus
narrator. An earlier run with studio corpus references for EN/HI is preserved in git history —
the comparison between the two is itself a finding (below).

**Predicted MOS vs. real-speech anchors** (same predictors, this language's reference speaker) —
absolute cross-language MOS is invalid, so read TTS against its anchor:

| Lang | real anchor D / U | best TTS here D / U |
|---|---|---|
| EN | 4.00 / **2.42** (my phone-mic voice) | Kokoro 4.63 / 4.51 — TTS scores far above the real recording |
| AR | 4.39 / **3.02** (studio narrator) | Habibi 4.44 / 2.93, Chatterbox 4.58 / 3.15 (**at real-speech level**) |
| HI | 4.08 / **2.60** (my phone-mic voice) | Kokoro 4.64 / 4.26 — same pattern |

The anchors are the headline: **genuine human speech scores as low as UTMOS 2.4–3.0** on these
English-trained predictors (my own phone-mic recordings score *below* the clones of me). They
reward channel cleanliness as much as naturalness, so absolute predicted-MOS numbers cannot be
compared across languages or recording conditions — the human panel is the arbiter.

**Speaker-cosine note.** The cosine is from VoxCeleb-trained ECAPA-TDNN; a generic "0.75"
is embedding-specific. For *this* embedding, genuine same-speaker pairs top out ~0.72–0.80
and different speakers sit near 0. Measured against the per-language real-vs-real ceiling C
and real-vs-other floor F (`run_sim.py --ref2/--other`):

| Lang | clone cosine S | ceiling C | floor F | normalized (S−F)/(C−F) |
|---|---|---|---|---|
| EN (Chatterbox, my voice) | 0.480 (min 0.35) | 0.833 | 0.045 | **0.55** |
| AR (Chatterbox, corpus narrator) | 0.743 (min 0.67) | 0.851 | 0.064 | 0.86 |
| AR (Habibi MSA, corpus narrator) | 0.779 (min 0.70) | 0.851 | 0.064 | 0.91 |
| HI (Chatterbox, my voice) | 0.778 (min 0.73) | 0.923 | 0.155 | 0.81 |

Hindi and both Arabic clones are **clearly the same speaker** (0.81–0.91 of their ceilings;
Habibi is the strongest Arabic clone). **English is an honest miss at 0.55 normalized** —
"related voice", not "same person". The control that makes this a finding rather than a
mystery: the identical pipeline cloned a studio-mic LibriTTS reference at 0.725 (0.88 of
ceiling), and a second take of my voice scored the same 0.47 — so the gap is the reference
recording channel (and possibly accent coverage), not the model config or the specific cut.
For a voice-agent product this is the number-one operational lesson: **reference capture
quality is a product feature**, and the intake flow needs level/noise checks or reference
enhancement before conditioning.

## The call, per language, and why

**English → Kokoro-82M (default), Chatterbox for cloning — with a caveat.** Kokoro passes
*every* automated target and is 5–10× under the latency/RTF bars on a fanless laptop; nothing
else came close on speed-for-quality. Its only miss is a minor date/time formatting slip
(`en_03`). Kokoro can't clone (fixed voice bank). Chatterbox cloned a studio-quality reference
convincingly (cos 0.725, 0.88 of ceiling) but only reached 0.48 on my phone-mic recording —
so the recommendation stands *conditional on reference quality*, and the intake flow must
enforce it (see failure modes).

**Hindi → Kokoro-82M (default), Chatterbox for cloning.** Same story: Kokoro is fast (RTF
0.092, 0.48 s) and natural. Its 11.1 % round-trip WER looks like a miss but is dominated by
ASR — see below. Chatterbox-ML cloning **held up even on my phone-mic reference** (cosine
0.778, 0.81 of ceiling — clearly me), with cloned-output WER of 16.1 %; against the earlier
studio reference it scored 0.87, the best clone of the whole benchmark.

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
| en/chatterbox (my voice) | 0.585 | 3.12† | 33.8 |
| hi/kokoro | 0.155 | 1.14 | 40.9 |
| hi/chatterbox (my voice) | 0.278 | 0.72 | 17.7 |
| ar/mms_vd | 0.396 | 0.94 | 34.0 |
| ar/chatterbox | 0.264 | 0.62 | 27.9 |
| ar/habibi | 0.245 | **1.31** | 42.5 |

† single-row ratio inflated by a near-monotone neutral row on this clone; treat qualitatively.

- **Hard-token WER** (WER on the names/numbers/currency/digits/acronym rows only) is the
  metric I trust most for a real voice agent, and it is **3–5× the corpus WER** for every
  system — a model can ace overall intelligibility and still misread an account number or a
  name. Kokoro English is the only system that stays comfortably safe here (4.9 %); everything
  else — and especially the phone-mic clones (en 0.59) — needs the text-normalization frontend
  from the roadmap before it touches live numbers.
- **Expressiveness** (pitch spread on the emotion row ÷ the neutral row) separates models that
  MOS and WER rate as tied: the MMS floor barely modulates (0.72–0.94, i.e. it reads an
  exclamation like a ledger), while Habibi (1.31) genuinely lifts pitch for affect. Chatterbox
  on Arabic actually *flattens* on the emotion row (0.62) — a concrete weakness the human panel
  should confirm.
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
- **Zero-shot cloning is only as good as the reference recording.** The same pipeline that
  cloned a studio reference at cosine 0.725 managed only 0.48 on my phone-mic English
  recording (Hindi degraded less: 0.87→0.78). A second take scored identically, isolating the
  cause to the reference channel/accent rather than the cut. Production mitigation: reference
  intake checks (level, noise, duration), reference enhancement before conditioning, or brief
  target-speaker fine-tuning.
- **Cloning is not real-time on 8 GB.** Chatterbox RTF ~1.1 and 4–5 s full clips fail the RTF
  ≤ 0.5 / < 2 s targets that fixed-voice Kokoro meets. First-chunk (0.95–1.5 s) is usable but
  the sub-500 ms streaming target needs streaming/chunking or GPU-class hardware.
- **Predicted MOS penalizes real speech.** My genuine phone-mic recordings score UTMOS 2.4–2.6 —
  *below* the clones of me — and Distill-MOS vs UTMOS disagree by >1 point on Arabic. The
  predictors measure channel cleanliness as much as naturalness; absolute numbers are not
  comparable across languages or recording conditions. Hence anchors + the human panel.

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
Cloning references: my own voice for English and Hindi (recorded for this task); Arabic uses
the CC BY 4.0 Arabic Speech Corpus narrator — provenance and attribution in
`references/LICENSES.md`. All benchmark numbers come from real runs on the M2; the dated
build log, including dead ends, is `notes/WORKLOG.md`.
