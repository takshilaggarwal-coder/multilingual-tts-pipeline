# Multilingual Voice AI Pipeline — submission write-up

**Track A (code).** Repo runs; every number below was produced by real runs on the named
hardware. Reproduction: `README.md`. Honest build log incl. dead ends: `notes/WORKLOG.md`.
Full licensing + limitations: `notes/LIMITATIONS_AND_DISCLOSURE.md`.

## Recommended setup (one paragraph)

Use a **per-language router of open-source models, MLX-first for Apple Silicon**, split by
whether the request needs *speed* or *naturalness*. **Speed/intelligibility path: English and
Hindi → Kokoro-82M** (Apache-2.0) — RTF ~0.09, ~0.3–0.5 s latency, 1.3 % English WER — but my
3-listener blinded panel rated it only ~2.8/5, audibly synthetic prosody despite predicted-MOS
of 4.6. **Naturalness/cloning path → Chatterbox-Multilingual (4-bit MLX)** — one 607 MB
Apache/MIT checkpoint, unwatermarked on this path, cloning **my own voice**: the Hindi clone
was the panel's best TTS at **human MOS 4.47** (the only system to pass the 4.0 bar; A/B
"same speaker", cosine 0.78) at the honest cost of RTF ~1.1 and 4–5 s clips. English cloning
worked only with a studio reference (0.73) and failed on my phone-mic one (0.48, humans said
"different") — reference capture quality turns out to be a product feature. **Arabic** has no
fast+natural option: **Habibi-TTS MSA** (Apache-2.0) wins accuracy (WER 9.4 %, the only Arabic
under the 10 % bar; cosine 0.78) but is offline-only at RTF ~5; Chatterbox is the practical
middle (WER 13.4 %); the MMS floor is fast but robotic. Evaluation is fully open-source
(faster-whisper + Hindi-tuned cross-check, Distill-MOS/UTMOS as labeled proxies anchored to
real speech, ECAPA cosine with ceiling/floor calibration, blinded P.808-style human panel).
No closed API anywhere. **Arabic** is the hard case: its only *fast* option is the
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

## Human MOS — the graded naturalness metric

3-listener blinded panel (me + two recruited raters), P.808-style ACR over randomized clips
with a hidden real-speech anchor; 5 clips per system per rater. **The anchor scored highest
(4.67), validating the panel.** Arabic naturalness was **not** rated — no Arabic-fluent
listener was available; that is disclosed rather than papered over (Arabic gets predicted-MOS
+ the A/B identity judgments below).

| System | Human MOS | 95 % CI | Predicted (D/U) | Verdict vs 4.0 target |
|---|---|---|---|---|
| hi/chatterbox (clone of my voice) | **4.47** | ±0.46 | 4.01 / 3.17 | **pass** |
| en/chatterbox (clone of my voice) | 3.67 | ±0.61 | 3.96 / 3.55 | miss |
| en/kokoro | 2.87 | ±0.69 | 4.63 / 4.51 | miss |
| hi/kokoro | 2.80 | ±0.60 | 4.64 / 4.26 | miss |
| *real-speech anchor* | *4.67* | ±0.54 | — | *(sanity check)* |

Two findings worth the whole exercise:

1. **The human panel inverts the predicted-MOS ranking.** Kokoro — which the automated
   predictors scored at 4.6 — lands at 2.8–2.9 with human listeners, while the Chatterbox
   clones they score lower win with humans. The predictors reward clean, artifact-free audio;
   humans hear Kokoro's flat prosody and mark it synthetic. This is precisely why the brief's
   "the listening test matters" is right, and why every predicted-MOS number above is labeled
   a proxy.
2. **Cloning A/B (same-speaker judgments, 3 raters):** hi clone **2× same / 1 unsure** —
   confirms the 0.78 cosine; en clone **2× different / 1 unsure** — humans independently
   confirm the phone-mic reference finding flagged by the 0.48 cosine. Arabic (n=2, identity
   judgment needs no fluency): chatterbox 1 same / 1 unsure; habibi 1 different / 1 unsure —
   too small to call, noted as such.

Panel caveats, stated plainly: 3 listeners, 15 ratings/system → wide CIs (±0.5–0.7); listeners
are Indian-English/Hindi speakers, which may penalize Kokoro's American-accented English less
or more than a US panel would; one rater's unlabeled sheet was interpreted as Hindi by
elimination. Raw sheets preserved in `results/human_mos.json`.

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

**English → route by what matters.** For **latency + intelligibility** (IVR prompts, agent
responses read fast), **Kokoro-82M**: 5–10× under the latency/RTF bars, WER 1.3 %, hard-token
WER 4.9 % — but the human panel rates it only 2.87, audibly synthetic prosody. For
**naturalness**, **Chatterbox** (human MOS 3.67, the best-rated English TTS here) at the cost
of RTF ~1.1 and WER. Honest bottom line: **no English system passed the 4.0 human-MOS bar on
this panel**, and English cloning is only trustworthy with a quality reference (0.73 studio
vs 0.48 phone-mic, confirmed "different" by human A/B).

**Hindi → Chatterbox clone for quality (the benchmark's best result), Kokoro for speed.**
The clone of my own phone-mic voice scored **human MOS 4.47 — the only system to pass the
4.0 bar** — with A/B "same speaker" and cosine 0.778. Kokoro-hi stays the latency pick (RTF
0.092, 0.48 s) and its 11.1 % WER is ASR-dominated (see below), but human listeners rate its
espeak-G2P prosody 2.80 — fast, intelligible, and audibly robotic. Where the product can
afford ~5 s generation or streaming chunks, Hindi should ship the cloning path.

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
- **Predicted MOS penalizes real speech and misses prosody.** My genuine phone-mic recordings
  score UTMOS 2.4–2.6 — *below* the clones of me — and the human panel then inverted the
  predicted ranking entirely (Kokoro predicted 4.6, humans 2.8). The predictors measure channel
  cleanliness, not humanness; absolute numbers are not comparable across languages, recording
  conditions, or model families. Hence anchors + the human panel as the graded metric.
- **Human-MOS coverage gap.** No Arabic-fluent listener was available before the deadline, so
  Arabic naturalness carries predicted-MOS + a 2-rater A/B only. With more time: recruit 2–3
  native Arabic raters (the kit is ready to send as-is).

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
