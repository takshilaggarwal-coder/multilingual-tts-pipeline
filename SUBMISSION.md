# Multilingual voice pipeline — write-up

Track A. The repo runs, and every number in here came out of a real run on my laptop.
Setup and reproduction steps are in the README. The day-by-day log, including the things
that broke, is in `notes/WORKLOG.md`. Licensing details in `notes/LIMITATIONS_AND_DISCLOSURE.md`.

## What I'd ship

A per-language router, split by what the request actually needs: speed or naturalness.
No single open model wins all three languages, and the hint in the brief was right —
I checked anyway.

For **fast responses** in English and Hindi I'd use **Kokoro-82M** (Apache-2.0, runs on MLX).
It's absurdly fast on a fanless 8GB MacBook: RTF around 0.09, first audio in 0.3–0.5s,
1.3% round-trip WER in English. The catch, which I only learned from the listening test:
people hear it as robotic. My 3-listener panel scored it ~2.8/5 while the automated MOS
predictors gave it 4.6. More on that below, because it ended up being the most useful
result in the whole exercise.

For **naturalness and voice cloning** I'd use **Chatterbox-Multilingual, 4-bit MLX build**.
One 607MB checkpoint covers all three languages with zero-shot cloning. The Hindi clone of
my own voice was the best-rated system in the listening test (4.47/5 — the only one to
clear the 4.0 bar) and raters called it the same speaker. It costs real time though:
RTF ~1.1, so a sentence takes 4–5 seconds.

**Arabic** is genuinely the hard one. **Habibi-TTS** (the Apache-licensed MSA checkpoint)
was the accuracy winner — 9.4% WER, the only Arabic system under the 10% target, and the
strongest Arabic clone — but at RTF ~5 it's batch-only on this hardware. Chatterbox is the
practical middle at 13.4% WER. MMS-TTS is fast and tiny but sounds like 2015.

Generation and evaluation are open-source end to end. No closed APIs anywhere.

## Hardware and how I measured

MacBook Pro, Apple M2, 8GB unified RAM, no CUDA. The 8GB is what shaped the design:
PyTorch builds of the cloning models swap-death around 10GB, so everything that could run
on MLX runs on MLX, one model in memory at a time.

Latency and RTF are measured warm (model already loaded), from the synthesis call to first
audio chunk or full waveform, on the ~10-word `*_01` sentence per language. The first
utterance gets re-run so nobody pays one-time compilation cost in the numbers. Cold-load
times are in each `timings.json`.

## Results against the section-3 targets

Regenerate with `eval/aggregate.py`; full table in `results/results.md`.
Targets: MOS ≥ 4.0, speaker cosine ≥ 0.75, latency < 2s full / < 0.5s first chunk,
RTF ≤ 0.5, WER ≤ 10%.

| Lang | Model | Human MOS | Pred. MOS (D/U) | Spk cos | Latency full/1st | RTF | WER |
|---|---|---|---|---|---|---|---|
| EN | Kokoro-82M | 2.87 | 4.63 / 4.51 | — | 0.29s / 0.28s | **0.095** | **1.3%** |
| EN | Chatterbox (clone of me) | 3.67 | 3.96 / 3.55 | 0.48 | 3.87s / 0.95s | 1.13 | 17.1% |
| HI | Kokoro-82M | 2.80 | 4.64 / 4.26 | — | 0.48s / 0.47s | **0.092** | 11.1%* |
| HI | Chatterbox (clone of me) | **4.47** | 4.01 / 3.17 | **0.778** | 5.17s / 1.48s | 1.10 | 16.1% |
| AR | MMS-TTS + digit fix | n/r | 4.44 / 3.37 | — | 0.73s / 0.73s | 0.126 | 22.2% |
| AR | Chatterbox (corpus narrator) | n/r | 4.58 / 3.15 | 0.743 | 5.63s / 1.81s | 1.06 | 13.4% |
| AR | Habibi-TTS MSA | n/r | 4.44 / 2.93 | **0.779** | 36.4s | 5.0 | **9.4%** |

\* mostly ASR error, not TTS error — explained under failure modes.
n/r = not rated by the human panel (no Arabic-fluent listener in time; see below).

Latency and RTF: Kokoro and MMS clear the bars easily; the cloning models don't, on this
hardware. WER: only Kokoro-EN and Habibi-AR get under 10%.

## The listening test, and why it mattered

Three listeners (me plus two people I recruited), blinded and randomized clips, a hidden
real-speech recording mixed in as a sanity anchor, five clips per system per rater.
Sheets and scoring code in `eval/listening_test/`; raw results in `results/human_mos.json`.

The anchor came back highest (4.67), so the panel wasn't rating noise. And then it flatly
contradicted the automated predictors:

| System | Humans | Predictors said |
|---|---|---|
| HI Chatterbox (my voice) | **4.47 ± 0.46** | 4.01 / 3.17 |
| EN Chatterbox (my voice) | 3.67 ± 0.61 | 3.96 / 3.55 |
| EN Kokoro | 2.87 ± 0.69 | 4.63 / 4.51 |
| HI Kokoro | 2.80 ± 0.60 | 4.64 / 4.26 |

Kokoro, which UTMOS and Distill-MOS loved, got hammered by actual ears. The predictors are
trained to detect artifacts and noise; Kokoro produces spotless audio with flat prosody, and
that flatness is exactly what humans punish. I went in expecting the panel to roughly confirm
the predicted numbers and it inverted them instead. Every predicted-MOS figure in this doc
should be read with that in mind — I kept them because they're still useful for comparing
similar systems, but the panel is the metric that counts.

The same-speaker A/B judgments lined up with the embedding math, which was reassuring:
the Hindi clone got 2× "same" / 1 "unsure" (cosine 0.78), and the English clone got
2× "different" (cosine 0.48 — see the next section for why English cloning went wrong).

Caveats I want on the record: 3 listeners is a small panel and the CIs are wide (±0.5–0.7).
Nobody on the panel speaks Arabic, so Arabic has no human naturalness score — I'd rather
say that than invent one. The A/B identity judgments don't need language fluency, so Arabic
did get those (n=2, split verdicts, too small to conclude much). One rater returned an
unlabeled sheet that I matched to Hindi by elimination.

## Cloning: the reference recording is half the battle

I recorded my own voice for the English and Hindi references (phone mic, quiet room;
Arabic uses the professional narrator from the CC-BY-4.0 Arabic Speech Corpus, since I
don't speak Arabic). Speaker similarity is ECAPA cosine, and since a bare "0.75 target"
means nothing without context for a given embedding, I calibrated per language: ceiling =
two different real takes of the same speaker, floor = two different speakers.

| Clone | cosine | ceiling | floor | position |
|---|---|---|---|---|
| HI Chatterbox (me) | 0.778 | 0.923 | 0.155 | 0.81 |
| AR Habibi (narrator) | 0.779 | 0.851 | 0.064 | 0.91 |
| AR Chatterbox (narrator) | 0.743 | 0.851 | 0.064 | 0.86 |
| EN Chatterbox (me) | 0.480 | 0.833 | 0.045 | 0.55 |

Three of the four are clearly the same speaker. English is not, and I dug into why: earlier
in the week I'd run the identical pipeline against a studio-recorded LibriTTS reference and
got 0.725. Swapping in my phone recording dropped it to 0.48, and a second take of my voice
scored the same 0.47, so it isn't the specific cut — it's the recording channel (possibly
plus accent coverage). Hindi degraded much less (0.87 studio → 0.78 phone). For a voice
product this is a design input, not a footnote: customers will hand you phone-quality
references, so the intake flow needs level/noise checks or reference enhancement before
conditioning, and English apparently needs it most.

Predicted MOS told the same story from another angle. My genuine recorded voice scores
UTMOS 2.4–2.6 — lower than the clones of me. Real Arabic studio speech scores 3.02. So I
report a real-speech anchor per language (`results/mos_anchors.json`) and read every
predicted score against it rather than as an absolute.

## Per-language calls

**English.** Kokoro for anything latency-sensitive; it's also the only system here I'd
trust with numbers and names untouched (hard-token WER 4.9%). When naturalness matters
more, Chatterbox — humans rated it a full point above Kokoro. Neither cleared the 4.0
human-MOS bar on my panel, which I'm reporting as-is. English cloning only works with a
decent reference recording.

**Hindi.** The surprise winner. The Chatterbox clone of my voice was the panel's favorite
system overall and passed both the MOS and similarity targets; if the product can absorb
~5s generation (or stream in chunks), that's the ship. Kokoro-hi stays the fast path.
Its 11.1% WER is misleading: Whisper's own error floor on clean Hindi speech is ~19%, and
a Hindi-finetuned Whisper cross-check agreed with large-v3 on different rows (each nails
what the other flubs), so the TTS-attributable error is roughly 6%. The one failure both
ASRs agree on is Hinglish — espeak's G2P mangles Latin-script words inside Devanagari.

**Arabic.** Take your pick of tradeoffs. Habibi MSA when quality and accuracy matter and
you can batch (9.4% WER, best clone, RTF 5). Chatterbox when you need it interactive-ish
(13.4% WER, first chunk 1.8s). MMS only as a tiny fallback. What I couldn't find in open
source: fast + natural + Arabic in one model with a usable license.

## Extra metrics I added

The section-3 six left gaps, so I added three (code: `eval/extra_metrics.py`,
table: `results/extra_metrics.md`):

**Hard-token WER** — WER computed only on the rows with names, numbers, currency, digit
strings and acronyms. It runs 3–5× the corpus WER for every system (Kokoro-EN: 1.3% corpus
but 4.9% hard-token; my English clone: 58%). Corpus WER flatters everyone; this is the
number a voice agent actually lives or dies by, and it says none of these models should
read an account number without a text-normalization frontend.

**Expressiveness** — pitch spread on the emotion-row sentence divided by the neutral row.
The MMS floor barely moves (0.7–0.9, it reads excitement like a grocery list); Habibi
genuinely modulates (1.31). It also flagged that Chatterbox-Arabic flattens on emotional
text (0.62), which I'd want a native listener to verify.

**Audio hygiene** — clipping and edge-silence checks. Everything came back clean, which
means the differences above are real and not artifacts.

Given more time I'd add energy cost per second of audio, prosody drift on paragraph-length
input, and a bilingual rater scoring code-switch pronunciation.

## Where it breaks

- **Arabic digits, silently.** MMS's character tokenizer drops characters it doesn't know,
  so "1249" just never gets spoken. Found it staring at deletion-heavy WER transcripts.
  Fixed with num2words pre-verbalization in the frontend: 30.2% → 22.2% WER. I kept both
  runs in `outputs/` for the before/after.
- **Hinglish.** Kokoro-hi's worst row by far (0.41 WER on both ASRs). Inline English words
  in Devanagari sentences need transliteration before G2P; that's a frontend fix, not a
  model fix.
- **Phone-mic cloning references** (the 0.48 story above).
- **Cloning isn't real-time on 8GB.** RTF ~1.1 misses the 0.5 target. First chunk lands at
  0.95–1.5s, so chunked streaming would make it feel fine, but the sub-500ms streaming bar
  needs GPU-class hardware or a smaller model.
- **Predicted MOS can't be trusted alone.** It penalizes real recordings, misses prosody,
  and my panel inverted its rankings. Anchors + humans or it doesn't count.
- **Arabic human MOS is missing.** No fluent listener in time. The kit is built and blinded;
  extending the panel is a ten-minute ask per rater.
- **`en_03`** ("March 23rd at 3:45 PM") — Kokoro's one English WER miss. Dates and clock
  times belong in the normalization frontend too.

## What's missing in open source, and what I'd do next

Ranked by return on effort:

1. **A real text-normalization frontend** for numbers, dates, currency, and IDs across all
   three languages. Model-agnostic, no training, and the digit fix already proved the value.
   This is the single biggest quality lever in the whole stack.
2. **Hinglish handling**: transliterate embedded Latin script to Devanagari (AI4Bharat
   IndicXlit) before G2P.
3. **Reference intake for cloning**: level/noise gate plus enhancement, given the 0.73 → 0.48
   studio-vs-phone gap.
4. **Chunked/clause-level streaming** over the fast models to decouple first-audio latency
   from utterance length.
5. **MOS predictors recalibrated for AR/HI** against small human-rated sets — or accept that
   panels are the metric and make running them cheap.
6. Longer term: the open ecosystem still has no fast, natural, permissively-licensed Arabic
   model, and dialect coverage in usable licenses is nearly zero. That's a fine-tuning
   project (Habibi MSA as the base) more than a search problem.

## Disclosure

Everything generating or scoring audio here is open-source; no closed APIs were used for
anything. The recommended router (Kokoro / Chatterbox-4bit-MLX / Habibi-MSA) is Apache/MIT
licensed and commercially safe. One correction I made along the way: upstream PyTorch
Chatterbox watermarks its output (Resemble PerTh) by default, but the MLX build used here
has no watermarking step — I verified this on-device after initially assuming otherwise,
so the delivered clips are unwatermarked, and a PyTorch-path deployment would need to
disclose watermarking to users. Non-commercial components (MMS weights, NISQA, F5 base)
appear only as benchmark controls, never in the recommended path. Cloning references:
my own voice for English and Hindi; the CC-BY-4.0 Arabic Speech Corpus narrator for Arabic
(attribution in `references/LICENSES.md`). Whisper's transcripts of my reference cuts are
in `references/transcripts.json` if you want to check that it's really me reading.
