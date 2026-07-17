# Reference Speaker Clips — Sources & Licenses

Reference voice clips used as the cloning target for the multilingual TTS benchmark.
One consistent speaker per language.

- **English and Hindi: my own voice** (recorded 2026-07-18 for this task — see below).
- **Arabic: the Arabic Speech Corpus's professional MSA narrator** (CC BY 4.0), per the
  brief's allowance for openly licensed samples. License page fetched and verified
  2026-07-15.

`<lang>_ref_main.wav` is the primary reference take; `<lang>_ref_02.wav` is a second,
different take of the same speaker used only to calibrate the real-vs-real speaker-
similarity ceiling. `<lang>_ref_10s.wav` is a ~10 s cut of the main take (cut at a
silence boundary) used as the actual cloning conditioning input, with its transcript in
`transcripts.json` produced by transcribing the exact cut.

---

## English — own voice

- **Speaker:** me (the candidate), recorded with my own consent for this submission.
- **Recording:** phone/laptop mic, quiet room, 2026-07-18; two takes (36 s + 19 s) of a
  short first-person script. Converted m4a → 24 kHz mono 16-bit WAV with macOS
  `afconvert`; peak-normalized to −1 dBFS.
- **Files:** `en/en_ref_main.wav` (36.0 s, take 1), `en/en_ref_02.wav` (19.3 s, take 2),
  `en/en_ref_10s.wav` (10.5 s cut of take 1 — the cloning reference).
- **License:** my own recording; no third-party rights involved.

## Arabic (MSA) — Arabic Speech Corpus (Nawar Halabi)

- **Corpus:** Arabic Speech Corpus, Nawar Halabi (PhD, University of Southampton, 2016).
  Single professional speaker, studio-recorded Modern Standard Arabic, built specifically
  for speech synthesis.
- **License:** CC BY 4.0.
- **License verified at:**
  - https://en.arabicspeechcorpus.com/ — "Arabic Speech Corpus by Nawar Halabi is licensed
    under a Creative Commons Attribution 4.0 International License."
  - https://huggingface.co/datasets/halabi2016/arabic_speech_corpus (card metadata `license: cc-by-4.0`)
- **Access path:** Hugging Face mirror `tunis-ai/arabic_speech_corpus` (data-only parquet
  repack of the same corpus, same CC BY 4.0 card), split `test`, audio fetched per-utterance
  via the HF datasets-server rows API (no gating).
- **Speaker:** the corpus's single professional male MSA narrator (unnamed in the corpus;
  recorded with consent for speech-synthesis research).
- **Files** (original 48 kHz mono, saved unmodified as 16-bit PCM WAV):

| file | corpus file | duration |
|---|---|---|
| `ar/ar_ref_01.wav` (= `ar_ref_main.wav`) | `ARA NORM 0031.wav` (test) | 28.7 s |
| `ar/ar_ref_02.wav` | `ARA NORM 0016.wav` (test) | 22.4 s |
| `ar/ar_ref_03.wav` | `ARA NORM 0050.wav` (test) | 22.6 s |

- `ar/ar_ref_10s.wav` = 10.6 s silence-boundary cut of `ar_ref_main.wav` (cloning reference).
- **Required attribution:** "Arabic Speech Corpus by Nawar Halabi
  (www.arabicspeechcorpus.com), licensed under CC BY 4.0
  (https://creativecommons.org/licenses/by/4.0/)."

## Hindi — own voice

- **Speaker:** me (the candidate), recorded with my own consent for this submission.
- **Recording:** phone/laptop mic, quiet room, 2026-07-18; two takes (43 s + 23 s) of a
  short first-person Hindi script. Converted m4a → 24 kHz mono 16-bit WAV with macOS
  `afconvert`; peak-normalized to −1 dBFS.
- **Files:** `hi/hi_ref_main.wav` (42.8 s, take 1), `hi/hi_ref_02.wav` (22.8 s, take 2),
  `hi/hi_ref_10s.wav` (9.7 s cut of take 1 — the cloning reference).
- **License:** my own recording; no third-party rights involved.

---

### Earlier corpus references (superseded 2026-07-18)

The first benchmark iteration used openly licensed corpus speakers for English and Hindi
as well: LibriTTS-R speaker 84 (CC BY 4.0, verified at https://www.openslr.org/141/) and
the SYSPIN Hindi female voice artist (IISc Bengaluru SPIRE Lab, CC BY 4.0, verified at
https://vaani.iisc.ac.in/dataset/syspindataset). Those clips were replaced by my own
recordings for the final run; the git history preserves the earlier files and the
benchmark numbers measured against them.

### Rejected candidates (for the record)

- `SPRINGLab/IndicTTS-Hindi` (IIT Madras Indic TTS): restrictive custom license requiring
  a signed agreement — not permissive. Skipped.
- `ai4bharat/indicvoices_r`, `ai4bharat/Rasa`: gated on Hugging Face (login required). Skipped.
- Mozilla Common Voice: gated on HF. Not needed.
