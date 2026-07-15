# Reference Speaker Clips — Sources & Licenses

Reference voice clips used as the cloning target for the multilingual TTS benchmark.
One consistent speaker per language. All corpora were recorded for speech-synthesis /
speech research with speaker participation, and all are under permissive open licenses
(CC BY 4.0 throughout — no NC/ND restrictions). License pages were fetched and verified
on 2026-07-15.

`<lang>_ref_main.wav` is a byte-identical copy of `<lang>_ref_01.wav` (the best single clip).

---

## English — LibriTTS-R (speaker 84)

- **Corpus:** LibriTTS-R (sound-quality-restored LibriTTS), Google / Koizumi et al. 2023.
- **License:** CC BY 4.0.
- **License verified at:**
  - https://www.openslr.org/141/ ("License: CC BY 4.0")
  - https://huggingface.co/datasets/mythicinfinity/libritts_r (card metadata `license: cc-by-4.0`)
- **Access path:** Hugging Face `mythicinfinity/libritts_r`, config `dev`, split `dev.clean`,
  audio fetched per-utterance via the HF datasets-server rows API (no gating).
- **Speaker:** corpus speaker ID **84** (female LibriVox volunteer reader; source recordings
  are public-domain LibriVox audiobooks, chapter 121550).
- **Files taken** (original 24 kHz mono, saved unmodified as 16-bit PCM WAV):

| file | corpus utterance ID | duration | transcript |
|---|---|---|---|
| `en/en_ref_01.wav` (= `en_ref_main.wav`) | `84_121550_000184_000000` | 10.0 s | So was I standing; and she said: "If thou In hearing sufferest pain, lift up thy beard And thou shalt feel a greater pain in seeing." |
| `en/en_ref_02.wav` | `84_121550_000292_000000` | 8.6 s | I will too, if not written, at least painted, Thou bear it back within thee, for the reason That cinct with palm the pilgrim's staff is borne." |
| `en/en_ref_03.wav` | `84_121550_000136_000000` | 8.9 s | "Look at me well; in sooth I'm Beatrice! How didst thou deign to come unto the Mountain? Didst thou not know that man is happy here?" |
| `en/en_ref_04.wav` | `84_121550_000093_000000` | 8.7 s | And such as thou shalt find them in his pages, Such were they here; saving that in their plumage john is with me, and differeth from him. |

- **Total:** 36.2 s across 4 clips.
- **Required attribution:** "LibriTTS-R" — Y. Koizumi et al., *LibriTTS-R: A Restored
  Multi-Speaker Text-to-Speech Corpus*, Interspeech 2023. Licensed CC BY 4.0
  (https://creativecommons.org/licenses/by/4.0/). Derived from LibriTTS / LibriSpeech /
  LibriVox public-domain recordings.

---

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
- **Files taken** (original 48 kHz mono, saved unmodified as 16-bit PCM WAV):

| file | corpus file | duration |
|---|---|---|
| `ar/ar_ref_01.wav` (= `ar_ref_main.wav`) | `ARA NORM 0031.wav` (test) | 28.7 s |
| `ar/ar_ref_02.wav` | `ARA NORM 0016.wav` (test) | 22.4 s |
| `ar/ar_ref_03.wav` | `ARA NORM 0050.wav` (test) | 22.6 s |

- **Total:** 73.7 s across 3 clips.
- **Required attribution:** "Arabic Speech Corpus by Nawar Halabi
  (www.arabicspeechcorpus.com), licensed under CC BY 4.0
  (https://creativecommons.org/licenses/by/4.0/)."

---

## Hindi — SYSPIN Hindi Female TTS corpus (IISc Bengaluru, SPIRE Lab)

- **Corpus:** SYSPIN_S1.0 — professional single-speaker TTS corpora in nine Indian
  languages (1 male + 1 female voice artist per language), SPIRE Lab, IISc Bengaluru,
  with Bhashini AI Solutions. Studio recordings made expressly for TTS.
- **License:** CC BY 4.0.
- **License verified at:**
  - https://aikosh.indiaai.gov.in/home/datasets/details/iisc_syspin_s1_0_corpus.html —
    "Attribution 4.0 International (CC BY-4.0)"
  - https://vaani.iisc.ac.in/dataset/syspindataset — "released under a CC-BY-4.0 license"
- **Access path:** Hugging Face ungated mirror `SayantanJoker/SYSPIN_Hindi_Female_TTS`
  (44.1 kHz repack of the official 48 kHz corpus), split `train`, audio fetched
  per-utterance via the HF datasets-server first-rows API.
  *Caveat:* the mirror's own card carries no license metadata; the license above is that
  of the underlying official SYSPIN corpus, verified at the two URLs listed.
- **Speaker:** the SYSPIN **Hindi female** professional voice artist (unnamed in the
  public corpus; recorded under contract for the project).
- **Files taken** (44.1 kHz mono, 16-bit PCM WAV). Processing applied: leading/trailing
  silence trimmed to ~0.2 s and peak-normalized to −1 dBFS (source clips were recorded
  at low level, peak ≈ 0.18):

| file | corpus file_name | duration |
|---|---|---|
| `hi/hi_ref_01.wav` (= `hi_ref_main.wav`) | `sample_65` | 16.6 s |
| `hi/hi_ref_02.wav` | `sample_67` | 15.6 s |
| `hi/hi_ref_03.wav` | `sample_18` | 14.9 s |

- **Total:** 47.1 s across 3 clips.
- **Required attribution:** "SYSPIN_S1.0 Corpus — A TTS Corpus of 900+ hours in nine
  Indian Languages (Abhayjeet et al., 2025), SPIRE Lab, IISc Bengaluru. Licensed CC BY 4.0
  (https://creativecommons.org/licenses/by/4.0/)."

---

### Rejected candidates (for the record)

- `SPRINGLab/IndicTTS-Hindi` (IIT Madras Indic TTS): restrictive custom license requiring
  a signed agreement — not permissive. Skipped.
- `ai4bharat/indicvoices_r`, `ai4bharat/Rasa`: gated on Hugging Face (login required). Skipped.
- Mozilla Common Voice: gated on HF. Not needed — SYSPIN is higher quality and CC BY 4.0.
