# Assumptions & Calls Made

The brief says: if something is ambiguous, make a call and state it. These are the calls.

1. **Hardware.** Everything runs on a MacBook Pro (Apple M2, 8 GB unified RAM, macOS, no CUDA GPU).
   All latency/RTF numbers are reported on this hardware, named per metric. 8 GB RAM is a hard
   constraint that shaped model selection — the heaviest open models (e.g. CosyVoice2 at full
   precision, large diffusion TTS) are not benchmarkable here; where a model is widely considered
   SOTA but couldn't run on this machine, that is stated rather than guessed at.
2. **Track A** (code repo), which also carries Track B's audio clips and a written comparison.
3. **Reference voice for cloning.** English and Hindi clone **my own voice** (recorded for
   this task, 2026-07-18 — consent is mine to give). Arabic uses an openly licensed sample
   per the brief's explicit allowance: the Arabic Speech Corpus's professional MSA narrator
   (CC BY 4.0, recorded for speech-synthesis research). Provenance, verification URLs, and
   attribution in `references/LICENSES.md`. No real person cloned without consent. An earlier
   iteration used CC BY 4.0 corpus voices for EN/HI too (LibriTTS-R spk 84, SYSPIN); those
   numbers are preserved in git history.
4. **MOS.** Human MOS comes from a small real listener panel (candidate + recruited listeners,
   native/fluent per language) using the blinded, randomized listening kit in
   `eval/listening_test/` — the kit is generated and ready to send; panel collection is the
   one step that cannot be automated and its status is reported honestly in the results. An
   open predicted-MOS model is also reported, clearly labeled as an automated proxy — it is
   NOT a substitute for the human panel and its validity on Arabic/Hindi is caveated (scores
   are anchored against real speech from the same language's reference speaker).
5. **Latency definition.** "Latency to first audio" is measured warm (model already loaded),
   from synthesis-call start to first audio chunk (streaming) or to complete waveform (batch),
   on the ~10-word `*_01` sentence per language. Cold-start (model load) is reported separately.
6. **WER definition.** Round-trip WER uses per-language ASR on generated audio vs. input text,
   after language-appropriate text normalization (case/punct folding for English; diacritic and
   orthography normalization for Arabic; Devanagari normalization and digit folding for Hindi).
   Normalization code is in the repo — WER without stated normalization is meaningless.
7. **Disclosure.** Core speech generation: open-source models only. Evaluation stack: also
   open-source. No closed APIs or closed tools were used anywhere — generation or evaluation.
   All runs were executed for real on the hardware above.
