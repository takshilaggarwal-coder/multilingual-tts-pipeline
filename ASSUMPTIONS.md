# Assumptions & Calls Made

The brief says: if something is ambiguous, make a call and state it. These are the calls.

1. **Hardware.** Everything runs on a MacBook Pro (Apple M2, 8 GB unified RAM, macOS, no CUDA GPU).
   All latency/RTF numbers are reported on this hardware, named per metric. 8 GB RAM is a hard
   constraint that shaped model selection — the heaviest open models (e.g. CosyVoice2 at full
   precision, large diffusion TTS) are not benchmarkable here; where a model is widely considered
   SOTA but couldn't run on this machine, that is stated rather than guessed at.
2. **Track A** (code repo), which also carries Track B's audio clips and a written comparison.
3. **Reference voice for cloning.** Openly licensed speech samples (per the brief's explicit
   allowance), one reference speaker per language. Provenance and licenses in
   `references/LICENSES.md`. No real person cloned without consent.
4. **MOS.** Human MOS comes from a small real listener panel (candidate + recruited listeners)
   using the blinded, randomized listening kit in `eval/listening_test/`. An open predicted-MOS
   model is also reported, clearly labeled as an automated proxy — it is NOT a substitute for
   the human panel and its validity on Arabic/Hindi is caveated.
5. **Latency definition.** "Latency to first audio" is measured warm (model already loaded),
   from synthesis-call start to first audio chunk (streaming) or to complete waveform (batch),
   on the ~10-word `*_01` sentence per language. Cold-start (model load) is reported separately.
6. **WER definition.** Round-trip WER uses per-language ASR on generated audio vs. input text,
   after language-appropriate text normalization (case/punct folding for English; diacritic and
   orthography normalization for Arabic; Devanagari normalization and digit folding for Hindi).
   Normalization code is in the repo — WER without stated normalization is meaningless.
7. **Disclosure.** Core speech generation: open-source models only. Evaluation stack: also
   open-source. AI coding assistant (Claude Code) was used throughout for research, code, and
   automation — as the brief invites — with all runs executed for real on the hardware above.
