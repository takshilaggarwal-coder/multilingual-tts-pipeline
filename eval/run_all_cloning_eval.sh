#!/bin/zsh
# Full eval sweep for the Chatterbox cloning outputs (EN/AR/HI).
# WER (slow, Whisper CPU) + predicted MOS + speaker similarity w/ calibration.
# Run from repo root: zsh eval/run_all_cloning_eval.sh
set -e
cd "$(dirname "$0")"          # -> eval/
P=../envs/eval/bin/python

for L in en ar hi; do
  echo "@@@ $L WER start"
  $P run_wer.py --outdir ../outputs/$L/chatterbox --texts texts/$L.tsv --lang $L 2>/dev/null | tail -1
  echo "@@@ $L MOS start"
  $P run_mos.py --outdir ../outputs/$L/chatterbox --texts texts/$L.tsv 2>/dev/null | tail -1
done

echo "@@@ SIM start"
# same-speaker ceiling = ref_02 vs main; floor = a different language's speaker
$P run_sim.py --outdir ../outputs/en/chatterbox --ref ../references/en/en_ref_main.wav \
   --ref2 ../references/en/en_ref_02.wav --other ../references/hi/hi_ref_main.wav 2>/dev/null | tail -3
$P run_sim.py --outdir ../outputs/ar/chatterbox --ref ../references/ar/ar_ref_10s.wav \
   --ref2 ../references/ar/ar_ref_02.wav --other ../references/en/en_ref_main.wav 2>/dev/null | tail -3
$P run_sim.py --outdir ../outputs/hi/chatterbox --ref ../references/hi/hi_ref_10s.wav \
   --ref2 ../references/hi/hi_ref_02.wav --other ../references/en/en_ref_main.wav 2>/dev/null | tail -3
echo "@@@ ALL CLONING EVAL DONE"
