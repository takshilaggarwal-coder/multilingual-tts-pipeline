#!/bin/zsh
# Habibi Arabic eval (WER+MOS+sim) + real-speech predicted-MOS anchors per language.
set -e
cd "$(dirname "$0")"
P=../envs/eval/bin/python

echo "@@@ habibi WER"
$P run_wer.py --outdir ../outputs/ar/habibi --texts texts/ar.tsv --lang ar 2>/dev/null | tail -1
echo "@@@ habibi MOS"
$P run_mos.py --outdir ../outputs/ar/habibi --texts texts/ar.tsv 2>/dev/null | tail -1
echo "@@@ habibi SIM"
$P run_sim.py --outdir ../outputs/ar/habibi --ref ../references/ar/ar_ref_10s.wav \
   --ref2 ../references/ar/ar_ref_02.wav --other ../references/en/en_ref_main.wav 2>/dev/null | tail -3

echo "@@@ real-speech MOS anchors"
$P - <<'PY'
import json
from mos import predicted_mos
out={}
for L in ("en","ar","hi"):
    out[L]=predicted_mos(f"../references/{L}/{L}_ref_main.wav")
    print(f"  {L} anchor: {out[L]}")
json.dump(out, open("../results/mos_anchors.json","w"), indent=2, ensure_ascii=False)
print("-> results/mos_anchors.json")
PY
echo "@@@ DONE"
