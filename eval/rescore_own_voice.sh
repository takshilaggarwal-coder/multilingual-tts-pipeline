#!/bin/zsh
# Re-score EN/HI chatterbox after the own-voice re-clone: WER + MOS + SIM.
# SIM floors now use the Arabic corpus narrator (EN/HI refs are the same person,
# so cross-language own-voice would no longer be a "different speaker" floor).
# Also re-anchors AR's floor to the new en_ref_main for reproducibility.
set -e
cd "$(dirname "$0")"
P=../envs/eval/bin/python

for L in en hi; do
  echo "@@@ $L WER"
  $P run_wer.py --outdir ../outputs/$L/chatterbox --texts texts/$L.tsv --lang $L 2>/dev/null | tail -1
  echo "@@@ $L MOS"
  $P run_mos.py --outdir ../outputs/$L/chatterbox --texts texts/$L.tsv 2>/dev/null | tail -1
done

echo "@@@ SIM"
$P run_sim.py --outdir ../outputs/en/chatterbox --ref ../references/en/en_ref_10s.wav \
   --ref2 ../references/en/en_ref_02.wav --other ../references/ar/ar_ref_10s.wav 2>/dev/null | tail -3
$P run_sim.py --outdir ../outputs/hi/chatterbox --ref ../references/hi/hi_ref_10s.wav \
   --ref2 ../references/hi/hi_ref_02.wav --other ../references/ar/ar_ref_10s.wav 2>/dev/null | tail -3
$P run_sim.py --outdir ../outputs/ar/chatterbox --ref ../references/ar/ar_ref_10s.wav \
   --ref2 ../references/ar/ar_ref_02.wav --other ../references/en/en_ref_main.wav 2>/dev/null | tail -3
$P run_sim.py --outdir ../outputs/ar/habibi --ref ../references/ar/ar_ref_10s.wav \
   --ref2 ../references/ar/ar_ref_02.wav --other ../references/en/en_ref_main.wav 2>/dev/null | tail -3

echo "@@@ own-voice MOS anchors (en/hi refs changed)"
$P - <<'PY'
import json
from mos import predicted_mos
out = json.load(open("../results/mos_anchors.json"))
for L in ("en","hi"):
    out[L] = predicted_mos(f"../references/{L}/{L}_ref_main.wav")
    print(f"  {L} anchor now: {out[L]}")
json.dump(out, open("../results/mos_anchors.json","w"), indent=2, ensure_ascii=False)
PY
echo "@@@ DONE"
