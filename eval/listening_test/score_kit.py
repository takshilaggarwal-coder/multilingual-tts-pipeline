#!/usr/bin/env python
"""Score filled listening-test sheets -> human MOS per system + A/B verdicts.

    python score_kit.py filled_sheets/*.csv [--ab filled_ab/*.csv]

Reads any number of filled RATING_SHEET.csv copies (one per rater), joins with
kit/key.json, and reports per-system MOS with rater count and a 95% CI
(t-distribution over per-clip ratings). A/B sheets aggregate to
same/unsure/different counts per cloning system. Writes results to
results/human_mos.json in the repo root.
"""
import argparse
import csv
import json
import math
import statistics as st
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
KEY = json.loads((HERE / "kit" / "key.json").read_text())

# two-sided 95% t critical values by df (capped at 30 -> 1.96)
T95 = {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57, 6: 2.45, 7: 2.36, 8: 2.31,
       9: 2.26, 10: 2.23, 15: 2.13, 20: 2.09, 25: 2.06, 30: 1.96}


def t95(df):
    if df <= 0:
        return float("nan")
    return T95.get(df, T95[max(k for k in T95 if k <= df)] if df > 30 else
                   T95[min(k for k in T95 if k >= df)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sheets", nargs="+", help="filled RATING_SHEET.csv files")
    ap.add_argument("--ab", nargs="*", default=[], help="filled AB_SHEET.csv files")
    args = ap.parse_args()

    ratings = defaultdict(list)  # system -> [scores]
    raters = set()
    for path in args.sheets:
        lang = Path(path).parent.name  # kit/<lang>/RATING_SHEET.csv layout preserved
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                score = (row.get("naturalness_1_to_5") or "").strip()
                if not score:
                    continue
                clip_key = f"{lang}/{row['clip']}"
                meta = KEY["clips"].get(clip_key)
                if meta is None:
                    continue
                ratings[meta["system"]].append(float(score))
                if row.get("rater"):
                    raters.add((row["rater"], lang))

    out = {"n_sheets": len(args.sheets), "raters": sorted({r for r, _ in raters}), "systems": {}}
    print(f"{'system':28} {'MOS':>5} {'95%CI':>12} {'n':>4}")
    for system, scores in sorted(ratings.items()):
        mos = st.mean(scores)
        ci = (t95(len(scores) - 1) * st.stdev(scores) / math.sqrt(len(scores))
              if len(scores) > 1 else float("nan"))
        out["systems"][system] = {"mos": round(mos, 2), "ci95": round(ci, 2),
                                  "n_ratings": len(scores)}
        print(f"{system:28} {mos:5.2f} {f'±{ci:.2f}':>12} {len(scores):>4}")

    if args.ab:
        ab = defaultdict(lambda: defaultdict(int))
        for path in args.ab:
            with open(path, newline="") as f:
                for row in csv.DictReader(f):
                    v = (row.get("same_speaker (same/unsure/different)") or "").strip().lower()
                    if v in ("same", "unsure", "different"):
                        meta = KEY["ab_pairs"].get(row["pair"])
                        if meta:
                            ab[meta["system"]][v] += 1
        out["ab"] = {k: dict(v) for k, v in ab.items()}
        print("\nA/B same-speaker judgments:")
        for system, counts in ab.items():
            print(f"  {system:26} {dict(counts)}")

    res = REPO / "results" / "human_mos.json"
    res.write_text(json.dumps(out, indent=2))
    print(f"\n-> {res}")


if __name__ == "__main__":
    main()
