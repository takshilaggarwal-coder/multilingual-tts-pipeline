#!/usr/bin/env python
"""Build the blinded human listening kit (MOS + cloning A/B).

    python make_kit.py            # from repo root or eval/listening_test/

Takes every benchmarked pipeline in outputs/<lang>/<model>/, samples a fixed
utterance subset per system, copies clips under opaque randomized names into
kit/<lang>/, and hides a real-speech anchor (references/<lang>/*_ref_main.wav)
in the same pool. Raters never see which file is which system; the mapping
lives in key.json (git-ignored inside the kit) until scoring.

Outputs per language:
  kit/<lang>/c<nn>.wav          blinded clips (systems + hidden real anchor)
  kit/<lang>/RATING_SHEET.csv   one row per clip: naturalness 1-5 (ACR)
  kit/ab/<lang>_pair<k>_{A,B}.wav + AB_SHEET.csv   cloning same-speaker A/B
  kit/INSTRUCTIONS.md           P.808-style ACR wording, per-language notes
  kit/key.json                  clip -> (system, utterance) mapping [do not share]

Protocol: ITU-T P.808-lite ACR. >=3 listeners per language, native/fluent for
ar/hi. Score with score_kit.py after collecting filled sheets.
"""
import csv
import json
import random
import shutil
import sys
from pathlib import Path

SEED = 20260715
UTT_IDS = ["01", "02", "05", "08", "12"]  # fixed subset: latency/short/names/emotion/neutral
REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs"
REFS = REPO / "references"
KIT = Path(__file__).resolve().parent / "kit"

INSTRUCTIONS = """# Listening test — instructions (please read first)

Thank you for helping. This takes about 10 minutes per language.

## Part 1 — Naturalness (RATING_SHEET.csv in each language folder)

Listen to each clip ONCE with headphones, then rate how natural — how human —
the voice sounds, on this scale (ITU ACR):

  5  Excellent — completely natural, indistinguishable from a person
  4  Good — natural, minor artifacts you notice but don't mind
  3  Fair — noticeably synthetic, but easy to listen to
  2  Poor — clearly robotic / distorted in places
  1  Bad — unpleasant, hard to listen to

Rate the VOICE, not the sentence content. Don't compare clips to each other —
rate each on its own. Put your number in the `naturalness_1_to_5` column.
If a clip has a glitch (silence, cutoff, wrong language), note it in `comment`.

Please do the language(s) you speak fluently; for a language you understand
partially, you may still rate naturalness but write your fluency in the sheet
header row where indicated.

## Part 2 — Same speaker? (kit/ab/AB_SHEET.csv)

Each pair has file A (a real recording) and file B (synthesized). Question:
does B sound like THE SAME PERSON as A? Answer `same`, `unsure`, or `different`
in the `same_speaker` column. Judge the voice identity (timbre, pitch), not
audio quality or accent.

Save the CSVs and send them back. Your name/initials go in the `rater` column.
"""


def collect_systems():
    systems = {}
    for lang_dir in sorted(OUT.glob("*")):
        if not lang_dir.is_dir() or lang_dir.name.startswith("_"):
            continue
        for model_dir in sorted(lang_dir.glob("*")):
            if (model_dir / "timings.json").exists():
                systems.setdefault(lang_dir.name, []).append(model_dir)
    return systems


def main():
    rng = random.Random(SEED)
    if KIT.exists():
        shutil.rmtree(KIT)
    (KIT / "ab").mkdir(parents=True)

    key = {"seed": SEED, "clips": {}, "ab_pairs": {}}
    systems = collect_systems()
    if not systems:
        sys.exit("no benchmarked systems found under outputs/")

    for lang, model_dirs in systems.items():
        pool = []
        for md in model_dirs:
            for uid in UTT_IDS:
                wav = md / f"{lang}_{uid}.wav"
                if wav.exists():
                    pool.append((f"{md.parent.name}/{md.name}", wav))
        anchor = REFS / lang / f"{lang}_ref_main.wav"
        if anchor.exists():
            pool.append(("REAL_ANCHOR", anchor))
        rng.shuffle(pool)

        lang_kit = KIT / lang
        lang_kit.mkdir(parents=True)
        rows = []
        for i, (system, wav) in enumerate(pool, 1):
            name = f"c{i:02d}.wav"
            shutil.copy2(wav, lang_kit / name)
            key["clips"][f"{lang}/{name}"] = {"system": system, "source": str(wav.relative_to(REPO))}
            rows.append(name)

        with open(lang_kit / "RATING_SHEET.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["rater", "fluency_note", "clip", "naturalness_1_to_5", "comment"])
            for name in rows:
                w.writerow(["", "", name, "", ""])
        print(f"[{lang}] {len(rows)} blinded clips ({len(model_dirs)} systems"
              f"{' + real anchor' if anchor.exists() else ''})")

    # cloning A/B pairs: real reference vs cloned output, per cloning system
    ab_rows = []
    k = 0
    for lang, model_dirs in systems.items():
        ref = REFS / lang / f"{lang}_ref_main.wav"
        if not ref.exists():
            continue
        for md in model_dirs:
            cfg = json.loads((md / "timings.json").read_text()).get("config", {})
            if not cfg.get("ref_audio"):
                continue  # not a cloning system
            uid = rng.choice(UTT_IDS)
            wav = md / f"{lang}_{uid}.wav"
            if not wav.exists():
                continue
            k += 1
            pa, pb = KIT / "ab" / f"{lang}_pair{k}_A.wav", KIT / "ab" / f"{lang}_pair{k}_B.wav"
            shutil.copy2(ref, pa)
            shutil.copy2(wav, pb)
            key["ab_pairs"][f"{lang}_pair{k}"] = {"system": f"{md.parent.name}/{md.name}",
                                                  "clone": str(wav.relative_to(REPO))}
            ab_rows.append(f"{lang}_pair{k}")

    with open(KIT / "ab" / "AB_SHEET.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rater", "pair", "same_speaker (same/unsure/different)", "comment"])
        for p in ab_rows:
            w.writerow(["", p, "", ""])
    print(f"[ab] {len(ab_rows)} cloning A/B pairs")

    (KIT / "INSTRUCTIONS.md").write_text(INSTRUCTIONS)
    (KIT / "key.json").write_text(json.dumps(key, indent=2))
    print(f"\nkit -> {KIT}\nSend raters the kit folder MINUS key.json.")


if __name__ == "__main__":
    main()
