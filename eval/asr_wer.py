"""Round-trip intelligibility scoring: TTS output -> ASR -> WER vs. input text.

ASR: faster-whisper (CTranslate2) on CPU, int8. Model is selectable; default
large-v3 for best Arabic/Hindi accuracy on this 8 GB machine. The ASR language
is forced (never auto-detected) so a mispronounced clip is penalized as errors
rather than silently transcribed as another language.

Caveat disclosed in the write-up: whisper itself has nonzero WER on clean
human speech (especially ar/hi), so round-trip WER is an upper bound on the
TTS model's true intelligibility error. We report the same pipeline applied
to the human reference clips as a floor where available.
"""
import json
from pathlib import Path

import jiwer

from normalizers import NORMALIZERS

_MODEL_CACHE = {}


def get_model(model_size="large-v3", compute_type="int8"):
    key = (model_size, compute_type)
    if key not in _MODEL_CACHE:
        from faster_whisper import WhisperModel
        _MODEL_CACHE[key] = WhisperModel(model_size, device="cpu", compute_type=compute_type)
    return _MODEL_CACHE[key]


def transcribe(wav_path, language, model_size="large-v3", compute_type="int8"):
    model = get_model(model_size, compute_type)
    segments, _info = model.transcribe(
        str(wav_path),
        language=language,
        beam_size=5,
        vad_filter=False,
        condition_on_previous_text=False,
    )
    return " ".join(seg.text.strip() for seg in segments).strip()


def score_utterance(reference_text, wav_path, lang, model_size="large-v3"):
    normalize = NORMALIZERS[lang]
    hyp_raw = transcribe(wav_path, lang, model_size)
    ref = normalize(reference_text)
    hyp = normalize(hyp_raw)
    if not ref:
        raise ValueError(f"empty normalized reference for {wav_path}")
    measures = jiwer.process_words(ref, hyp if hyp else " ")
    return {
        "wer": round(measures.wer, 4),
        "substitutions": measures.substitutions,
        "deletions": measures.deletions,
        "insertions": measures.insertions,
        "ref_words": len(ref.split()),
        "asr_raw": hyp_raw,
        "ref_norm": ref,
        "hyp_norm": hyp,
    }


def corpus_wer(utterance_scores):
    """Aggregate WER over all utterances (error-weighted, not mean-of-WERs)."""
    errors = sum(u["substitutions"] + u["deletions"] + u["insertions"] for u in utterance_scores)
    words = sum(u["ref_words"] for u in utterance_scores)
    return round(errors / words, 4) if words else None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Score one clip (debugging)")
    parser.add_argument("wav")
    parser.add_argument("text")
    parser.add_argument("--lang", required=True, choices=["en", "ar", "hi"])
    parser.add_argument("--model", default="large-v3")
    args = parser.parse_args()
    print(json.dumps(score_utterance(args.text, Path(args.wav), args.lang, args.model),
                     indent=2, ensure_ascii=False))
