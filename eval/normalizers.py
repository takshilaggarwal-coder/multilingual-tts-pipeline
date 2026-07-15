"""Per-language text normalization for round-trip WER.

WER between raw TTS input and raw ASR output is dominated by orthographic noise
(diacritics, digit verbalization, punctuation), not intelligibility. Both the
reference text and the ASR transcript are passed through the same normalizer
before scoring. Choices here are disclosed in the write-up:

- en: OpenAI Whisper's EnglishTextNormalizer (case, punctuation, number formats).
- ar: NFC → strip tashkeel/tatweel → unify alef/hamza/yaa/taa-marbuta forms →
      Eastern-Arabic digits → ASCII → digits verbalized with num2words(ar) →
      strip punctuation.
- hi: NFC → Devanagari digits → ASCII → nukta folding (क़→क …) → danda/punct
      stripped → digits verbalized with num2words(hi).

Digit verbalization on BOTH sides means "1249" matches whether the ASR emits
digits or words — but multi-digit strings read digit-by-digit by a model
(e.g. train numbers) will still count as errors; that is intentional and noted
as a failure mode rather than normalized away.
"""
import re
import unicodedata

try:
    from num2words import num2words
except ImportError:  # pragma: no cover
    num2words = None

_PUNCT = re.compile(r"[^\w\s]|_", re.UNICODE)
_WS = re.compile(r"\s+")

_AR_DIACRITICS = re.compile(
    r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06DC\u06DF-\u06E8\u06EA-\u06ED]"
)
_AR_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
_DEV_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
_NUM = re.compile(r"\d+")


def _verbalize_digits(text, lang):
    if num2words is None:
        return text

    def repl(m):
        try:
            return " " + num2words(int(m.group()), lang=lang) + " "
        except (NotImplementedError, OverflowError, ValueError):
            return m.group()

    return _NUM.sub(repl, text)


def _strip_punct(text):
    return _WS.sub(" ", _PUNCT.sub(" ", text)).strip()


def normalize_english(text):
    try:
        from whisper_normalizer.english import EnglishTextNormalizer
        return EnglishTextNormalizer()(text)
    except ImportError:
        text = text.lower()
        text = _verbalize_digits(text, "en")
        return _strip_punct(text)


def normalize_arabic(text):
    text = unicodedata.normalize("NFC", text)
    text = _AR_DIACRITICS.sub("", text)
    text = text.replace("ـ", "")  # tatweel
    text = re.sub("[إأآٱ]", "ا", text)
    text = text.replace("ؤ", "و").replace("ئ", "ي").replace("ء", "")
    text = text.replace("ى", "ي")
    text = text.replace("ة", "ه")
    text = text.translate(_AR_DIGITS)
    text = _verbalize_digits(text, "ar")
    # re-apply letter unification: num2words output may contain hamza forms
    text = re.sub("[إأآٱ]", "ا", text).replace("ة", "ه").replace("ى", "ي")
    text = _AR_DIACRITICS.sub("", text)
    return _strip_punct(text)


def _fold_nukta(text):
    # decompose so nukta (U+093C) becomes a separate combining mark, drop it
    decomposed = unicodedata.normalize("NFD", text)
    return unicodedata.normalize("NFC", decomposed.replace("़", ""))


def normalize_hindi(text):
    text = unicodedata.normalize("NFC", text)
    text = text.translate(_DEV_DIGITS)
    text = _fold_nukta(text)
    text = text.replace("।", " ").replace("॥", " ")
    text = _verbalize_digits(text, "hi")
    return _strip_punct(text)


NORMALIZERS = {
    "en": normalize_english,
    "ar": normalize_arabic,
    "hi": normalize_hindi,
}
