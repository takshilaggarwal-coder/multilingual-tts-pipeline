| System | Hard-token WER | Expressiveness (emo/neutral F0) | Mean F0 std (Hz) | Max clipped |
|---|---|---|---|---|
| ar/chatterbox | 0.2642 | 0.62 | 27.9 | 0.0 |
| ar/habibi | 0.2453 | 1.31 | 42.5 | 1e-05 |
| ar/mms | 0.4717 | 0.72 | 33.2 | 0.0 |
| ar/mms_vd | 0.3962 | 0.94 | 34.0 | 0.0 |
| en/chatterbox | 0.5854 | 3.12 | 33.8 | 0.0 |
| en/kokoro | 0.0488 | 1.17 | 38.6 | 0.0 |
| hi/chatterbox | 0.2784 | 0.72 | 17.7 | 0.0 |
| hi/kokoro | 0.1546 | 1.14 | 40.9 | 0.0 |

*Hard-token WER* = WER on the names/numbers/currency/digits/acronym rows only (the tokens a voice agent must not fumble). *Expressiveness* = ratio of pitch spread on the emotion row to the neutral row; ~1.0 means a flat reader, higher means the model modulates for affect. *Max clipped* = worst per-clip fraction of samples at full scale (artefact check).
