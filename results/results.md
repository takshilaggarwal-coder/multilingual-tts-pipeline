| Lang | Model | Distill-MOS | UTMOS | Spk cos | Lat full (s) | 1st chunk (s) | RTF med | WER |
|---|---|---|---|---|---|---|---|---|
| ar | mms | 4.447 | 3.272 | - | 1.222 | 1.222 | 0.222 | 0.302 |
| ar | mms_vd | 4.442 | 3.371 | - | 0.731 | 0.731 | 0.126 | 0.222 |
| en | kokoro | 4.630 | 4.512 | - | 0.287 | 0.279 | 0.095 | 0.013 |
| hi | kokoro | 4.638 | 4.260 | - | 0.484 | 0.465 | 0.092 | 0.111 |

**Targets:** MOS ≥4.0 · speaker cosine ≥0.75 · latency <2 s full / <0.5 s first-chunk · RTF ≤0.5 · round-trip WER ≤0.10. Predicted MOS (Distill/UTMOS) is an English-trained proxy — human MOS is the graded metric; for AR/HI compare against the per-language real-speech anchor, not the absolute number.
