# 2025_BANANA

AI model for Korea Computer Congress 2025

> with: Woohyun Park, Gyumin Kim

Woohyun Park, Kangmin Ra, Gyumin Kim. **"Prediction of Banana Freshness Using an Encoder-Decoder LSTM Model with Explainable AI**. KIISE KCC 2025 Proceedings. 2025.7

This project predicts a banana’s entire freshness trajectory from only the first 10–30% of an image sequence captured hourly in varied environments. We propose a variable-length Encoder–Decoder LSTM that supports early-partial inference.

#### dataset

Dataset folder provides banana time-series images and their corresponding labels, grouped by sequence.

Each sequence is contained in a `banana_n/` directory with images and labels separated.

```
Dataset/
├─ banana_n/
│  ├─ image/    # time-series images
│  └─ label/    # labels for the sequence
```

For training/evaluation, use only the `image/` and `label/` folders inside each `banana_n/`.
