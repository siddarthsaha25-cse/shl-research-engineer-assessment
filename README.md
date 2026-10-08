# SHL Research Engineer Challenge - Speech Assessment Modeling

## Overview
This repository contains the machine learning pipeline developed for the SHL Research Engineer assessment. The objective is to predict evaluation ratings from speech audio recordings.

## Methodology
- **Acoustic Features:** Extracted temporal dynamics (speech activity ratio, duration, RMS, zero-crossing rate), spectral descriptors (centroid, bandwidth, rolloff, flatness, contrast across 7 bands), and phonetic representations (MFCCs with delta and delta-delta, Chroma, Tonnetz).
- **Cross-Validation:** Leakage-free 5-Fold Cross Validation.
- **Ensemble Model:** Weighted ensemble combining LightGBM, CatBoost, and Ridge Regression.

## Results
- Baseline LightGBM: 0.7674
- Advanced Acoustic Ensemble: **0.6944** (Best Score)
