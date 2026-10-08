# SHL Research Engineer Challenge - Speech Assessment Modeling

## Overview
This repository contains the machine learning pipeline developed for the SHL Research Engineer assessment. The objective is to predict evaluation ratings from speech audio recordings.

## Experiments & Scripts
- `baseline.py`: Baseline LightGBM model utilizing primary acoustic descriptors (Public Score: 0.7674).
- `advanced_model.py`: **Final winning solution** (Public Score: 0.6944). Comprehensive acoustic extraction (prosodic dynamics, spectral contrast, MFCC deltas, Tonnetz) with a 5-Fold cross-validated ensemble of LightGBM, CatBoost, and Ridge.
- `advanced_tuned.py`: Experimentation with higher-order statistical moments (skew/kurtosis) and ExtraTrees.
- `wav2vec_model.py`: Self-supervised speech representation experiment utilizing pre-trained `facebook/wav2vec2-base` embeddings.

## Results Summary
- Baseline LightGBM: 0.7674
- **Advanced Acoustic Ensemble: 0.6944 (Selected Best)**
