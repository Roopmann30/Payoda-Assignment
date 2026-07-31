# Spaceship Titanic - Machine Learning Solution

This repository contains an end-to-end machine learning pipeline for the Kaggle **Spaceship Titanic** competition (Binary Classification). The goal is to predict whether passengers were transported to an alternate dimension based on their personal records, amenities usage, and cabin locations.

## Approach & Methodology
* **Feature Engineering**: Extracted passenger groups (`GroupSize`), segmented cabin details (`Deck`, `CabinNum`, `Side`), log-transformed luxury spending features, and created interaction features between home planets and destinations.
* **Missing Value Imputation**: Handled missing categorical attributes with `'Unknown'`, zero-filled financial expenditures, filled ages with medians, and defaulted missing booleans to `False`.
* **Model Architecture**: A robust soft-voting ensemble comprising **LightGBM**, **XGBoost**, and **CatBoost**.
* **Validation Strategy**: 10-Fold Stratified Cross-Validation ensuring unbiased out-of-fold (OOF) performance evaluation.

## OOF Performance Metrics
* **CV Accuracy (mean)**: 0.8126
* **OOF Accuracy**: 0.8126
* **OOF Precision**: 0.8127
* **OOF Recall**: 0.8159
* **OOF F1 Score**: 0.8143
* **OOF ROC-AUC**: 0.9043
* **OOF Log Loss**: 0.3812

## Project Structure
```text
├── spaceship_titanic_84.py   # Main end-to-end training and prediction script
├── requirements.txt          # Python dependencies
├── submission.csv            # Final Kaggle-formatted submission file
└── README.md                 # Project documentation