# Enhancing Heart Disease Prediction Using Deep Learning with Explainable AI (SHAP)

This project compares three binary classification models for heart disease prediction:

- Logistic Regression
- Random Forest
- Artificial Neural Network built with TensorFlow/Keras

The app is designed for final-year project work, portfolio demonstrations, and research-style discussion. It includes preprocessing, model evaluation, ANN regularization, early stopping, and SHAP-based explanations.

## Project Structure

- `preprocess.py` handles dataset generation, scaling, optional feature selection, and single-patient preprocessing.
- `model.py` trains Logistic Regression, Random Forest, and ANN models, saves artifacts, and exposes inference helpers.
- `evaluate.py` computes Accuracy, Precision, Recall, F1-score, and ROC-AUC.
- `explain.py` generates SHAP summary and waterfall plots for Random Forest and ANN.
- `app.py` is the Streamlit frontend.
- `train_and_save_model.py` is a convenience training entry point.

## Why Preprocessing Matters

`StandardScaler` is important because Logistic Regression and ANN models are sensitive to feature magnitude. Scaled features help the optimizer converge faster and make learned weights more stable. Optional `SelectKBest` can reduce noise and produce a cleaner research comparison.

## Installation

```powershell
cd "c:\Users\sumeet verma\Desktop\New folder2\Heart-diseaseprediction"
.\.venv\Scripts\python -m pip install -r requirements.txt
```

## Train the Models

```powershell
.\.venv\Scripts\python train_and_save_model.py
```

## Run the Streamlit App

```powershell
.\.venv\Scripts\python -m streamlit run app.py --server.headless true --browser.gatherUsageStats false
```

## Notes

- The current repository does not include a raw CSV dataset, so the training pipeline builds a reproducible synthetic clinical-style dataset. This keeps the project runnable end to end.
- Saved artifacts are written to the `artifacts/` directory.
- SHAP is available for Random Forest and ANN inside the app.
- In this repository's current Windows environment, TensorFlow wheels are not available from the configured package index. The ANN training path is ready in code, but the full deep learning workflow should be run inside the Ubuntu Vagrant VM or another Linux environment.
