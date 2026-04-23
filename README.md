# Heart Disease Prediction using Deep Learning with Explainable AI (SHAP)

This project provides a browser-based heart disease risk dashboard with:

- live patient input validation
- real-time prediction updates
- dynamic ANN visualization
- SHAP-based explainability
- downloadable PDF patient reports

The main production web interface is a Flask app served by `webapp.py`.

## Main Features

- Predicts heart disease risk from patient inputs
- Compares Logistic Regression, Random Forest, and ANN outputs
- Shows a live ANN graph that updates when inputs change
- Displays SHAP feature impact in the browser
- Lets users download a PDF report after prediction
- Works on Windows without TensorFlow at runtime by using a portable ANN inference loader

## Project Structure

- `webapp.py`
  Main Flask application and API routes.

- `api_service.py`
  Input validation, prediction pipeline, SHAP response building, and ANN visualization payload generation.

- `ann_runtime.py`
  Portable ANN runtime that loads the saved `.keras` model and computes ANN activations with NumPy and `h5py`.

- `static/app.js`
  Frontend logic for live form updates, debounced API calls, ANN rendering, SHAP charts, and PDF download.

- `static/styles.css`
  Main UI styling for the dashboard.

- `templates/index.html`
  Main browser dashboard template.

- `reporting.py`
  PDF report generation using `reportlab`.

- `model.py`
  Model training, artifact loading, and runtime prediction integration.

- `preprocess.py`
  Synthetic dataset generation, schema definition, encoding, scaling, and preprocessing.

- `train_and_save_model.py`
  Retrains and saves project artifacts.

- `app.py`
  Separate Streamlit research/demo app. This is optional and not the main production dashboard.

## Dataset Note

This repository uses a synthetic but clinically styled heart disease dataset generated in code by `create_synthetic_heart_dataset()` inside `preprocess.py`.

Saved artifacts include:

- `artifacts/project_bundle.pkl`
- `artifacts/ann_model.keras`
- `artifacts/model_comparison.csv`
- `artifacts/heart_disease_research_dataset.csv`

## Dependencies

All required Python dependencies are listed in `requirements.txt`.

Pinned packages:

- `pandas==3.0.2`
- `numpy==2.4.4`
- `scikit-learn==1.8.0`
- `matplotlib==3.10.8`
- `seaborn==0.13.2`
- `streamlit==1.56.0`
- `flask==3.1.3`
- `joblib==1.5.3`
- `plotly==6.6.0`
- `shap==0.51.0`
- `reportlab==4.4.10`
- `h5py==3.16.0`
- `tensorflow` on non-Windows systems only

## Setup

### 1. Create a virtual environment

Windows PowerShell:

```powershell
cd "c:\Users\sumeet verma\Desktop\New folder2\Heart-diseaseprediction"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
cd /path/to/Heart-diseaseprediction
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

Windows:

```powershell
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -r requirements.txt
```

Linux/macOS:

```bash
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```

## Run the Main Flask Dashboard

Windows:

```powershell
.\.venv\Scripts\python webapp.py
```

Linux/macOS:

```bash
.venv/bin/python webapp.py
```

Open:

```text
http://127.0.0.1:5000
```

## Available Routes

- `GET /`
  Main dashboard

- `GET /health`
  Health check endpoint

- `POST /api/predict`
  Prediction API returning:
  - validated input data
  - model probabilities
  - primary prediction
  - SHAP values
  - feature importance
  - ANN visualization data
  - inference time

- `POST /download-report`
  Generates and downloads a PDF report

## Retraining the Models

If you want to regenerate model artifacts:

```powershell
.\.venv\Scripts\python train_and_save_model.py
```

Important:

- On Windows, runtime prediction works without TensorFlow because the app can read the saved ANN model using the portable loader in `ann_runtime.py`.
- Full ANN retraining still requires TensorFlow, so Linux, WSL, or Vagrant is the safest environment for retraining.

## Optional Streamlit Research App

If you also want to run the older Streamlit research interface:

```powershell
.\.venv\Scripts\python -m streamlit run app.py --server.headless true --browser.gatherUsageStats false
```

Default URL:

```text
http://localhost:8501
```

## Vagrant / Ubuntu Workflow

If you want a Linux environment for full TensorFlow retraining:

From the outer workspace:

```powershell
cd "c:\Users\sumeet verma\Desktop\New folder2"
$env:VAGRANT_HOME="c:\Users\sumeet verma\Desktop\New folder2\.vagrant-home"
vagrant up
vagrant ssh
```

Inside the VM:

```bash
cd /vagrant/Heart-diseaseprediction
python3 -m venv ~/heartenv
~/heartenv/bin/python -m pip install --upgrade pip
~/heartenv/bin/python -m pip install -r requirements.txt
~/heartenv/bin/python train_and_save_model.py
~/heartenv/bin/python webapp.py
```

## Handoff Notes

If you give this project to someone else, they should:

1. Clone or copy the full project folder including `artifacts/`, `templates/`, and `static/`.
2. Create a virtual environment.
3. Install `requirements.txt`.
4. Run `python webapp.py`.
5. Open `http://127.0.0.1:5000`.

## Verified in This Workspace

The following were verified after the updates:

- prediction API responds successfully
- ANN visualization payload is returned dynamically
- PDF report generation works
- `/download-report` returns a real PDF file
- frontend and backend dependencies are listed in `requirements.txt`


'only final check '
