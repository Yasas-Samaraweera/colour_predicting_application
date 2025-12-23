# 🎨 Colour Predicting Application — Textile Dyeing

An AI-driven predictive system for textile dyeing colour estimation. This project predicts final fabric colour and shade quality from chemical and process parameters (not from images). It models how reactive dye properties and dyeing conditions influence resulting colour on cellulosic fabrics — useful for simulation, recipe tuning, and process optimization to reduce trial-and-error in industrial dyeing.

## Key ideas

- Input: dye type, concentration, temperature, time, pH, salt, liquor ratio, and related process parameters  
- Output: predicted colour (CIELAB / RGB / HEX), ΔE, confidence score, and process recommendations  
- Goal: reduce sampling runs and speed up recipe tuning for reactive dyeing on cellulosic fibers

## Features

- Predict final colour in CIELAB (L*, a*, b*), RGB, and HEX colour spaces  
- Estimate colour difference (ΔE) to a target shade  
- Support reactive dye–specific parameters (concentration, salt, alkali, time, temperature)  
- Recommend process modifications (temperature, time, salt, pH) to improve match  
- Batch simulation of multiple dyeing recipes  
- CLI and lightweight API / Web interface (FastAPI / Streamlit optional)  
- Extensible and modular codebase for model and data improvements

## Typical inputs

- Dye identifier and properties  
- Dye concentration (g/L or % owf)  
- Temperature and fixation time  
- pH / alkali dosing  
- Salt concentration  
- Liquor ratio (LR)  
- Fabric type (cotton, blends) and GSM  
- Auxiliary chemicals and process history

## Typical outputs

- Predicted colour values (L*, a*, b*, RGB, HEX)  
- Estimated ΔE to a target shade  
- Prediction confidence score  
- Suggested recipe or parameter adjustments  
- JSON and tabular outputs for integration

## Tech stack

- Python 3.8+  
- scikit-learn (Random Forest regression or other regressors)  
- NumPy, Pandas  
- FastAPI or Streamlit for optional UI  
- CLI-based inference script for quick usage

## Getting started

Clone the repository and create a virtual environment:

```bash
git clone https://github.com/Yasas-Samaraweera/colour_predicting_application.git
cd colour_predicting_application

python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## CLI example

Run a prediction from the command line:

```bash
python predict.py \
  --dye "ReactiveRedX" \
  --concentration 2.5 \
  --temperature 60 \
  --time 40 \
  --pH 10.5 \
  --salt 40 \
  --liquor-ratio "10:1" \
  --fabric "cotton"
```

(The exact CLI flags may vary depending on `predict.py` implementation. Use `python predict.py --help` to list available options.)

## Project structure

```
data/
notebooks/
src/
  ├── preprocess.py
  ├── features.py
  ├── train.py
  ├── predict.py
app.py
predict.py
requirements.txt
README.md
```

- data/: raw and processed datasets, CSVs used for training and evaluation  
- notebooks/: EDA and modelling notebooks  
- src/: core code for preprocessing, feature engineering, training and prediction  
- predict.py / app.py: CLI script and optional web API entrypoints

## Notes & next steps

- Models and feature pipelines should be versioned (e.g., with MLflow or simple artifact folders).  
- Add unit/integration tests for preprocessing and inference paths.  
- Consider expanding model types (ensemble, neural nets) and tracking prediction uncertainty.  
- Provide example datasets and notebook demonstrating training → evaluation → prediction.

---

Maintainer: Yasas Samaraweera  
AI / ML Engineer – Textile & Manufacturing Applications
