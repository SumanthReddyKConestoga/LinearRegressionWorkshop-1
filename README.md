````markdown
# 🏭 Linear Regression Architecture Workshop — Robot Failure Prediction (MLOps Ready)

## Executive Summary
This project builds a **Univariate Linear Regression** solution to **predict when the next robot failure is likely to occur** using robot sensor signals (Axis measurements).  
We implement Linear Regression in two ways—**from scratch** (gradient descent) and **scikit-learn**—then compare performance and generate clear plots as evidence.  
The project follows **MLOps-style architecture**: modular code, config-driven experiments, reproducible runs, and experiment tracking.

---

## 🎯 Problem Statement
Manufacturing robots generate continuous sensor data. Failures are costly and often detected too late.  
Our goal is to use one key sensor feature (example: `Axis #1`) to predict:

✅ **Time remaining until the next failure event** (e.g., `time_to_failure_days`)

This supports **predictive maintenance** by enabling proactive alerts (e.g., “raise an alert ~2 weeks before failure”).

---

## ✅ What We Built (Workshop Deliverables)
### Session 1 — Linear Regression
- Loaded robot CSV into Pandas and inspected data quality
- Preprocessed data (missing values, normalization, train/test split)
- Implemented **manual Linear Regression** (MSE + Gradient Descent)
- Implemented **scikit-learn Linear Regression** for comparison
- Evaluated with: **RMSE, MAE, R²**
- Produced regression plots to show model fit

### Session 2 — MLOps Architecture
- Refactored notebook logic into modular scripts (`src/`)
- Parameterized experiments using YAML config (`configs/experiment_config.yaml`)
- Saved experiment outputs to:
  - `experiments/results.csv` (metrics tracking)
  - `experiments/plots/` (visual proof)
- Ensured reproducibility: anyone can clone + run and get the same outputs

---

## 📌 How “Failure” is Defined in This Workshop
The dataset does not include an explicit `failure = 1` column.  
So we define failure events **from abnormal sensor behavior**, using a simple and explainable rule:

- Compute anomaly score (e.g., rolling z-score) on a selected axis
- Mark a **failure event** when the sensor deviation crosses a threshold
- Compute target label:

✅ `time_to_failure_days = (next_failure_time - current_time)`

Then Linear Regression learns:

**Sensor Axis value → time remaining until next failure**

---

## 🗂️ Project Folder Structure
```text
LinearRegressionArchitecture_Workshop1/
│── configs/
│   └── experiment_config.yaml
│
│── data/
│   ├── raw/
│   │   └── RMBR4-2_export_test.csv
│   └── processed/
│       └── processed_robot_data.csv
│
│── experiments/
│   ├── results.csv
│   └── plots/
│       ├── robot_pm_univariate_v1_scratch_scatter_line.png
│       ├── robot_pm_univariate_v1_scratch_residuals.png
│       ├── robot_pm_univariate_v1_sklearn_scatter_line.png
│       └── robot_pm_univariate_v1_sklearn_residuals.png
│
│── notebooks/
│   ├── EDA.ipynb
│   ├── linear_regression.ipynb
│   └── RobotPM_MLOps.ipynb
│
│── dashboard/
│   └── app.py
│
│── scripts/
│   └── generate_notebooks.py
│
│── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── model.py
│   ├── evaluation.py
│   └── run_experiment.py
│
│── requirements.txt
└── README.md
````

---

## 🧠 What Each Module Does (3 lines each)

### `src/data_loader.py`

* Loads robot sensor data from CSV (and supports DB/API expansion).
* Ensures consistent column formats and clean DataFrame output.
* Supplies the raw input needed for failure-time prediction.

### `src/preprocessing.py`

* Cleans missing values, sorts by time, normalizes features.
* Creates the prediction label: **time until next failure**.
* Outputs model-ready X (sensor axis) and y (time-to-failure).

### `src/model.py`

* Implements Linear Regression **from scratch** using gradient descent.
* Runs scikit-learn LinearRegression as the baseline comparison.
* Produces predicted values of **time until next failure**.

### `src/evaluation.py`

* Computes RMSE, MAE, and R² to measure model quality.
* Generates regression plots and residual diagnostics.
* Saves metrics and graphs as proof of failure prediction performance.

### `src/run_experiment.py`

* Orchestrates the full pipeline using YAML configuration.
* Runs preprocessing → training → evaluation → saves outputs.
* Produces repeatable results to predict **next failure timing**.

### `configs/experiment_config.yaml`

* Stores all experiment parameters (data path, feature axis, thresholds, learning rate).
* Enables reruns without changing code (config-driven workflow).
* Defines what “failure prediction” means for a given experiment.

---

## 📊 Outputs Produced

### 1) Experiment Tracking

* `experiments/results.csv`

  * Contains metrics for scratch vs sklearn models:

    * RMSE, MAE, R²
    * run_tag timestamp

### 2) Visual Proof (Plots)

Saved under `experiments/plots/`:

* **Scatter + Regression line** (model fit)
* **Residual plot** (error distribution)
* (Optional) time-series with failure markers if enabled

---

## ▶️ How to Run the Project (Step-by-step)

### 1) Create and activate virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

### 2) Install dependencies

```powershell
pip install -r requirements.txt
```

### 3) Run the full pipeline (recommended)

```powershell
python -m src.run_experiment
```

### 4) Open outputs

* Metrics: `experiments/results.csv`
* Plots: `experiments/plots/`

---

## 📓 Notebooks (For Presentation)

* `notebooks/EDA.ipynb`
  Data understanding: missing values, feature selection, trends

* `notebooks/linear_regression.ipynb`
  Manual LR vs sklearn comparison, plots, and metrics

* `notebooks/RobotPM_MLOps.ipynb`
  Documents the MLOps refactor, config-driven execution, tracking outputs

---

## 🖥️ (Optional) Run Dashboard

If you want a UI to view the dataset/stream:

```powershell
streamlit run dashboard/app.py
```

---

## 🧾 Key Design Decisions (MLOps)

* **Separation of concerns:** loader, preprocessing, model, evaluation are independent modules
* **Config-driven:** all tunable values live in YAML (no hard-coded magic values)
* **Experiment tracking:** results saved in `experiments/results.csv`
* **Reproducibility:** same config + same code = same output plots and metrics

---
