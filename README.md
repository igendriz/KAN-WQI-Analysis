# Interpretable Water Quality Analysis Using Kolmogorov-Arnold Networks (KAN)

This repository provides a structured and reproducible pipeline for modeling and interpreting water quality time series using **Kolmogorov-Arnold Networks (KANs)**. The project demonstrates how KAN models can be applied to IoT sensor data (pH, temperature, TDS) to predict the **Water Quality Index (WQI)** via interpretable symbolic expressions.

---

## 📌 Project objectives

* Train and evaluate symbolic regression models using KAN
* Derive human-readable symbolic expressions relating input features to WQI (both linear and non-linear aggregations)
* Assess model performance using regression metrics (MAE, RMSE, R²)
* Explore model sparsification for sensor reduction and deployment optimization
* Perform robust validation using **temporal cross-validation**

---

## 📁 Project structure

```text
/KAN-WQI-Analysis/
├── data/
│   ├── raw/                         # Original CSV data files
│   └── processed/                   # Cleaned, interpolated, and smoothed data
│
├── notebooks/
│   ├── case-study_01/               # Modeling and evaluation for WA-WQI (linear formulation)
│   ├── case-study_02/               # Symbolic model with temporal CV for WQM (non-linear formulation)
│   ├── EDA_Preprocessing.ipynb      # Visual data exploration and preprocessing
│   └── Baseline_wqi_models_temporal_cv.ipynb  # Temporal cross-validation of models
│
├── scripts/
│   ├── preprocessing.py             # Data cleaning and transformation pipeline
│   ├── train_model.py               # Training routines for KAN (under development)
│   └── evaluation.py                # Utilities for computing metrics and generating plots
│
├── models/
│   ├── saved_weights/               # Trained model checkpoints
│   ├── history/                     # Training logs and loss curves
│   └── model_cards/                 # Symbolic expressions and summary tables
│
├── results/
│   └── figures/                     # Plots, visualizations, and symbolic diagrams
│
├── README.md                        # Project documentation
└── requirements.txt                 # Python dependencies
```

---

## 📊 Case studies

This project presents two structured and complementary case studies:

1. **Case Study 01**: Development and evaluation of a symbolic KAN model for the **Weighted Arithmetic Water Quality Index (WA-WQI)**. Although the underlying aggregation is linear, the model was trained to recover an interpretable expression and was tested on a held-out portion of unseen data to assess predictive performance and symbolic alignment.

2. **Case Study 02**: Application of a symbolic KAN model for the **Weighted Quadratic Mean WQI (WQM-WQI)**, a non-linear formulation. This case study includes a **comparative analysis against baseline regressors** (e.g., MLP, Random Forest) and implements **5-fold temporal cross-validation**, where each fold corresponds to a time window **approximately one month later** than the training data. This setting provides a robust evaluation of the model’s generalization to future, temporally disjoint data.

Both case studies use real IoT sensor data and emphasize **model interpretability**, **symbolic regression**, and **robust generalization**, with special attention to deployment in monitoring scenarios where transparency and temporal transferability are essential.


---

## 🔧 Requirements

Install the required Python dependencies with:

```bash
pip install -r requirements.txt
```

---

## 🚀 Quick start

1. **Run preprocessing (aggregation, filtering, missing value imputation):**

   ```bash
   python scripts/preprocessing.py
   ```

2. **Train the KAN model (script in development):**

   ```bash
   # Not yet implemented — training is currently performed within notebooks
   ```

3. **Evaluate and visualize results:**

   Open and run the following notebooks:

   * `notebooks/case-study_01/...`
   * `notebooks/case-study_02/...`
   * `notebooks/Baseline_wqi_models_temporal_cv.ipynb`

---

## 📈 Outputs

* Symbolic expressions automatically derived from KAN models
* Comparative performance metrics: MAE, RMSE, R²
* Visual analyses: input-output relationships, feature correlations, training dynamics
* Temporal cross-validation results with statistical summaries

---

## 📬 Related preprint

This project is associated with the following preprint:

> **Kolmogorov-Arnold Networks for Interpretable Analysis of Water Quality Time Series Data**
> [https://www.preprints.org/manuscript/202505.1650/v1](https://www.preprints.org/manuscript/202505.1650/v1)

---

## 📚 References

* Liu, S. et al. (2024). *Kolmogorov-Arnold Networks*. arXiv preprint.
* [A simple dataset of water quality on aquaponic fish ponds based on an internet of things measurement device](https://doi.org/10.1016/j.dib.2023.109248)