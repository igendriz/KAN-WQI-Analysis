# Interpretable Water Quality Analysis Using Kolmogorov-Arnold Networks (KAN)

This repository contains a structured and reproducible pipeline for analyzing water quality time series data using **Kolmogorov-Arnold Networks (KAN)**. The goal is to demonstrate how KAN can provide interpretable, data-driven models for predicting the **Water Quality Index (WQI)** based on sensor measurements such as pH, total dissolved solids (TDS), and temperature.

---

## 📌 Project Objectives

* Train and evaluate KAN models on IoT-based water quality data
* Derive symbolic expressions representing learned input-output relationships
* Assess performance using regression metrics (MAE, RMSE, R²)
* Explore feature sparsification and model pruning for sensor optimization

---

## 📁 Project Structure

```
/KAN-WQI-Analysis/
├── data/
│   ├── raw/                   # Original data files (CSV)
│   └── processed/            # Cleaned, interpolated, and smoothed CSVs
│
├── notebooks/
│   ├── EDA_Preprocessing.ipynb    # Visual exploration and preprocessing
│   └── Evaluate_Models.ipynb      # Performance metrics and result plots
│
├── scripts/
│   ├── preprocessing.py      # Data cleaning pipeline
│   ├── train_model.py        # Training routines for KAN models
│   └── evaluation.py         # Regression metrics and utility functions
│
├── models/
│   ├── saved_weights/        # Trained model weights
│   ├── history/              # Training/validation loss curves
│   └── model_cards/          # Descriptive model documentation
│
├── results/
│   └── figures/              # Heatmaps, architecture, prediction plots
│
├── README.md                 # Project description and usage
└── requirements.txt          # Dependencies
```

---

## 🔧 Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
```

---

## 🚀 Quick Start

1. **Preprocess Data**

   ```bash
   python scripts/preprocessing.py
   ```

2. **Train Models**

   ```bash
   python scripts/train_model.py
   ```

3. **Evaluate Models** Run `Evaluate_Models.ipynb` to visualize predictions and compare symbolic outputs.

---

## 📊 Outputs

* Symbolic equations extracted from trained KAN models
* Comparative performance metrics (MAE, RMSE, R²)
* Exploratory heatmaps using correlation and mutual information

---

## 📜 License

This project is open-source and available under the MIT License.

---

## 📬 Contact

Ignacio Sánchez-Gendriz
For inquiries, please contact via GitHub or institutional email.

---

## 📚 References

* Liu, S. et al. (2024). *Kolmogorov-Arnold Networks*.
* Siswanto et al. (2023). *Dataset for water quality monitoring in aquaponics*. Mendeley Data.
