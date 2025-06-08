# Interpretable Water Quality Analysis Using Kolmogorov-Arnold Networks (KAN)

This repository provides a structured and reproducible pipeline for modeling and interpreting water quality time series using **Kolmogorov-Arnold Networks (KAN)**. The project demonstrates how KAN models can be applied to sensor data (pH, temperature, TDS) to predict the **Water Quality Index (WQI)** through interpretable symbolic expressions.

---

## 📌 Project Objectives

* Train and evaluate symbolic regression models using KAN
* Derive human-readable symbolic expressions relating input features to WQI
* Assess model performance using regression metrics (MAE, RMSE, R²)
* Explore model sparsification for sensor reduction and deployment optimization

---

## 📁 Project Structure

```text
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

Install Python dependencies using:

```bash
pip install -r requirements.txt
````

> Note: For LaTeX-rendered plot labels in `matplotlib`, a system-level LaTeX installation (e.g., [MiKTeX](https://miktex.org/) or TeX Live) is recommended.

---

## 🚀 Quick Start

1. **Preprocess the dataset**:

   ```bash
   python scripts/preprocessing.py
   ```

2. **Train the KAN model**:

   ```bash
   python scripts/train_model.py
   ```

3. **Evaluate results**:
   Open and run `notebooks/Evaluate_Models.ipynb` to visualize predictions and inspect symbolic outputs.

---

## 📊 Outputs

* Symbolic expressions automatically extracted from trained KAN models
* Evaluation metrics: MAE, RMSE, R²
* Visualization of input-output relationships and feature influence (e.g., correlation, mutual information heatmaps)

---

## 📜 License

This project is licensed under the **MIT License**. See the `LICENSE` file for details.

---

## 📬 Contact

**Ignacio Sánchez-Gendriz**
For inquiries, please contact via GitHub or institutional email.

---

## 📚 References

* Liu, S. et al. (2024). *Kolmogorov-Arnold Networks*. arXiv preprint.
* Siswanto et al. (2023). *Dataset for water quality monitoring in aquaponics*. Mendeley Data.
