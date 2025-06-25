# 📄 Model Card: Weighted Quadratic Mean (WQM) Models for Water Quality Index Prediction

This model card documents a set of **baseline models** based on **Weighted Quadratic Mean (WQM)** aggregation techniques for predicting the **Water Quality Index (WQI)** in aquaponic systems using IoT sensor measurements. These models serve as **mathematically grounded baselines**, using pre-defined transformations and parameterized weights to compute WQI from three water quality indicators:

* pH
* Water temperature
* Total Dissolved Solids (TDS)

The WQM approach models WQI as a **nonlinear combination of normalized quality scores** using a quadratic mean (root mean square) aggregation:

$$
\text{WQI}_{\text{WQM}} = \sqrt{ \sum_{i} w_i \cdot q_i^2 }
$$

where $w_i$ are the weights and $q_i$ are the normalized quality scores for each input variable.

---

## 🧠 Model Overview

The WQM models were implemented using a **Kolmogorov–Arnold Network (KAN)** to approximate the nonlinear WQI formulation through data-driven learning. Both **full** and **pruned** KAN models were trained, and their **symbolic expressions** were extracted to ensure interpretability and transparency.

---

## 📊 Dataset Description

* **Source**:
  [A simple dataset of water quality on aquaponic fish ponds based on an internet of things measurement device](https://doi.org/10.1016/j.dib.2023.109248)

* **Sensor Variables**:

  * `pH`: Water pH
  * `water_temp`: Water temperature (°C)
  * `TDS`: Total dissolved solids (ppm)

* **Target Variable**:

  * `WQI`: Computed from quality scores using the WQM method

* **Preprocessing**:

  * Conversion to quality scores ($q_i$)
  * Temporal smoothing
  * Outlier removal and imputation
  * Final WQI standardization

* **Data Splits**:

  * Training: \~67%
  * Validation: \~33%

---

## ⚙️ Model Details

| Model        | Inputs        | Symbolic Expression | Notes                                    |
| ------------ | ------------- | ------------------- | ---------------------------------------- |
| WQM (Full)   | pH, temp, TDS | ✅                   | Includes all 3 quality scores            |
| WQM (Pruned) | pH, temp      | ✅                   | TDS excluded due to limited contribution |

All symbolic forms are **extracted from the trained KAN models** and available in the project notebooks.

---

## 📈 Performance (Validation Set)

Fifteen baseline models were tested using **LazyPredict**. Top-performing models were selected and compared against the symbolic output of the pruned KAN.

| Model                 | R²    | MAE   | RMSE  |
| --------------------- | ----- | ----- | ----- |
| KAN-Symbolic          | 1.000 | 0.169 | 0.201 |
| MLPRegressor          | 0.998 | 0.241 | 0.453 |
| KAN-Model             | 0.985 | 0.585 | 1.250 |
| RandomForestRegressor | 0.882 | 2.071 | 3.464 |
| BaggingRegressor      | 0.880 | 2.115 | 3.499 |

---

## 🧪 Temporal cross-validation (CV)

To assess the temporal robustness of the trained KAN models, we conducted a **sliding-window cross-validation** on a held-out segment of the dataset.

* **Training period**: `2023-01-30 12:00:00` to `2023-02-05 12:00:00`
* **Evaluation period**: `2023-03-01 12:00:00` to `2023-03-08 12:00:00`

We applied **5 overlapping folds** across the final 3.5 days of data, each spanning a 2-day window and shifting forward by 8 hours. This approach ensures a realistic, forward-only prediction setting with no data leakage.

### 🔁 Sliding window folds

| Fold | Start Date & Time   | End Date & Time     |
| ---- | ------------------- | ------------------- |
| 1    | 2023-03-04 00:00:00 | 2023-03-05 23:59:00 |
| 2    | 2023-03-04 08:00:00 | 2023-03-06 07:59:00 |
| 3    | 2023-03-04 16:00:00 | 2023-03-06 15:59:00 |
| 4    | 2023-03-05 00:00:00 | 2023-03-06 23:59:00 |
| 5    | 2023-03-05 08:00:00 | 2023-03-07 07:59:00 |

This **time-aware validation** strategy is critical for real-world scenarios in which models must generalize to future sensor data under dynamic conditions.

### 📊 Cross-validation performance summary (mean ± std)

| Model                 | R²            | MAE           | RMSE          |
| --------------------- | ------------- | ------------- | ------------- |
| KAN-Symbolic          | 0.998 ± 0.001 | 0.449 ± 0.166 | 0.685 ± 0.312 |
| MLPRegressor          | 0.996 ± 0.001 | 0.687 ± 0.226 | 1.157 ± 0.266 |
| KAN-Model             | 0.919 ± 0.100 | 2.764 ± 1.507 | 4.266 ± 2.152 |
| RandomForestRegressor | 0.904 ± 0.059 | 4.141 ± 0.947 | 5.218 ± 1.021 |
| BaggingRegressor      | 0.902 ± 0.059 | 4.205 ± 0.936 | 5.281 ± 1.002 |

---

## ✅ Intended use

* Provide a **transparent baseline** for WQI estimation using interpretable symbolic models
* Enable comparison with complex black-box models
* Serve as a foundation for **hybrid modeling** that combines physical formulas with learned components

---

## ⚠️ Limitations and scope of applicability

These models are tailored for WQI estimation using three specific input variables. However, the same methodology — particularly KAN-based symbolic modeling — can be **retrained and extended** to other nonlinear applications involving environmental or sensor-based time series data.

---

## 📦 Reproducibility

* **Code**: `./scripts/`
* **Notebook**: `/notebooks/case-study_02/Train_Evaluate_Models02.ipynb`
* **Validation metrics**: `models/model_cards/03_baseline_performance_WQM.csv`
* **Temporal CV metrics**: `models/model_cards/03a_cv-baseline_performance_WQM.csv`