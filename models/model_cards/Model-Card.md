# 📄 Model Card: Kolmogorov–Arnold Networks (KAN) for Water Quality Index Prediction

This model card documents two data-driven models developed using **Kolmogorov–Arnold Networks (KANs)** to predict the **Water Quality Index (WQI)** in aquaponic ponds from IoT sensor data. The models include:

* **KAN (Full)**: Uses all three input variables: pH, water temperature, and total dissolved solids (TDS).
* **KAN (Pruned)**: A reduced version that uses only two variables (pH and temperature), discarding TDS based on data-driven feature relevance.

---

## 🧠 Model Overview

Both models are based on the **Kolmogorov–Arnold Network (KAN)** architecture, which uses **edge-based activation functions** instead of traditional node activations. This structure enables the extraction of **interpretable symbolic expressions** that approximate the learned functions in a human-readable form.

The symbolic expressions for both the full and pruned models were **automatically extracted** using built-in functionalities provided by the [`pykan`](https://github.com/KindXiaoming/pykan) package, ensuring consistency between the model’s internal representation and its symbolic approximation.

The **pruned model** was obtained by removing **TDS** as an input variable, based on `pykan` functionalities, which was also confirmed by statistical and mutual information analyses that indicated its **limited relevance** for WQI prediction in the case study of the used dataset. Despite the reduced input dimensionality, the pruned model maintained predictive performance compared to the full model.

All model implementations and training routines were developed using the **`pykan` library**, which is openly documented and maintained on GitHub.

---

## 📊 Dataset Description

* **Source**:
  [A simple dataset of water quality on aquaponic fish ponds based on an internet of things measurement device](https://doi.org/10.1016/j.dib.2023.109248)

* **Sensor Variables**:

  * `pH`: Water pH
  * `water_temp`: Water temperature (°C)
  * `TDS`: Total dissolved solids (ppm)

* **Target Variable**:

  * `WQI`: Computed using a weighted formulation with domain-specific normalization and aggregation procedures.

* **Preprocessing Steps**:

  * Temporal smoothing
  * Outlier removal
  * Interpolation
  * Normalization to standardized WQI scale

* **Splits**:

  * Training: 70%
  * Validation: 15%
  * Test: 15%

> For more details on the **preprocessing steps** and **data split methodology**, refer to the accompanying [preprint manuscript](https://www.preprints.org/manuscript/202505.1650/v1), which describes the full experimental pipeline used in this study.

---

## ⚙️ Model Details

| Model        | Inputs        | Parameters | Symbolic Expression Available | Notes                  |
| ------------ | ------------- | ---------- | ----------------------------- | ---------------------- |
| KAN (Full)   | pH, temp, TDS | ✅          | ✅                             | Original 3-input model |
| KAN (Pruned) | pH, temp      | ✅          | ✅                             | Discards TDS           |

---

## 📈 Performance (Test Set)

| Model             | R²     | MAE    | RMSE   |
| ----------------- | ------ | ------ | ------ |
| KAN (Full)        | 0.9862 | 0.7306 | 1.6330 |
| Symbolic (Full)   | 0.9862 | 0.7322 | 1.6344 |
| KAN (Pruned)      | 0.9899 | 0.4695 | 1.3959 |
| Symbolic (Pruned) | 0.9899 | 0.4706 | 1.3963 |

> 🔍 The pruned model slightly outperforms the full model, while symbolic expressions match their respective model predictions with negligible loss.

---

## ✅ Intended Use

These models are intended for:

* Low-complexity, data-driven WQI prediction in aquaponics and freshwater systems
* Applications where sensor limitations exist and **interpretability is critical**
* Environments requiring symbolic models for transparency, audits, or hybrid physical/data modeling

---

## ⚠️ Limitations and scope of applicability

* The results are contextualized within the specific **WQI formulation**, **normalization strategies**, and **dataset characteristics** used in this study. In particular, the reduced relevance of **TDS** as a predictor may be influenced by the applied normalization or by the dataset’s empirical properties.
* The **feature pruning** (removal of TDS) reflects insights derived from this specific dataset and modeling context, and may not generalize without further validation.
* While the methodology—based on symbolic extraction, model pruning, and statistical evaluation—is broadly applicable, its **generalization to other datasets**, **WQI formulations**, or **IoT sensor configurations** requires re-application and re-validation of the full pipeline.

---

## 🧪 Reproducibility

* Source code: `/scripts/`, `/notebooks/`
* Trained models: `/notebooks/Train_Evaluate_Models.ipnb/`
* Symbolic expressions: included in `/notebooks/Train_Evaluate_Models.ipnb/`
* Performance metrics: saved in `kan_performance_summary.csv`