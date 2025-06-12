### 🧪 Additional Case Study: WQI Classification Based on Aquaculture Water Quality Dataset

To further assess the versatility of Kolmogorov–Arnold Networks (KANs), this study introduces a second case focused on a **classification task** using a public dataset designed for **aquaculture water quality assessment**.

📂 **Dataset**:
[Aquaculture – Water Quality Dataset (AWD)](https://data.mendeley.com/datasets/y78ty2g293/1)
🗓️ Published: October 2024
📌 DOI: [10.17632/y78ty2g293.1](https://doi.org/10.17632/y78ty2g293.1)
👥 Contributors: Venkataramana Veeramsetty, Rajeshwarrao Arabelli, T. Bernatin

---

### 🧾 Dataset Summary

The dataset includes **4,300 samples**, labeled into three WQI classes based on aquaculture suitability:

* **Class 0**: Excellent water quality (1,400 samples)
* **Class 1**: Good water quality (1,400 samples)
* **Class 2**: Poor water quality (1,500 samples)

Each instance consists of **14 physicochemical input features**:

* Temperature
* Turbidity
* Dissolved Oxygen (DO)
* Biochemical Oxygen Demand (BOD)
* Carbon Dioxide (CO₂)
* pH
* Alkalinity
* Hardness
* Calcium
* Ammonia
* Nitrite
* Phosphorus
* Hydrogen Sulfide (H₂S)
* Plankton

The WQI class labels were derived based on threshold ranges for aquaculture suitability, including acceptable, desirable, and stress conditions.

---

### 🎯 Objective

This case study aims to test KANs in a **classification setting with nonlinear, high-dimensional input features**, where no known symbolic expression defines the class boundaries. It presents a more challenging and realistic scenario for deploying models in aquaculture systems.

---

### ⚙️ Experimental Setup

* The dataset will be split into **train-validation** and **test** subsets.
* The **test set** will be reserved exclusively for final evaluation.
* Model selection and evaluation will use **k-fold cross-validation** on the training-validation portion to ensure robust ranking.

To benchmark KAN’s performance:

* We will use the **`lazypredict`** package to train and evaluate a broad range of classical machine learning classifiers using default settings.
* The **top 3 ranked classifiers** (based on cross-validated accuracy) will then be **fine-tuned** via hyperparameter optimization.
* The final comparison will include these optimized models and the KAN classifier, all tested on the **held-out test set**.

---

### 🔍 Expected Outcomes

This study will:

* Demonstrate KAN's capacity to handle **classification problems with no symbolic prior knowledge**.
* Evaluate whether symbolic models can remain competitive in **high-dimensional, nonlinear tasks**.
* Contrast KAN results with well-established ML classifiers in a **real-world aquaculture context**.
