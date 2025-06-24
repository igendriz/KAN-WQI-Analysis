## ✅ Objective

If you want to:

1. **Train** a KAN model with 3 inputs.
2. **Extract symbolic expression** (without altering the trained model).
3. **Prune the model** to reduce the number of inputs (e.g., from 3 → 2).
4. **Extract symbolic expression again** from the pruned model.

---

## 🔁 General Strategy: Use `auto_save=True` and rely on versioned checkpoints.

Each step that alters the model (training, pruning, etc.) will **automatically save** a new version in `./model/`.

We’ll use `rewind()` to restore a version when needed and keep things clean.

---

## 🔢 Step-by-step Implementation

---

### 🔹 **Step 1 — Train a model with 3 inputs**

```python
from kan import create_dataset, KAN
import torch

# Set up
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = create_dataset(f, n_var=3, device=device)  # 3-input function

# Initialize model with auto_save ON
model = KAN(width=[3, 10, 1], grid=5, k=3, seed=1, auto_save=True, device=device)

# Train the model
model.fit(dataset, opt="LBFGS", steps=100)

# Result: model saved as version 0.1 (original trained model)
```

---

### 🔹 **Step 2 — Extract symbolic expression (non-destructive)**

```python
# Rewind to the trained version 0.1 (if needed)
model_symb = model.rewind('0.1')  # model_symb is version 1.1

# Activate symbolic extraction
model_symb.save_act = True
_ = model_symb(dataset.train_input)  # Forward pass to populate internal states

# Extract symbolic formula
model_symb.auto_symbolic(lib=['x', 'x^2', 'log', 'sin', 'sqrt'])
formula = model_symb.symbolic_formula()[0][0]
```

✅ The original model remains unchanged because you extracted on a rewinded instance.

---

### 🔹 **Step 3 — Prune model to reduce inputs (e.g., to 2 inputs)**

```python
# Rewind again to clean trained model before pruning
model_pruned = model.rewind('0.1')  # model_pruned is version 2.1

# Apply pruning
model_pruned = model_pruned.prune()  # Automatically saved as version 2.2

# (Optional) Evaluate which inputs were removed:
# ...
```

Now you have a new model that may depend on only 2 inputs.

---

### 🔹 **Step 4 — Extract symbolic expression from pruned model**

```python
model_pruned.save_act = True
_ = model_pruned(dataset.train_input)  # Forward pass to fill activations

model_pruned.auto_symbolic(lib=['x', 'x^2', 'log', 'sin', 'sqrt'])
pruned_formula = model_pruned.symbolic_formula()[0][0]
```

---

## ✅ Summary Table

| Step | Description                         | Model Version |
| ---- | ----------------------------------- | ------------- |
| 1    | Train base model with 3 inputs      | `0.1`         |
| 2    | Rewind + symbolic extraction (safe) | `1.1`         |
| 3    | Rewind + prune to 2 inputs          | `2.2`         |
| 4    | Symbolic extraction of pruned model | (on `2.2`)    |

---

## 🧠 Final Notes

* **Avoid using `.copy()`** unless you know internal states are correctly handled.
* **Use `rewind()`** before symbolic/pruning steps.
* Use **`.symbolic_formula()` only after `.auto_symbolic()`** and a forward pass.
* At any point, you can load a specific version for inference using:

  ```python
  model = KAN.loadckpt('./model/0.1')
  ```

