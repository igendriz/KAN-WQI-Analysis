# ✅ Recommended PyKAN Installation Steps (Stable and Reproducible Setup)

To ensure a stable and compatible environment for using [PyKAN](https://github.com/KindXiaoming/pykan), follow the steps below. These practices are based on tested dependencies, reproducibility, and the authors’ own recommendations.

---

## ⚠ Why Python Version Matters

- `pykan` and its dependencies (e.g., `numpy==1.24.4`) are **not compatible with Python 3.12+**
- The official repository explicitly recommends **Python 3.9.7**
- Errors such as `pkgutil.ImpImporter` or `build_meta` failures are caused by using newer Python versions

---

## 📌 Pre-requisites

- **Python version:** 3.9.7 (strictly recommended)
- **Virtual environment manager:** `venv` or `conda`
- **Operating System:** Windows, Linux, or macOS

---

## 🧰 1. Install Python 3.9.7

If not already installed, download from:  
👉 https://www.python.org/downloads/release/python-397/

### Example (confirmed on this system):
```text
C:\Users\soyig\AppData\Local\Programs\Python\Python397\python.exe
```

During installation:
- ✅ Check **“Add Python to PATH”** (optional)
- ✅ Enable pip and venv modules
- 📁 Install to a fixed directory like above for reuse

---

## 🧪 2. Create and activate the virtual environment

### Windows (example using system-installed Python 3.9.7):
```powershell
C:\Users\soyig\AppData\Local\Programs\Python\Python397\python.exe -m venv venv
.\venv\Scripts\activate
```

### macOS/Linux:
```bash
python3.9 -m venv venv
source venv/bin/activate
```

---

## 🔗 3. Install PyKAN 

### Installation via PyPI
```bash
pip install pykan
```

### Install PyKAN from GitHub
```bash
pip install git+https://github.com/KindXiaoming/pykan.git
```

> This fetches the latest development version directly from the repository.

---

## 📄 4. Install required dependencies

Install the libraries used in examples, training scripts, and data handling:

```bash
pip install -r requirements.txt
```

Content of `requirements.txt`:
```txt
matplotlib==3.6.2
numpy==1.24.4
scikit_learn==1.1.3
setuptools==65.5.0
sympy==1.11.1
torch==2.2.2
tqdm==4.66.2
pandas==2.0.1
seaborn
pyyaml
```

---

## ✅ Final Notes

- This setup is intended for **using PyKAN**, not modifying its source code.
- All packages, including PyKAN, will be installed **inside the virtual environment** under `venv/Lib/site-packages/`.
- **Do not install PyKAN globally** — this avoids dependency conflicts and versioning issues across projects.
- For VSCode users, make sure to select the correct interpreter from `.venv\Scripts\python.exe` to ensure code runs inside the right environment.

