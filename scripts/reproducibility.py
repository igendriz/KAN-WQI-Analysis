# scripts/reproducibility.py  (CPU-friendly flavour)

import os, random
from typing import Optional

import numpy as np
import torch


def set_global_seed(
    seed: int = 42,
    *,
    deterministic: bool = True,
    single_thread_blas: bool = False,
) -> None:
    """
    Fully determinise a PyTorch + NumPy workflow on **CPU or GPU**.

    Parameters
    ----------
    seed : int, default=42
        Seed reused for Python ``random``, NumPy and PyTorch.
    deterministic : bool, default=True
        Call ``torch.use_deterministic_algorithms(True)`` and, when on
        GPU, set the usual cuDNN flags.  Harmless on CPU.
    single_thread_blas : bool, default=False
        If *True* sets ``OMP_NUM_THREADS = MKL_NUM_THREADS = 1`` so
        OpenMP / MKL reductions happen in a fixed order.  Useful when
        you need bit-for-bit identical results across runs.

    Notes
    -----
    * With blas threads left >1 you may still see differences on the
      order of 1e-7 in large matrix multiplications—numerically safe.
    """

    # ---------------------------------------------------------------- Python RNG
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    # ---------------------------------------------------------------- NumPy RNG
    np.random.seed(seed)

    # ---------------------------------------------------------------- PyTorch RNG
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # ---------------------------------------------------------------- Deterministic kernels
    if deterministic:
        torch.use_deterministic_algorithms(True)
        # cuDNN flags have no effect on CPU but are harmless
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # ---------------------------------------------------------------- BLAS threads (CPU only)
    if single_thread_blas:
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
