import numpy as np

# ------------------------------------------------------------------
# Symbolic formula 
# ------------------------------------------------------------------
predict_symb_01 = lambda x: (
    11.7642 * np.sqrt(
        0.0009 * (-x[:, 0] - 0.9895) ** 2
        + 0.0073 * (-0.5975 * x[:, 1] - 1) ** 2
        + 1
    )
    - 4.0678
)

# ------------------------------------------------------------------
# Lightweight wrapper
# ------------------------------------------------------------------
class SymbolicKANRegressor:
    """
    Prediction-only wrapper around a fixed symbolic KAN expression.

    Parameters
    ----------
    func : callable
        A vectorised function f(X) -> y_hat that expects a NumPy
        array of shape (n_samples, n_features).

    Notes
    -----
    * No `fit` method – the model is already “trained”.
    * Implements `__call__`, so you can use `model(X)` or
      `model.predict(X)` interchangeably.
    """

    def __init__(self, func=predict_symb_01):
        self.func = func

    # ------------------------------------------------- main API
    def predict(self, X):
        """Return predictions for input matrix `X`."""
        X = self._validate_X(X)
        return self.func(X)

    # Alias so you can do:  y = model(X)
    __call__ = predict

    # ------------------------------------------------- helpers
    @staticmethod
    def _validate_X(X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:                # allow a single sample row-vector
            X = X.reshape(1, -1)
        if X.ndim != 2:
            raise ValueError("X must be 2-D (n_samples, n_features).")
        return X
