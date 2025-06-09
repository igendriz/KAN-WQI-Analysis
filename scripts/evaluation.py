import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def compute_metrics(y_true, y_pred):
    """
    Compute R², MAE, and RMSE between predicted and true values.

    Accepts either NumPy arrays or PyTorch tensors as input.
    """
    # Convert to NumPy if input is a PyTorch tensor
    if hasattr(y_true, 'numpy'):
        y_true = y_true.numpy()
    if hasattr(y_pred, 'numpy'):
        y_pred = y_pred.numpy()

    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    return r2, mae, rmse


def evaluate_model_predictions(train_labels, train_preds, val_labels, val_preds, test_labels, test_preds):
    """
    Evaluate regression metrics (R², MAE, RMSE) for training, validation, and test sets.

    Parameters:
        train_labels (torch.Tensor): Ground truth labels for training set
        train_preds  (torch.Tensor): Predicted values for training set

        val_labels (torch.Tensor): Ground truth labels for validation set
        val_preds  (torch.Tensor): Predicted values for validation set

        test_labels (torch.Tensor): Ground truth labels for test set
        test_preds  (torch.Tensor): Predicted values for test set
    """

    # Training set metrics
    r2_train, mae_train, rmse_train = compute_metrics(train_labels, train_preds)
    print("📘 Train Metrics:")
    print(f"R² Score: {r2_train:.3f}")
    print(f"MAE     : {mae_train:.3f}")
    print(f"RMSE    : {rmse_train:.3f}\n")

    # Validation set metrics
    r2_val, mae_val, rmse_val = compute_metrics(val_labels, val_preds)
    print("📙 Validation Metrics:")
    print(f"R² Score: {r2_val:.3f}")
    print(f"MAE     : {mae_val:.3f}")
    print(f"RMSE    : {rmse_val:.3f}\n")

    # Test set metrics
    r2_test, mae_test, rmse_test = compute_metrics(test_labels, test_preds)
    print("📗 Test Metrics:")
    print(f"R² Score: {r2_test:.3f}")
    print(f"MAE     : {mae_test:.3f}")
    print(f"RMSE    : {rmse_test:.3f}\n")