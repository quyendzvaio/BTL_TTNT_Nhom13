from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    sorted_idx = np.argsort(y_test)
    y_test_sorted = y_test.iloc[sorted_idx]
    y_pred_sorted = y_pred[sorted_idx]

    plt.figure(figsize=(12, 6))
    plt.plot(y_test_sorted.values, label="Actual", marker='o', linestyle='-', markersize=4)
    plt.plot(y_pred_sorted, label="Predicted", marker='x', linestyle='--', markersize=4)
    plt.xlabel("Sample Index (Sorted)")
    plt.ylabel("SalePrice")
    plt.title("Actual vs. Predicted SalePrice")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show() 

    return mse, r2


