from src.data_loader import load_data
from src.preprocessing import create_preprocessor
from src.train import train_model
from src.evaluate import evaluate_model
from src.utils import save_model, save_metrics

DATA_PATH = "data/train.csv"
MODEL_PATH = "model/model.joblib"
METRIC_PATH = "outputs/metrics.txt"

def main():
    X, y = load_data(DATA_PATH)
    preprocessor = create_preprocessor(X)
    grid, X_test, y_test = train_model(X, y, preprocessor)

    best_model = grid.best_estimator_
    mse, r2 = evaluate_model(best_model, X_test, y_test)
    
    save_model(best_model, MODEL_PATH)
    save_metrics(grid.best_params_, mse, r2, METRIC_PATH)

    print("Best Parameters:", grid.best_params_)
    print("MSE:", mse)
    print("R2:", r2)

if __name__ == "__main__":
    main()
