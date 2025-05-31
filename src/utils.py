import joblib

def save_model(model, path):
    joblib.dump(model, path)

def save_metrics(params, mse, r2, path):
    with open(path, "w") as f:
        f.write(f"Best Parameters: {params}\n")
        f.write(f"MSE: {mse}\n")
        f.write(f"R2: {r2}\n")
