import pandas as pd

def load_data(path):
    df = pd.read_csv(path)
    df.drop(columns=["Id"], inplace=True)
    X = df.drop("SalePrice", axis=1)
    y = df["SalePrice"]
    return X, y
