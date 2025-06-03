import pandas as pd

def load_data(path):
    df = pd.read_csv(path)
    df.drop(columns=["Id"], inplace=True)
    df = df.corr(numeric_only=True)["SalePrice"].sort_values(ascending=False)
    df = df[:11].index
    X = df.drop("SalePrice", axis=1)
    y = df["SalePrice"]
    return X, y

