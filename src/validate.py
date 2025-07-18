import pandas as pd
from train import load_data

def validate_data(df):
    assert df.isnull().sum().sum() == 0, "Missing values found"
    assert df.shape[1] == 5, "Expected 5 columns (4 features + target)"

if __name__ == "__main__":
    df = load_data()
    validate_data(df)
    print("Data validation passed.")
