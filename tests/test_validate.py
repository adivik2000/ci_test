from src.validate import validate_data
from src.train import load_data

def test_data_validation():
    df = load_data()
    validate_data(df)
