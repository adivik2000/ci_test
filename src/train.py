from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import joblib
import pandas as pd

def load_data():
    df = pd.read_csv("../data/iris.csv")
    return df

def preprocess_data(df):
    X = df.drop(columns='species')
    y = df['species']
    return train_test_split(X, y, test_size=0.4, random_state=42)

def train_model(X_train, y_train):
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    return clf

if __name__ == "__main__":
    df = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(df)
    model = train_model(X_train, y_train)
    joblib.dump(model, "model.joblib")
