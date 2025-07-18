from sklearn.metrics import accuracy_score
import joblib
import pandas as pd
from train import preprocess_data, load_data
import json

def evaluate_model():
    df = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(df)
    model = joblib.load("model.joblib")
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    return acc

if __name__ == "__main__":
    accuracy = evaluate_model()
    print(f"Model accuracy: {accuracy:.2f}")
    metrics = {
        "accuracy": round(accuracy, 4)
    }

    # Save both Markdown and JSON for humans and automation
    with open("metrics.json", "w") as f:
        json.dump(metrics, f)

    with open("report.md", "w") as f:
        f.write("## 📊 Evaluation Report\n")
        f.write(f"Model accuracy: {metrics['accuracy']}\n")
