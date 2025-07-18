from src.evaluate import evaluate_model

def test_accuracy():
    acc = evaluate_model()
    assert acc > 0.8
