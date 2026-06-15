import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from TrainModel import train_models

# Train once for testing
model, scaler, features = train_models()

def test_model_returns_prediction():
    sample = np.array([[30, 60000, 5000, 20000, 10000]])

    scaled = scaler.transform(sample)
    pred = model.predict(scaled)

    assert pred is not None
    assert len(pred) == 1


def test_prediction_is_positive():
    sample = np.array([[40, 120000, 2000, 50000, 30000]])

    scaled = scaler.transform(sample)
    pred = model.predict(scaled)

    assert pred[0] > 0


def test_higher_income_increases_net_worth():
    low = scaler.transform([[25, 30000, 5000, 10000, 5000]])
    high = scaler.transform([[25, 100000, 5000, 10000, 5000]])

    low_pred = model.predict(low)[0]
    high_pred = model.predict(high)[0]

    assert high_pred > low_pred