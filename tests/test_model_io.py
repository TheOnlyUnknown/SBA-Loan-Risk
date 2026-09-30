import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.model_io import save_model, load_model

def test_save_and_load_model_round_trip_predictions(tmp_path):
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]})
    y = pd.Series([0, 0, 1, 1])
    model = Pipeline([("clf", LogisticRegression())])
    model.fit(X, y)
    path = tmp_path / "model.joblib"
    save_model(model, str(path))
    loaded = load_model(str(path))
    original_preds = model.predict_proba(X)
    loaded_preds = loaded.predict_proba(X)
    assert np.array_equal(original_preds, loaded_preds)
