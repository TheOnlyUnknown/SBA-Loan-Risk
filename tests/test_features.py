import pandas as pd

from src.features import (
    select_model_features,
    build_preprocessor,
    build_xgboost_model,
    DROP_FOR_MODELING,
)

def test_select_model_features_drops_expected_columns():
    df = pd.DataFrame({col: [1] for col in DROP_FOR_MODELING + ["GrossApproval"]})
    result = select_model_features(df)
    assert list(result.columns) == ["GrossApproval"]

def test_build_preprocessor_handles_mixed_numeric_and_categorical():
    df = pd.DataFrame({"amount": [1.0, 2.0], "category": ["a", "b"]})
    preprocessor = build_preprocessor(df)
    transformed = preprocessor.fit_transform(df)
    assert transformed.shape[0] == 2

def test_build_xgboost_model_fits_and_predicts_probabilities():
    X = pd.DataFrame({
        "amount": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "category": ["a", "b", "a", "b", "a", "b"],
    })
    y = pd.Series([0, 1, 0, 1, 0, 1])
    model = build_xgboost_model(X)
    model.fit(X, y)
    preds = model.predict_proba(X)
    assert preds.shape == (6, 2)
