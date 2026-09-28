import shap
import numpy as np
import pandas as pd

def compute_shap_values(model, X):
    preprocessor = model.named_steps["preprocess"]
    estimator = model.named_steps["model"]

    X_transformed = preprocessor.transform(X)
    feature_names = preprocessor.get_feature_names_out()

    explainer = shap.TreeExplainer(estimator)
    shap_values = explainer.shap_values(X_transformed)

    return shap_values, feature_names, X_transformed

def global_feature_importance(shap_values, feature_names, top_n=20):
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False)
    return importance.head(top_n).reset_index(drop=True)

def feature_direction(shap_values, feature_names, X_transformed, feature):
    idx = list(feature_names).index(feature)
    values = X_transformed[:, idx]
    if hasattr(values, "toarray"):
        values = values.toarray().ravel()
    contributions = shap_values[:, idx]
    correlation = np.corrcoef(values, contributions)[0, 1]
    return correlation
