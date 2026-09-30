import json

from src.data_loader import load_and_label
from src.split import time_aware_split
from src.features import select_model_features, build_xgboost_model
from src.threshold import threshold_sweep, best_f1_threshold, report_at_threshold
from src.model_io import save_model

CSV_PATH = "data/raw/foia_7a_fy2020_present.csv"
MODEL_PATH = "models/xgboost_pipeline.joblib"
METADATA_PATH = "models/metadata.json"

def main():
    X, y = load_and_label(CSV_PATH)
    X_train, X_test, y_train, y_test, cutoff = time_aware_split(X, y)

    X_train_m = select_model_features(X_train)
    X_test_m = select_model_features(X_test)

    model = build_xgboost_model(X_train_m)
    model.fit(X_train_m, y_train)

    save_model(model, MODEL_PATH)

    y_proba = model.predict_proba(X_test_m)[:, 1]
    y_true = y_test.values

    sweep = threshold_sweep(y_true, y_proba)
    best = best_f1_threshold(sweep)
    report = report_at_threshold(y_true, y_proba, best["threshold"])

    metadata = {
        "threshold": float(report["threshold"]),
        "precision": float(report["precision"]),
        "recall": float(report["recall"]),
        "f1": float(report["f1"]),
        "lift": float(report["lift"]),
        "split_cutoff": str(cutoff),
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
    }
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    print("Model saved to:", MODEL_PATH)
    print("Metadata saved to:", METADATA_PATH)
    print("Split cutoff:", cutoff)
    print("Train size:", len(X_train), "Test size:", len(X_test))
    print("Chosen threshold:", round(report["threshold"], 4))
    print("Precision:", round(report["precision"], 4))
    print("Recall:", round(report["recall"], 4))
    print("F1:", round(report["f1"], 4))
    print("Lift vs base rate:", round(report["lift"], 2))

if __name__ == "__main__":
    main()
