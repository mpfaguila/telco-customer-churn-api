from pathlib import Path
import json
import joblib
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_curve, roc_auc_score
from xgboost import XGBClassifier

from src.modeling import load_data, FEATURES, TARGET, build_preprocessor

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "Telco_customer_churn.xlsx"
ARTIFACT_PATH = ROOT / "artifacts" / "telecom_churn_model.joblib"
METADATA_PATH = ROOT / "artifacts" / "model_metadata.json"


def ks_cutoff(y_true, proba):
    """Retorna o cutoff que maximiza TPR - FPR (máximo KS/Youden)."""
    fpr, tpr, thresholds = roc_curve(y_true, proba)
    idx = int(np.argmax(tpr - fpr))
    return float(thresholds[idx]), float((tpr - fpr)[idx])


def main():
    df = load_data(DATA_PATH)
    X = df[FEATURES].copy()
    y = df[TARGET].astype(int).copy()

    # Mantemos 20% isolado como teste final, igual ao notebook.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Configuração selecionada no notebook após Grid Search.
    model = XGBClassifier(
        n_estimators=120,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        n_jobs=1,
        eval_metric="logloss",
    )

    pipeline = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("model", model),
    ])

    # O cutoff é escolhido apenas com previsões OOF no treino.
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    oof_proba = cross_val_predict(
        pipeline, X_train, y_train, cv=cv, method="predict_proba", n_jobs=1
    )[:, 1]
    cutoff, train_oof_ks = ks_cutoff(y_train, oof_proba)

    # Treino final em todo o conjunto de treino.
    pipeline.fit(X_train, y_train)
    test_proba = pipeline.predict_proba(X_test)[:, 1]

    artifact = {
        "pipeline": pipeline,
        "cutoff": cutoff,
        "features": FEATURES,
        "model_name": "XGBoost",
        "model_version": "1.0",
    }

    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, ARTIFACT_PATH)

    metadata = {
        "model_name": "XGBoost",
        "model_version": "1.0",
        "cutoff": cutoff,
        "train_oof_ks": train_oof_ks,
        "test_auc": float(roc_auc_score(y_test, test_proba)),
        "records": int(len(df)),
        "features": len(FEATURES),
        "hyperparameters": {
            "n_estimators": 120,
            "learning_rate": 0.05,
            "max_depth": 3,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
        },
    }
    METADATA_PATH.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
