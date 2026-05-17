import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold

MODEL_DIR = Path(__file__).parent.parent / "models"


def train_severity_classifier(
    X: pd.DataFrame, y: pd.Series
) -> tuple[RandomForestClassifier, np.ndarray]:
    clf = RandomForestClassifier(
        n_estimators=300, max_depth=5, min_samples_leaf=2,
        random_state=42, class_weight="balanced",
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
    clf.fit(X, y)
    return clf, scores


def train_freight_regressor(
    X: pd.DataFrame, y: pd.Series
) -> tuple[RandomForestRegressor, np.ndarray]:
    mask = y.notna()
    X_c, y_c = X[mask], y[mask]
    reg = RandomForestRegressor(
        n_estimators=300, max_depth=5, min_samples_leaf=2, random_state=42,
    )
    scores = cross_val_score(reg, X_c, y_c, cv=KFold(5, shuffle=True, random_state=42), scoring="r2")
    reg.fit(X_c, y_c)
    return reg, scores


def train_recovery_regressor(
    X: pd.DataFrame, y: pd.Series
) -> tuple[RandomForestRegressor, np.ndarray]:
    mask = y.notna()
    X_c, y_c = X[mask], y[mask]
    reg = RandomForestRegressor(
        n_estimators=300, max_depth=5, min_samples_leaf=2, random_state=42,
    )
    scores = cross_val_score(reg, X_c, y_c, cv=KFold(5, shuffle=True, random_state=42), scoring="r2")
    reg.fit(X_c, y_c)
    return reg, scores


def save_models(
    severity_clf: RandomForestClassifier,
    freight_reg: RandomForestRegressor,
    recovery_reg: RandomForestRegressor,
    feature_columns: list[str],
) -> None:
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(severity_clf, MODEL_DIR / "severity_classifier.pkl")
    joblib.dump(freight_reg, MODEL_DIR / "freight_regressor.pkl")
    joblib.dump(recovery_reg, MODEL_DIR / "recovery_regressor.pkl")
    joblib.dump(feature_columns, MODEL_DIR / "feature_columns.pkl")


def load_models() -> tuple:
    severity_clf = joblib.load(MODEL_DIR / "severity_classifier.pkl")
    freight_reg = joblib.load(MODEL_DIR / "freight_regressor.pkl")
    recovery_reg = joblib.load(MODEL_DIR / "recovery_regressor.pkl")
    feature_columns = joblib.load(MODEL_DIR / "feature_columns.pkl")
    return severity_clf, freight_reg, recovery_reg, feature_columns
