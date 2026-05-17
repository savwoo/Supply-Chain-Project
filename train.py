"""Train and save all supply chain risk prediction models.

Usage:
    python train.py
"""
from risk_engine.data_loader import load_disruption_events
from risk_engine.features import build_disruption_features, SEVERITY_LABELS
from risk_engine.model import (
    train_severity_classifier,
    train_freight_regressor,
    train_recovery_regressor,
    save_models,
)


def main() -> None:
    print("Loading disruption events...")
    df = load_disruption_events()
    print(f"  {len(df)} events loaded")

    print("Building features...")
    X, y_severity, y_freight, y_recovery = build_disruption_features(df)
    print(f"  {X.shape[1]} features: {list(X.columns)}")

    print("\nTraining severity classifier (medium / high / extreme)...")
    severity_clf, sev_scores = train_severity_classifier(X, y_severity)
    print(f"  5-fold CV accuracy: {sev_scores.mean():.3f} ± {sev_scores.std():.3f}")
    dist = y_severity.map(SEVERITY_LABELS).value_counts().to_dict()
    print(f"  Class distribution: {dist}")

    print("\nTraining freight rate shock regressor...")
    freight_reg, freight_scores = train_freight_regressor(X, y_freight)
    print(f"  5-fold CV R²: {freight_scores.mean():.3f} ± {freight_scores.std():.3f}")

    print("\nTraining recovery months regressor...")
    recovery_reg, rec_scores = train_recovery_regressor(X, y_recovery)
    print(f"  5-fold CV R²: {rec_scores.mean():.3f} ± {rec_scores.std():.3f}")

    print("\nSaving models to models/...")
    save_models(severity_clf, freight_reg, recovery_reg, list(X.columns))
    print("Done.")


if __name__ == "__main__":
    main()
