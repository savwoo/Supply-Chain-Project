import pandas as pd

SEVERITY_MAP = {"medium": 0, "high": 1, "extreme": 2}
SEVERITY_LABELS = {0: "Medium", 1: "High", 2: "Extreme"}

_CAT_COLS = ["disruption_type", "region_affected"]
_BOOL_COLS = [
    "is_pandemic", "is_geopolitical", "is_natural",
    "is_financial", "straits_affected", "port_closure",
]
_NUM_COLS = ["duration_days", "year"]


def build_disruption_features(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    X = pd.get_dummies(df[_CAT_COLS + _BOOL_COLS + _NUM_COLS], columns=_CAT_COLS)
    for col in _BOOL_COLS + _NUM_COLS:
        X[col] = X[col].astype(float)
    y_severity = df["severity"].map(SEVERITY_MAP)
    y_freight = df["freight_rate_shock_pct"]
    y_recovery = df["recovery_months"]
    return X, y_severity, y_freight, y_recovery


def encode_prediction_input(
    disruption_type: str,
    region: str,
    duration_days: int,
    year: int,
    is_pandemic: bool,
    is_geopolitical: bool,
    is_natural: bool,
    is_financial: bool,
    straits_affected: bool,
    port_closure: bool,
    feature_columns: list[str],
) -> pd.DataFrame:
    row: dict = {
        "duration_days": float(duration_days),
        "year": float(year),
        "is_pandemic": float(is_pandemic),
        "is_geopolitical": float(is_geopolitical),
        "is_natural": float(is_natural),
        "is_financial": float(is_financial),
        "straits_affected": float(straits_affected),
        "port_closure": float(port_closure),
    }
    for col in feature_columns:
        if col.startswith("disruption_type_"):
            row[col] = 1.0 if disruption_type == col[len("disruption_type_"):] else 0.0
        elif col.startswith("region_affected_"):
            row[col] = 1.0 if region == col[len("region_affected_"):] else 0.0

    df_out = pd.DataFrame([row])
    for col in feature_columns:
        if col not in df_out.columns:
            df_out[col] = 0.0
    return df_out[list(feature_columns)]
