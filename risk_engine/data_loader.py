import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent


def load_disruption_events() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "disruption_events.csv", parse_dates=["date"])
    return df


def load_industry_exposure() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "industry_exposure.csv")


def load_port_congestion() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "port_congestion.csv", parse_dates=["week_start"])


def load_shipping_rates() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "shipping_rates.csv")
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_trade_flows() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "trade_flows.csv")
