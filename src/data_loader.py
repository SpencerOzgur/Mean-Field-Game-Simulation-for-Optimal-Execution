from pathlib import Path
import pandas as pd

DATA_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "hjn_market_params.csv"
)

def load_hjn_market_params():
    return pd.read_csv(DATA_PATH)

def get_hjn_market_params(ticker: str):
    df = load_hjn_market_params()

    row = df.loc[df["ticker"] == ticker]

    if row.empty:
        raise ValueError(f"{ticker} not found in HJN dataset")

    row = row.iloc[0]

    return {
        "S0": float(row["mid_price"]),
        "sigma": float(row["sigma"]),
    }