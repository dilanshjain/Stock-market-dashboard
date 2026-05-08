import pandas as pd
import numpy as np
import os
import logging

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR  = os.path.join(os.path.dirname(__file__), "..", "data")
IN_PATH   = os.path.join(DATA_DIR, "master_stock_data.csv")
OUT_PATH  = os.path.join(DATA_DIR, "transformed_stock_data.csv")

TRADING_DAYS_PER_YEAR = 252  # standard annualisation factor

# ── Helper: group-level transform shortcut ────────────────────────────────────
def gapply(df, col, func):
    return df.groupby("Stock")[col].transform(func)


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    log.info("=" * 55)
    log.info("Stock Market Data Pipeline — Transformation")
    log.info("=" * 55)

    # ── 1. Load ───────────────────────────────────────────────────────────────
    log.info(f"Loading → {IN_PATH}")
    df = pd.read_csv(IN_PATH)
    log.info(f"Raw shape: {df.shape}")

    # ── 2. Data types ─────────────────────────────────────────────────────────
    df["Date"] = pd.to_datetime(df["Date"])

    numeric_cols = ["Open", "High", "Low", "Close", "Volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # ── 3. Sort ───────────────────────────────────────────────────────────────
    df = df.sort_values(["Stock", "Date"]).reset_index(drop=True)

    # ── 4. Handle missing prices with Forward Fill ────────────────────────────
    # FIX: fillna(0) is WRONG for prices (a ₹0 stock price makes no sense).
    # Forward fill carries the last known price forward for market-closed days.
    df[numeric_cols] = df.groupby("Stock")[numeric_cols].ffill()

    # ── 5. Feature Engineering ────────────────────────────────────────────────
    log.info("Calculating features ...")

    # Daily Return %
    df["Daily_Return_%"] = gapply(df, "Close", lambda x: x.pct_change() * 100)

    # 7-Day Moving Average
    df["MA_7"] = gapply(df, "Close", lambda x: x.rolling(7, min_periods=1).mean())

    # 30-Day Moving Average
    df["MA_30"] = gapply(df, "Close", lambda x: x.rolling(30, min_periods=1).mean())

    # Price vs MA_7 signal: above (+1) or below (-1) the 7-day average
    df["MA7_Signal"] = np.where(df["Close"] > df["MA_7"], "Above", "Below")

    # Annualised Volatility (7-day rolling std × √252)
    # Industry standard: expressing daily volatility as annual percentage
    df["Volatility_Ann_%"] = gapply(
        df, "Daily_Return_%",
        lambda x: x.rolling(7, min_periods=2).std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    )

    # Daily Price Range % = (High - Low) / Low × 100
    # Shows intraday price swing — useful for detecting high-activity days
    df["Daily_Range_%"] = ((df["High"] - df["Low"]) / df["Low"]) * 100

    # Cumulative Return % from the first date in the dataset
    df["Cumulative_Return_%"] = gapply(
        df, "Close",
        lambda x: (x / x.iloc[0] - 1) * 100
    )

    # Volume Change % — spike in volume often precedes price movement
    df["Volume_Change_%"] = gapply(df, "Volume", lambda x: x.pct_change() * 100)

    # Above/Below 30-day MA signal
    df["Trend"] = np.where(df["Close"] > df["MA_30"], "Uptrend", "Downtrend")

    # ── 6. Add Market label (US or India) ────────────────────────────────────
    df["Market"] = np.where(df["Stock"].str.endswith(".NS"), "India NSE", "US")

    # ── 7. Round all float columns to 4 decimal places ───────────────────────
    float_cols = df.select_dtypes(include="float64").columns
    df[float_cols] = df[float_cols].round(4)

    # ── 8. Final column order ─────────────────────────────────────────────────
    col_order = [
        "Date", "Stock", "Market",
        "Open", "High", "Low", "Close", "Volume",
        "Daily_Return_%", "Cumulative_Return_%",
        "MA_7", "MA_30", "MA7_Signal", "Trend",
        "Volatility_Ann_%", "Daily_Range_%",
        "Volume_Change_%", "Extracted_At"
    ]
    # Only include columns that actually exist (safety check)
    col_order = [c for c in col_order if c in df.columns]
    df = df[col_order]

    # ── 9. Save ───────────────────────────────────────────────────────────────
    df.to_csv(OUT_PATH, index=False)

    # ── 10. Summary ───────────────────────────────────────────────────────────
    log.info("=" * 55)
    log.info(f"Transformed dataset saved → {OUT_PATH}")
    log.info(f"Shape          : {df.shape[0]:,} rows × {df.shape[1]} columns")
    log.info(f"Stocks         : {df['Stock'].nunique()} unique tickers")
    log.info(f"US stocks      : {df[df['Market']=='US']['Stock'].nunique()}")
    log.info(f"India stocks   : {df[df['Market']=='India NSE']['Stock'].nunique()}")
    log.info(f"Date range     : {df['Date'].min().date()} → {df['Date'].max().date()}")
    log.info(f"Features added : Daily_Return, MA_7, MA_30, Signals, Volatility,")
    log.info(f"                 Cumulative_Return, Daily_Range, Volume_Change, Trend")
    log.info("=" * 55)

    # ── 11. Quick insights preview ────────────────────────────────────────────
    print("\n── Top 5 by Cumulative Return % ──")
    top = (
        df.groupby("Stock")["Cumulative_Return_%"]
        .last()
        .sort_values(ascending=False)
        .head(5)
    )
    print(top.to_string())

    print("\n── Top 5 Most Volatile (Annualised) ──")
    vol = (
        df.groupby("Stock")["Volatility_Ann_%"]
        .mean()
        .sort_values(ascending=False)
        .head(5)
    )
    print(vol.to_string())

    print("\n── Sample rows ──")
    print(df.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
