import yfinance as yf
import pandas as pd
from datetime import datetime
import os
import logging

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log = logging.getLogger(__name__)

# ── Output folder ──────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(DATA_DIR, exist_ok=True)

# ── Stock list ─────────────────────────────────────────────────────────────────
STOCKS = [
    # 🇺🇸 US Tech
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",
    "NVDA", "TSLA", "NFLX", "AMD", "INTC",

    # 🇮🇳 India NSE — Banking
    "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS",
    "KOTAKBANK.NS", "AXISBANK.NS", "INDUSINDBK.NS",

    # 🇮🇳 India NSE — IT
    "TCS.NS", "INFY.NS", "WIPRO.NS", "HCLTECH.NS",
    "TECHM.NS",

    # 🇮🇳 India NSE — Large Cap
    "RELIANCE.NS", "LT.NS", "ITC.NS", "BHARTIARTL.NS",
    "ASIANPAINT.NS", "MARUTI.NS", "SUNPHARMA.NS",
    "TITAN.NS", "ULTRACEMCO.NS", "ONGC.NS",
    "BAJFINANCE.NS", "POWERGRID.NS", "NTPC.NS",
    "NESTLEIND.NS", "ADANIENT.NS", "COALINDIA.NS",
    "JSWSTEEL.NS", "TATASTEEL.NS", "HINDUNILVR.NS",
    "CIPLA.NS", "DRREDDY.NS", "EICHERMOT.NS",
    "HEROMOTOCO.NS", "BAJAJFINSV.NS", "GRASIM.NS",
    "UPL.NS", "BPCL.NS", "APOLLOHOSP.NS",
    "DIVISLAB.NS",
]

# ── Download ───────────────────────────────────────────────────────────────────
def download_stock(ticker: str) -> pd.DataFrame | None:
    """
    Download 6-month daily OHLCV data for a single ticker.
    Returns a clean long-format DataFrame or None on failure.
    """
    try:
        log.info(f"Downloading {ticker} ...")

        df = yf.download(
            ticker,
            period="6mo",
            interval="1d",
            progress=False,
            auto_adjust=True,   # adjusts for splits & dividends
        )

        if df.empty:
            log.warning(f"No data returned for {ticker} — skipping.")
            return None

     
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # ── Reset index so Date becomes a regular column ──────────────────────
        df.reset_index(inplace=True)

        # ── Keep only the columns we need ────────────────────────────────────
        keep = ["Date", "Open", "High", "Low", "Close", "Volume"]
        df = df[[c for c in keep if c in df.columns]]

        # ── Add metadata ──────────────────────────────────────────────────────
        df["Stock"]        = ticker
        df["Extracted_At"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        log.info(f"{ticker} ✓  ({len(df)} rows)")
        return df

    except Exception as e:
        log.error(f"Failed to download {ticker}: {e}")
        return None


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    log.info("=" * 55)
    log.info("Stock Market Data Pipeline — Extraction")
    log.info("=" * 55)

    all_data   = []
    successful = []
    failed     = []

    for ticker in STOCKS:
        result = download_stock(ticker)
        if result is not None:
            all_data.append(result)
            successful.append(ticker)
        else:
            failed.append(ticker)

    if not all_data:
        log.error("No data downloaded. Check your internet connection.")
        return

    # ── Combine all stocks into one long-format DataFrame ─────────────────────
    master_df = pd.concat(all_data, ignore_index=True)

    # ── Ensure correct data types ─────────────────────────────────────────────
    master_df["Date"] = pd.to_datetime(master_df["Date"])
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        master_df[col] = pd.to_numeric(master_df[col], errors="coerce")

    # ── Sort ──────────────────────────────────────────────────────────────────
    master_df = master_df.sort_values(["Stock", "Date"]).reset_index(drop=True)

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = os.path.join(DATA_DIR, "master_stock_data.csv")
    master_df.to_csv(out_path, index=False)

    # ── Summary ───────────────────────────────────────────────────────────────
    log.info("=" * 55)
    log.info(f"Master dataset saved → {out_path}")
    log.info(f"Shape          : {master_df.shape[0]:,} rows × {master_df.shape[1]} columns")
    log.info(f"Stocks         : {master_df['Stock'].nunique()} unique tickers")
    log.info(f"Date range     : {master_df['Date'].min().date()} → {master_df['Date'].max().date()}")
    log.info(f"Successful     : {len(successful)}")
    log.info(f"Failed         : {len(failed)} {failed if failed else ''}")
    log.info("=" * 55)

    print("\nPreview:")
    print(master_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
