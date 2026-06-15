import pandas as pd
import numpy as np


def add_technical_indicators(df):
    """
    Hitung indikator teknikal + lagged features per-ticker.
    Target: harga Close hari berikutnya (regresi, bukan klasifikasi).

    Catatan: pandas 2.x groupby().apply() membuang kolom groupby.
    Solusi: loop eksplisit + pd.concat().
    """
    if "Ticker" in df.columns and df["Ticker"].nunique() > 1:
        parts = []
        for _, group in df.groupby("Ticker", sort=False):
            parts.append(_calc_indicators(group))
        df = pd.concat(parts).reset_index(drop=True)
    else:
        df = _calc_indicators(df)

    df = df.dropna().reset_index(drop=True)

    features = [
        # Lagged prices — kunci utama R² tinggi (autokorelasi harga)
        "Close_Lag_1",
        "Close_Lag_2",
        "Close_Lag_5",
        # Momentum
        "Momentum_5",
        "Momentum_20",
        # Moving averages
        "MA_5",
        "MA_20",
        "MA_50",
        # Volatility & oscillator
        "Volatility_20",
        "RSI",
        # MACD
        "MACD",
        "MACD_Hist",
        # Returns
        "Return_1d",
        "Return_5d",
        "Return_20d",
        # Bollinger
        "BB_Upper",
        "BB_Lower",
        # Intraday
        "High_Low_Range",
        # Volume
        "Volume_Change",
        # Temporal
        "Month",
        "DayOfWeek",
    ]
    features = [f for f in features if f in df.columns]

    X             = df[features].values.astype(np.float64)
    y_price       = df["Target_Price"].values.astype(np.float64)  # Close besok (regresi)
    current_close = df["Close"].values.astype(np.float64)         # Close hari ini
    # Label biner: 1=Buy (harga naik), 0=Sell (harga turun) — untuk Naive Bayes
    y_direction   = (y_price > current_close).astype(np.int32)

    buy_pct = y_direction.mean() * 100
    print(f"  Fitur aktif ({len(features)}): {features}")
    print(f"  Target regresi  : harga Close besok  (${y_price.min():.2f} – ${y_price.max():.2f})")
    print(f"  Target klasifikasi: Buy={buy_pct:.1f}%  Sell={100-buy_pct:.1f}%")

    return X, y_price, y_direction, features, current_close


def _calc_indicators(df):
    close  = df["Close"]
    volume = df["Volume"] if "Volume" in df.columns else None
    high   = df["High"]   if "High"   in df.columns else None
    low    = df["Low"]    if "Low"    in df.columns else None

    df = df.copy()

    # ── Lagged close (paling penting untuk R² tinggi) ─────────────
    df["Close_Lag_1"] = close.shift(1)
    df["Close_Lag_2"] = close.shift(2)
    df["Close_Lag_5"] = close.shift(5)

    # ── Moving averages ───────────────────────────────────────────
    df["MA_5"]  = close.rolling(5,  min_periods=1).mean()
    df["MA_20"] = close.rolling(20, min_periods=1).mean()
    df["MA_50"] = close.rolling(50, min_periods=1).mean()

    # ── Momentum ──────────────────────────────────────────────────
    df["Momentum_5"]  = close - close.shift(5)
    df["Momentum_20"] = close - close.shift(20)

    # ── Volatility ────────────────────────────────────────────────
    df["Volatility_20"] = close.rolling(20, min_periods=1).std()

    # ── EMA & MACD ────────────────────────────────────────────────
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["MACD"]      = ema12 - ema26
    macd_signal     = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - macd_signal

    # ── Bollinger Bands ───────────────────────────────────────────
    bb_std          = close.rolling(20, min_periods=1).std()
    df["BB_Upper"]  = df["MA_20"] + 2 * bb_std
    df["BB_Lower"]  = df["MA_20"] - 2 * bb_std

    # ── RSI ───────────────────────────────────────────────────────
    delta           = close.diff()
    gain            = delta.clip(lower=0).rolling(14, min_periods=1).mean()
    loss            = (-delta.clip(upper=0)).rolling(14, min_periods=1).mean()
    rs              = gain / loss.replace(0, np.nan)
    df["RSI"]       = 100 - (100 / (1 + rs))

    # ── Returns ───────────────────────────────────────────────────
    df["Return_1d"]  = close.pct_change().clip(-0.5, 0.5)
    df["Return_5d"]  = ((close / (close.shift(5)  + 1e-10)) - 1) * 100
    df["Return_20d"] = ((close / (close.shift(20) + 1e-10)) - 1) * 100

    # ── Intraday range ────────────────────────────────────────────
    if high is not None and low is not None:
        df["High_Low_Range"] = (high - low) / close

    # ── Volume ────────────────────────────────────────────────────
    if volume is not None:
        df["Volume_Change"] = volume.pct_change().fillna(0).clip(-5, 5)

    # ── Temporal features ─────────────────────────────────────────
    if "Date" in df.columns:
        dates = pd.to_datetime(df["Date"])
        df["Month"]     = dates.dt.month.astype(float)
        df["DayOfWeek"] = dates.dt.dayofweek.astype(float)

    # ── TARGET: harga Close besok (regresi) ───────────────────────
    df["Target_Price"] = close.shift(-1)

    return df
