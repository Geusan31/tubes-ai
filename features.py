import pandas as pd
import numpy as np


def add_technical_indicators(df):
    """
    Hitung indikator teknikal per-ticker agar tidak cross-contaminate
    antar saham yang berbeda.

    Catatan: pandas 2.x groupby().apply() membuang kolom groupby dari hasil.
    Solusi: loop eksplisit + pd.concat() untuk mempertahankan kolom Ticker.
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
        "RSI",
        "MACD",
        "MACD_Hist",       # MACD histogram (MACD - signal line)
        "BB_Upper",
        "BB_Lower",
        "SMA_20",
        "EMA_12",
        "Close",
        "Volume_Change",
        "Return_1d",       # 1-day price return
        "High_Low_Range",  # (High-Low)/Close — intraday volatility
    ]
    # Hanya pakai kolom yang benar-benar ada
    features = [f for f in features if f in df.columns]

    X = df[features].values.astype(np.float64)
    y = df["Target"].values.astype(np.int32)

    unique, counts = np.unique(y, return_counts=True)
    print(f"  Fitur aktif ({len(features)}): {features}")
    print("  Distribusi label setelah feature engineering:")
    for u, c in zip(unique, counts):
        label = "Buy (1)" if u == 1 else "Sell (0)"
        print(f"    {label}: {c:,} ({c/len(y)*100:.1f}%)")

    return X, y, features


def _calc_indicators(df):
    close  = df["Close"]
    volume = df["Volume"] if "Volume" in df.columns else None
    high   = df["High"]   if "High"   in df.columns else None
    low    = df["Low"]    if "Low"    in df.columns else None

    df = df.copy()

    # SMA & EMA
    df["SMA_20"] = close.rolling(window=20).mean()
    df["EMA_12"] = close.ewm(span=12, adjust=False).mean()
    df["EMA_26"] = close.ewm(span=26, adjust=False).mean()

    # MACD + histogram
    df["MACD"]        = df["EMA_12"] - df["EMA_26"]
    macd_signal       = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"]   = df["MACD"] - macd_signal

    # Bollinger Bands
    bb_std           = close.rolling(window=20).std()
    df["BB_Upper"]   = df["SMA_20"] + 2 * bb_std
    df["BB_Lower"]   = df["SMA_20"] - 2 * bb_std

    # RSI
    delta            = close.diff()
    gain             = delta.clip(lower=0).rolling(window=14).mean()
    loss             = (-delta.clip(upper=0)).rolling(window=14).mean()
    rs               = gain / loss.replace(0, np.nan)
    df["RSI"]        = 100 - (100 / (1 + rs))

    # 1-day return
    df["Return_1d"]  = close.pct_change().clip(-0.5, 0.5)

    # Intraday range
    if high is not None and low is not None:
        df["High_Low_Range"] = (high - low) / close

    # Volume Change
    if volume is not None:
        df["Volume_Change"] = volume.pct_change().fillna(0).clip(-5, 5)

    # Label: 1 = harga besok naik (Buy), 0 = turun (Sell)
    df["Target"] = (close.shift(-1) > close).astype(int)

    return df
