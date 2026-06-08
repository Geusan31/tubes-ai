import pandas as pd
import numpy as np


def add_technical_indicators(df):
    """
    Hitung indikator teknikal per-ticker agar tidak cross-contaminate
    antar saham yang berbeda.

    FIX dari versi sebelumnya:
      - Versi lama: rolling window dihitung tanpa groupby ticker
        → rolling SMA saham A tercampur dengan saham B
      - Sekarang: groupby Ticker, lalu apply _calc_indicators
      - Normalisasi Min-Max manual (tanpa sklearn)
    """
    # Hitung indikator per ticker jika kolom Ticker ada
    if "Ticker" in df.columns and df["Ticker"].nunique() > 1:
        df = df.groupby("Ticker", group_keys=False).apply(_calc_indicators)
    else:
        df = _calc_indicators(df)

    df = df.dropna().reset_index(drop=True)

    features = [
        "RSI",
        "MACD",
        "BB_Upper",
        "BB_Lower",
        "SMA_20",
        "EMA_12",
        "Close",
        "Volume_Change",
    ]
    # Pastikan semua kolom fitur ada
    features = [f for f in features if f in df.columns]

    X = df[features].values.astype(np.float64)
    y = df["Target"].values.astype(np.int32)

    # Min-Max Normalisasi manual (from scratch)
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    denom = np.where((X_max - X_min) == 0, 1.0, X_max - X_min)
    X = (X - X_min) / denom

    # Distribusi label
    unique, counts = np.unique(y, return_counts=True)
    print("  Distribusi label setelah feature engineering:")
    for u, c in zip(unique, counts):
        label = "Buy (1)" if u == 1 else "Sell (0)"
        print(f"    {label}: {c:,} ({c/len(y)*100:.1f}%)")

    return X, y


def _calc_indicators(df):
    close = df["Close"]
    volume = df["Volume"] if "Volume" in df.columns else None

    # SMA & EMA
    df = df.copy()
    df["SMA_20"] = close.rolling(window=20).mean()
    df["EMA_12"] = close.ewm(span=12, adjust=False).mean()
    df["EMA_26"] = close.ewm(span=26, adjust=False).mean()

    # MACD
    df["MACD"] = df["EMA_12"] - df["EMA_26"]

    # Bollinger Bands
    bb_std = close.rolling(window=20).std()
    df["BB_Upper"] = df["SMA_20"] + 2 * bb_std
    df["BB_Lower"] = df["SMA_20"] - 2 * bb_std

    # RSI
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window=14).mean()
    loss = (-delta.clip(upper=0)).rolling(window=14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))

    # Volume Change
    if volume is not None:
        df["Volume_Change"] = volume.pct_change().fillna(0).clip(-5, 5)

    # Label: 1 = harga besok naik (Buy), 0 = turun (Sell)
    df["Target"] = (close.shift(-1) > close).astype(int)

    return df
