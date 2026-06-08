import pandas as pd
import numpy as np


def add_technical_indicators(df):
    # SMA & EMA
    df["SMA_20"] = df["Close"].rolling(window=20).mean()
    df["EMA_12"] = df["Close"].ewm(span=12, adjust=False).mean()
    df["EMA_26"] = df["Close"].ewm(span=26, adjust=False).mean()

    # MACD
    df["MACD"] = df["EMA_12"] - df["EMA_26"]

    # Bollinger Bands
    df["BB_std"] = df["Close"].rolling(window=20).std()
    df["BB_Upper"] = df["SMA_20"] + (df["BB_std"] * 2)
    df["BB_Lower"] = df["SMA_20"] - (df["BB_std"] * 2)

    # RSI (Relative Strength Index)
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    # Target Label: 1 jika besok harga naik (Buy), 0 jika turun (Sell)
    df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

    # Buang baris yang ada NaN akibat proses rolling window
    df = df.dropna().reset_index(drop=True)

    # Pilih fitur yang akan dipakai untuk model
    features = ["RSI", "MACD", "BB_Upper", "BB_Lower", "Close"]
    X = df[features].values
    y = df["Target"].values

    return X, y
