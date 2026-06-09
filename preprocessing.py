import pandas as pd
import numpy as np


def load_and_preprocess(filepath, sample_size=100000):
    """
    Load S&P 500 OHLCV CSV, sortir per tanggal dan ticker,
    lalu sample per-ticker agar representatif.

    FIX dari versi sebelumnya:
      - Versi lama: df.tail(sample_size) → hanya ambil saham terakhir
        secara alfabetis, tidak representatif
      - Sekarang: sample merata per ticker (stratified sampling)
    """
    print(f"  Membaca {filepath}...")
    df = pd.read_csv(filepath, parse_dates=["Date"], low_memory=False)

    # Standarisasi kolom
    df.columns = [c.strip() for c in df.columns]

    # Pastikan kolom wajib ada
    required = ["Date", "Close"]
    for col in required:
        if col not in df.columns:
            raise ValueError(
                f"Kolom '{col}' tidak ditemukan. " f"Kolom tersedia: {list(df.columns)}"
            )

    # Konversi tipe
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Hapus data tidak valid
    df = df.dropna(subset=["Date", "Close"])
    df = df[df["Close"] > 0]
    df = df.drop_duplicates()

    # Deteksi kolom ticker secara fleksibel (Ticker / Symbol / ticker / symbol)
    ticker_col = None
    for candidate in ["Ticker", "Symbol", "ticker", "symbol", "TICKER", "SYMBOL"]:
        if candidate in df.columns:
            ticker_col = candidate
            break
    if ticker_col and ticker_col != "Ticker":
        df = df.rename(columns={ticker_col: "Ticker"})
        print(f"  ✓ Kolom ticker terdeteksi sebagai '{ticker_col}', diubah ke 'Ticker'")

    # Sortir: ticker dulu, lalu tanggal (penting untuk rolling window yang benar)
    sort_cols = ["Date"]
    if "Ticker" in df.columns:
        sort_cols = ["Ticker", "Date"]
    df = df.sort_values(sort_cols).reset_index(drop=True)

    # Stratified sampling per ticker agar semua saham terwakili
    if len(df) > sample_size and "Ticker" in df.columns:
        n_tickers = df["Ticker"].nunique()
        rows_per_ticker = max(
            60, sample_size // n_tickers
        )  # min 60 agar rolling window cukup

        sampled = df.groupby("Ticker", group_keys=False).apply(
            lambda g: g.tail(rows_per_ticker)
        )
        df = sampled.reset_index(drop=True)

        # Trim ke sample_size jika masih terlalu besar
        if len(df) > sample_size:
            df = df.tail(sample_size).reset_index(drop=True)

    elif len(df) > sample_size:
        df = df.tail(sample_size).reset_index(drop=True)

    print(
        f"  ✓ Data dimuat: {len(df):,} baris, "
        f"{df['Ticker'].nunique() if 'Ticker' in df.columns else 'N/A'} ticker"
    )
    return df
