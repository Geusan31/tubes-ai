import pandas as pd


def load_and_preprocess(filepath, sample_size=10000):
    # Load dataset
    df = pd.read_csv(filepath)

    # Sortir berdasarkan tanggal
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date")

    # Drop missing values
    df = df.dropna()

    # Ambil sampel agar komputasi manual masuk akal untuk laptop
    if len(df) > sample_size:
        df = df.tail(sample_size).copy()

    df.reset_index(drop=True, inplace=True)
    return df
