import pandas as pd
import numpy as np

def load_and_preprocess_data(filepath, train_ratio=0.8):
    print("Memuat dataset...")
    df = pd.read_csv(filepath)
    
    # Memastikan data diurutkan berdasarkan tanggal
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    
    # Membuat Fitur (Features)
    df['Return'] = df['Close'].pct_change()
    df['MA_5'] = df['Close'].rolling(window=5).mean()
    df['MA_20'] = df['Close'].rolling(window=20).mean()
    df['Volatility'] = df['Return'].rolling(window=20).std()
    
    # Membuat Target (Label): 1 jika harga besok lebih tinggi dari hari ini, 0 jika turun
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    
    # Hapus baris yang memiliki nilai NaN akibat pergeseran (rolling/shift)
    df = df.dropna()
    
    # Pilih fitur yang akan digunakan
    features = ['Return', 'MA_5', 'MA_20', 'Volatility']
    X = df[features].values
    y = df['Target'].values
    
    # Normalisasi Fitur menggunakan Min-Max Scaling (Penting untuk KNN)
    X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    
    # Split Train dan Test (Time series split, bukan random split)
    split_idx = int(len(X) * train_ratio)
    
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"Data latih: {X_train.shape[0]} baris, Data uji: {X_test.shape[0]} baris")
    return X_train, X_test, y_train, y_test