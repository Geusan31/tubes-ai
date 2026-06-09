# Core Libraries
import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-v0_8-darkgrid')

# ML Libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor

np.random.seed(42)
print("✅ Environment Ready")
# Auto-discover CSV files
print("🔍 Loading data...")
all_csv_files = []
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        if filename.endswith('.csv'):
            all_csv_files.append(os.path.join(dirname, filename))

if len(all_csv_files) == 0:
    raise FileNotFoundError("❌ No CSV files found!")

df = pd.read_csv(all_csv_files[0])
print(f"✅ Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"\nColumns: {list(df.columns)}")
print(f"\nFirst 5 rows:")
display(df.head())

# Check if multi-stock dataset
potential_ticker_cols = [col for col in df.columns if any(x in col.lower() for x in ['symbol', 'ticker', 'stock'])]
if len(potential_ticker_cols) > 0:
    ticker_col = potential_ticker_cols[0]
    unique_tickers = df[ticker_col].nunique()
    print(f"\n⚠️ Multi-stock dataset detected!")
    print(f"   Ticker column: '{ticker_col}'")
    print(f"   Unique stocks: {unique_tickers}")
    print(f"   First 10 tickers: {df[ticker_col].unique()[:10].tolist()}")
else:
    ticker_col = None

print("🧹 Cleaning data...\n")
df_clean = df.copy()

# CRITICAL: If multi-stock dataset, filter to ONE stock or sample intelligently
if ticker_col is not None:
    print("🎯 Filtering strategy for multi-stock dataset:")
    
    # Option 1: Take most common stock
    most_common_ticker = df_clean[ticker_col].value_counts().index[0]
    print(f"   Most common ticker: {most_common_ticker} ({df_clean[ticker_col].value_counts().iloc[0]:,} records)")
    
    # Filter to single stock
    df_clean = df_clean[df_clean[ticker_col] == most_common_ticker].copy()
    print(f"   ✅ Filtered to {most_common_ticker}: {len(df_clean):,} rows\n")

# SPEED OPTIMIZATION: If still too large, sample intelligently
MAX_ROWS = 50000  # Limit for fast training
if len(df_clean) > MAX_ROWS:
    print(f"⚡ Dataset too large ({len(df_clean):,} rows)")
    print(f"   Sampling to {MAX_ROWS:,} most recent rows for speed...")
    df_clean = df_clean.tail(MAX_ROWS).reset_index(drop=True)
    print(f"   ✅ Sampled dataset: {len(df_clean):,} rows\n")

# Parse date if exists
date_cols = [col for col in df_clean.columns if 'date' in col.lower()]
if len(date_cols) > 0:
    date_col = date_cols[0]
    df_clean[date_col] = pd.to_datetime(df_clean[date_col])
    df_clean = df_clean.sort_values(date_col).reset_index(drop=True)
    print(f"✅ Date column: '{date_col}'")
    print(f"   Date range: {df_clean[date_col].min()} to {df_clean[date_col].max()}")

# Remove duplicates
initial_rows = len(df_clean)
df_clean = df_clean.drop_duplicates()
if initial_rows > len(df_clean):
    print(f"✅ Removed {initial_rows - len(df_clean)} duplicate rows")

# Handle missing values
numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
print(f"\n✅ Found {len(numeric_cols)} numeric columns")

missing_before = df_clean[numeric_cols].isnull().sum().sum()
if missing_before > 0:
    print(f"   Missing values: {missing_before}")
    df_clean[numeric_cols] = df_clean[numeric_cols].fillna(method='ffill').fillna(method='bfill')
    
    for col in numeric_cols:
        if df_clean[col].isnull().any():
            df_clean[col].fillna(df_clean[col].median(), inplace=True)

# Verify
final_missing = df_clean[numeric_cols].isnull().sum().sum()
assert final_missing == 0, f"❌ {final_missing} missing values remain!"
print(f"✅ Clean data: {df_clean.shape}")
print(f"✅ Zero missing values guaranteed")

# Find close price column
close_cols = [col for col in df_clean.columns if 'close' in col.lower()]
if len(close_cols) > 0:
    close_col = close_cols[0]
    print(f"📊 Using close price column: '{close_col}'\n")
    
    print(f"Price Statistics:")
    print(f"   Min: ${df_clean[close_col].min():,.2f}")
    print(f"   Max: ${df_clean[close_col].max():,.2f}")
    print(f"   Mean: ${df_clean[close_col].mean():,.2f}")
    
    # Quick visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    
    # Price trend
    if 'date_col' in locals():
        axes[0].plot(df_clean[date_col], df_clean[close_col], linewidth=1, color='steelblue')
    else:
        axes[0].plot(df_clean[close_col], linewidth=1, color='steelblue')
    axes[0].set_title('Price Trend', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Price ($)')
    axes[0].grid(True, alpha=0.3)
    
    # Daily returns
    returns = df_clean[close_col].pct_change() * 100
    axes[1].hist(returns.dropna(), bins=50, color='coral', edgecolor='black', alpha=0.7)
    axes[1].set_title('Daily Returns', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Return (%)')
    axes[1].axvline(0, color='red', linestyle='--')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
else:
    print("⚠️ Close column not found")
    close_col = None
    
print("🔧 Creating features...\n")
df_features = df_clean.copy()

if close_col is not None:
    close = df_features[close_col]
    
    # Moving Averages
    df_features['MA_5'] = close.rolling(5, min_periods=1).mean()
    df_features['MA_20'] = close.rolling(20, min_periods=1).mean()
    df_features['MA_50'] = close.rolling(50, min_periods=1).mean()
    
    # Momentum
    df_features['Momentum_5'] = close - close.shift(5)
    df_features['Momentum_20'] = close - close.shift(20)
    
    # Volatility
    df_features['Volatility_20'] = close.rolling(20, min_periods=1).std()
    
    # RSI
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14, min_periods=1).mean()
    loss = -delta.where(delta < 0, 0).rolling(14, min_periods=1).mean()
    rs = gain / (loss + 1e-10)
    df_features['RSI'] = 100 - (100 / (1 + rs))
    
    # Lagged features
    df_features['Close_Lag_1'] = close.shift(1)
    df_features['Close_Lag_2'] = close.shift(2)
    df_features['Close_Lag_5'] = close.shift(5)
    
    # Returns
    df_features['Return_5d'] = ((close / (close.shift(5) + 1e-10)) - 1) * 100
    df_features['Return_20d'] = ((close / (close.shift(20) + 1e-10)) - 1) * 100
    
    # Temporal
    if 'date_col' in locals():
        df_features['Month'] = df_features[date_col].dt.month
        df_features['DayOfWeek'] = df_features[date_col].dt.dayofweek
    
    print(f"✅ Created {df_features.shape[1] - df_clean.shape[1]} features")
    
    # Remove NaN
    df_features = df_features.dropna()
    print(f"✅ Final shape: {df_features.shape}")
else:
    print("⚠️ Cannot create features")

print("📊 Preparing data...\n")

# Target
df_features['Target'] = df_features[close_col].shift(-1)
df_features = df_features[:-1]

# Exclude columns
exclude_cols = ['Target']
if 'date_col' in locals():
    exclude_cols.append(date_col)
if 'ticker_col' in locals() and ticker_col in df_features.columns:
    exclude_cols.append(ticker_col)

# Exclude original OHLCV
for col in df_features.columns:
    if any(p in col.lower() for p in ['open', 'high', 'low', 'close', 'volume', 'adj']):
        if not any(eng in col for eng in ['MA_', 'Lag_', 'Ratio', 'Momentum', 'Volatility', 'RSI', 'Return', 'Month', 'Day']):
            if col not in exclude_cols:
                exclude_cols.append(col)

feature_cols = [col for col in df_features.columns if col not in exclude_cols]

X = df_features[feature_cols].copy()
y = df_features['Target'].copy()

# Remove categorical
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
if len(categorical_cols) > 0:
    print(f"Dropping {len(categorical_cols)} categorical columns")
    X = X.select_dtypes(include=[np.number])
    feature_cols = X.columns.tolist()

# Clean
X = X.replace([np.inf, -np.inf], np.nan)
valid_idx = X.notna().all(axis=1) & y.notna()
X = X[valid_idx]
y = y[valid_idx]

assert X.isnull().sum().sum() == 0
assert y.isnull().sum() == 0

print(f"✅ X: {X.shape}")
print(f"✅ y: {y.shape}")
print(f"✅ Features: {feature_cols}")

print("✂️ Splitting...\n")

split_idx = int(len(X) * 0.8)

X_train = X.iloc[:split_idx]
X_test = X.iloc[split_idx:]
y_train = y.iloc[:split_idx]
y_test = y.iloc[split_idx:]

print(f"✅ Train: {len(X_train):,}")
print(f"✅ Test: {len(X_test):,}")

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("✅ Scaled")

def evaluate_fast(model, X_tr, X_te, y_tr, y_te, name):
    print(f"\n{'='*60}\nTraining: {name}\n{'='*60}")
    try:
        model.fit(X_tr, y_tr)
        y_train_pred = model.predict(X_tr)
        y_test_pred = model.predict(X_te)
        
        train_r2 = r2_score(y_tr, y_train_pred)
        test_r2 = r2_score(y_te, y_test_pred)
        test_rmse = np.sqrt(mean_squared_error(y_te, y_test_pred))
        test_mae = mean_absolute_error(y_te, y_test_pred)
        test_mape = np.mean(np.abs((y_te - y_test_pred) / (np.abs(y_te) + 1e-10))) * 100
        
        print(f"\nR² Train: {train_r2:.4f} | Test: {test_r2:.4f}")
        print(f"RMSE: ${test_rmse:.2f} | MAE: ${test_mae:.2f} | MAPE: {test_mape:.2f}%")
        
        return {'name': name, 'model': model, 'test_r2': test_r2, 'test_rmse': test_rmse, 
                'test_mae': test_mae, 'test_mape': test_mape, 'predictions': y_test_pred}
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

results = []
# Gradient Boosting (VERY FAST settings)
gb = GradientBoostingRegressor(n_estimators=30, max_depth=3, learning_rate=0.1, random_state=42)
gb_res = evaluate_fast(gb, X_train, X_test, y_train, y_test, "Gradient Boosting")
if gb_res:
    results.append(gb_res)

# XGBoost (VERY FAST settings)
xgb = XGBRegressor(n_estimators=30, max_depth=3, learning_rate=0.1, 
                   tree_method='hist', random_state=42, n_jobs=-1)
xgb_res = evaluate_fast(xgb, X_train, X_test, y_train, y_test, "XGBoost")
if xgb_res:
    results.append(xgb_res)
    
print("\n" + "="*60)
print("📊 FINAL RESULTS")
print("="*60 + "\n")

if len(results) > 0:
    comparison = pd.DataFrame([{
        'Model': r['name'],
        'R²': r['test_r2'],
        'RMSE': r['test_rmse'],
        'MAPE (%)': r['test_mape']
    } for r in results]).sort_values('R²', ascending=False)
    
    display(comparison)
    
    best = results[0] if results[0]['test_r2'] >= results[-1]['test_r2'] else results[-1]
    for r in results:
        if r['test_r2'] > best['test_r2']:
            best = r
    
    print(f"\n🏆 BEST: {best['name']}")
    print(f"   R²: {best['test_r2']:.4f}")
    print(f"   RMSE: ${best['test_rmse']:.2f}")
    print(f"   MAPE: {best['test_mape']:.2f}%")
    
    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    ax.plot(y_test.values[:500], label='Actual', linewidth=2)
    ax.plot(best['predictions'][:500], label='Predicted', linewidth=2, linestyle='--')
    ax.set_title(f'{best["name"]} Predictions', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Feature importance
    if hasattr(best['model'], 'feature_importances_'):
        imp_df = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': best['model'].feature_importances_
        }).sort_values('Importance', ascending=False)
        print(f"\n📊 Top 10 Features:\n")
        display(imp_df.head(10))
else:
    print("❌ No models succeeded")

print("\n✅ Notebook Complete!")
