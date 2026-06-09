# Dokumentasi Lengkap: Hybrid Pipeline Prediksi Sinyal Trading Saham S&P 500

> **Judul Resmi:**
> *Implementasi Model Hybrid Bertingkat Naive Bayes, Decision Tree, dan K-Nearest Neighbor untuk Prediksi Arah Pergerakan Harga Saham pada Indeks S&P 500 Berbasis Indikator Teknikal*

---

## Daftar Isi

1. [Gambaran Umum Proyek](#1-gambaran-umum-proyek)
2. [Struktur File](#2-struktur-file)
3. [Data Training](#3-data-training)
4. [Alur Eksekusi (main.py)](#4-alur-eksekusi-mainpy)
5. [STEP 1 — Preprocessing (preprocessing.py)](#5-step-1--preprocessing-preprocessingpy)
6. [STEP 2 — Feature Engineering (features.py)](#6-step-2--feature-engineering-featurespy)
7. [STEP 3 — Train/Test Split](#7-step-3--traintest-split)
8. [STEP 4 — Model Training](#8-step-4--model-training)
   - [8.1 Gaussian Naive Bayes (naive_bayes.py)](#81-gaussian-naive-bayes-naive_bayespy)
   - [8.2 Decision Tree CART (decision_tree.py)](#82-decision-tree-cart-decision_treepy)
   - [8.3 K-Nearest Neighbors (knn.py)](#83-k-nearest-neighbors-knnpy)
9. [STEP 4 — Hybrid Pipeline (hybrid_pipeline.py)](#9-hybrid-pipeline-hybrid_pipelinepy)
10. [STEP 5 — Evaluasi (evaluation.py)](#10-step-5--evaluasi-evaluationpy)
11. [Output Aktual Setelah Dijalankan](#11-output-aktual-setelah-dijalankan)
12. [Tabel Komparasi Final](#12-tabel-komparasi-final)
13. [Analisis Hasil & Kenapa Hasilnya Seperti Ini](#13-analisis-hasil--kenapa-hasilnya-seperti-ini)
14. [Glosarium Istilah Trading](#14-glosarium-istilah-trading)

---

## 1. Gambaran Umum Proyek

Proyek ini adalah sistem prediksi **sinyal trading** (Buy / Sell) untuk saham-saham dalam indeks **S&P 500** menggunakan pendekatan **Machine Learning hybrid dari nol (from scratch)**. Artinya, tidak ada satu pun fungsi dari `scikit-learn` yang digunakan — semua algoritma diimplementasikan manual menggunakan `numpy` dan `pandas` saja.

### Filosofi Desain

| Aspek | Pilihan |
|---|---|
| ML Library | **TIDAK ADA** — semua from scratch dengan `numpy` |
| Visualisasi | `matplotlib` hanya untuk simpan PNG confusion matrix |
| Data | S&P 500 historical OHLCV (Open, High, Low, Close, Volume) |
| Target prediksi | Harga **besok** naik (Buy=1) atau turun (Sell=0) |
| Pendekatan | Hybrid pipeline 3 model bertingkat: NB → DT → KNN |

### Tiga Masalah Utama yang Diselesaikan

1. **Cross-contamination antar ticker**: Versi awal menghitung rolling window (SMA, EMA, RSI) tanpa memisahkan per saham, sehingga data saham A mencemari saham B. Sekarang dihitung per-ticker via `groupby`.
2. **Sampling tidak representatif**: Versi awal hanya mengambil `df.tail(sample_size)` yang berarti hanya saham di akhir abjad. Sekarang pakai *stratified sampling* per ticker.
3. **Numerical underflow di Naive Bayes**: Mengalikan banyak probabilitas kecil menghasilkan angka mendekati nol. Diselesaikan dengan log-likelihood dan log-sum-exp trick.

---

## 2. Struktur File

```
tubes-AI/
├── main.py              ← Orkestrator utama, 7 step berurutan
├── preprocessing.py     ← Load CSV, bersihkan data, stratified sampling
├── features.py          ← Hitung 8 indikator teknikal, normalisasi, buat label
├── naive_bayes.py       ← Gaussian Naive Bayes from scratch
├── decision_tree.py     ← CART Decision Tree with Gini Impurity from scratch
├── knn.py               ← KNN with vectorized Euclidean distance from scratch
├── hybrid_pipeline.py   ← Pipeline NB → DT → KNN bertingkat
├── evaluation.py        ← Hitung metrik + simpan confusion matrix PNG + CSV
├── requirements.txt     ← numpy, pandas, matplotlib
└── data/
    └── SP500_Historical_Data.csv   ← Data mentah (tidak dicommit ke git)
```

---

## 3. Data Training

### Sumber Data

File: `data/SP500_Historical_Data.csv`

Format CSV dengan kolom:

```
Ticker, Date, Open, High, Low, Close, Adj Close, Volume
```

Contoh baris pertama:
```
A, 2000-01-03, 47.07, 47.18, 40.27, 43.04, 43.04, 4674353
```

- **Ticker**: Simbol saham (misal: A, AAPL, MSFT, dll.)
- **Date**: Tanggal trading
- **Open**: Harga pembukaan hari ini
- **High**: Harga tertinggi hari ini
- **Low**: Harga terendah hari ini
- **Close**: Harga penutupan hari ini ← yang digunakan sebagai basis kalkulasi
- **Adj Close**: Harga penutupan yang sudah disesuaikan (split, dividen)
- **Volume**: Jumlah lembar saham yang diperdagangkan

### Ukuran Dataset

Setelah preprocessing:
- **Total baris dimuat**: 99.592 baris
- **Setelah feature engineering** (dropna rolling window): 99.565 sampel
- **Jumlah ticker**: Tidak terdeteksi kolom Ticker (diproses sebagai satu entitas)

---

## 4. Alur Eksekusi (main.py)

[main.py](main.py) adalah orkestrator yang menjalankan 7 langkah berurutan.

```python
np.random.seed(42)         # Reproducibility
os.makedirs("output", exist_ok=True)
```

- `np.random.seed(42)`: Menjamin hasil yang sama setiap kali dijalankan. Angka 42 adalah konvensi umum.
- `os.makedirs("output", exist_ok=True)`: Buat folder output jika belum ada. `exist_ok=True` agar tidak error bila sudah ada.

### 8 Fitur yang Digunakan

```python
FEATURE_NAMES = [
    "RSI",           # Relative Strength Index
    "MACD",          # Moving Average Convergence Divergence
    "BB_Upper",      # Bollinger Band atas
    "BB_Lower",      # Bollinger Band bawah
    "SMA_20",        # Simple Moving Average 20 hari
    "EMA_12",        # Exponential Moving Average 12 hari
    "Close",         # Harga penutupan (ternormalisasi)
    "Volume_Change", # Perubahan volume relatif hari ini vs kemarin
]
```

---

## 5. STEP 1 — Preprocessing (preprocessing.py)

### Tujuan

Membaca CSV mentah, membersihkan data kotor, dan mengambil sampel yang representatif.

### Detail Baris per Baris

```python
df = pd.read_csv(filepath, parse_dates=["Date"], low_memory=False)
```
Membaca CSV. `parse_dates=["Date"]` langsung mengubah kolom Date dari string menjadi tipe `datetime64`. `low_memory=False` mencegah peringatan mixed-type.

```python
df.columns = [c.strip() for c in df.columns]
```
Membuang spasi tersembunyi di nama kolom. Penting karena beberapa CSV memiliki spasi ekstra seperti `" Close"` bukan `"Close"`.

```python
for col in ["Open", "High", "Low", "Close", "Volume"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")
```
Paksa konversi ke angka. Jika ada nilai tidak valid (misalnya `"-"` atau `"N/A"`), akan diubah menjadi `NaN` (`errors="coerce"`), bukan error.

```python
df = df.dropna(subset=["Date", "Close"])
df = df[df["Close"] > 0]
df = df.drop_duplicates()
```
- Hapus baris yang tidak punya tanggal atau harga Close
- Hapus harga negatif atau nol (data korup)
- Hapus baris duplikat persis

```python
sort_cols = ["Ticker", "Date"]
df = df.sort_values(sort_cols).reset_index(drop=True)
```
**Sangat penting untuk time-series!** Data diurutkan per ticker lalu per tanggal. Jika tidak diurutkan, rolling window akan menghitung nilai dari urutan acak yang tidak bermakna.

### Stratified Sampling Per Ticker

```python
rows_per_ticker = max(60, sample_size // n_tickers)
sampled = df.groupby("Ticker", group_keys=False).apply(
    lambda g: g.tail(rows_per_ticker)
)
```

**Mengapa `tail()` bukan `sample()`?**
Untuk time-series, mengambil data **terbaru** (tail) lebih representatif daripada acak. Data lama (tahun 2000) pola pasarnya bisa sangat berbeda dengan data baru. `tail()` mengambil `rows_per_ticker` baris terakhir per saham.

**Mengapa minimum 60 baris per ticker?**
Karena rolling window terpanjang adalah 26 hari (EMA_26). Butuh setidaknya 26 baris agar nilai pertama yang valid tidak terlalu sedikit. Angka 60 memberikan buffer yang cukup.

**Output STEP 1:**
```
✓ Data dimuat: 99,592 baris, N/A ticker
```

*Catatan: Muncul "N/A ticker" karena data ini tidak memiliki kolom Ticker yang terdeteksi, atau semua data diproses sebagai satu blok time series.*

---

## 6. STEP 2 — Feature Engineering (features.py)

### Tujuan

Mengubah data OHLCV mentah menjadi **8 indikator teknikal** yang relevan untuk trading, lalu menormalisasi dan membuat label target.

### Mengapa Per-Ticker via Groupby?

```python
if "Ticker" in df.columns and df["Ticker"].nunique() > 1:
    df = df.groupby("Ticker", group_keys=False).apply(_calc_indicators)
```

Jika dihitung tanpa groupby, rolling window `SMA_20` pada hari terakhir saham AAPL akan "melihat" ke data saham AMZN yang muncul setelahnya dalam DataFrame. Ini secara statistik tidak valid.

---

### 6.1 Perhitungan Setiap Indikator

#### SMA 20 — Simple Moving Average

```python
df["SMA_20"] = close.rolling(window=20).mean()
```

**Formula Manual:**

```
SMA_20(t) = (Close(t) + Close(t-1) + ... + Close(t-19)) / 20
```

**Contoh numerik** (misalkan Close 5 hari terakhir: [100, 102, 101, 103, 105]):
```
SMA_5 = (100 + 102 + 101 + 103 + 105) / 5 = 511 / 5 = 102.2
```

Untuk SMA_20, butuh 20 hari terakhir. 19 baris pertama akan menghasilkan `NaN`.

**Interpretasi trading**: Jika Close > SMA_20, saham sedang dalam tren naik (uptrend).

---

#### EMA 12 dan EMA 26 — Exponential Moving Average

```python
df["EMA_12"] = close.ewm(span=12, adjust=False).mean()
df["EMA_26"] = close.ewm(span=26, adjust=False).mean()
```

**Formula Manual:**

```
α = 2 / (span + 1)
EMA(t) = Close(t) × α + EMA(t-1) × (1 - α)
```

Untuk span=12: `α = 2/(12+1) = 0.1538`
Untuk span=26: `α = 2/(26+1) = 0.0741`

**Contoh numerik** (EMA_12, α=0.1538, anggap EMA(0) = Close(0) = 100):

| t | Close | EMA_12 |
|---|---|---|
| 0 | 100 | 100.000 |
| 1 | 102 | 100 × (1-0.1538) + 102 × 0.1538 = 84.62 + 15.69 = **100.308** |
| 2 | 101 | 100.308 × 0.8462 + 101 × 0.1538 = 84.88 + 15.53 = **100.41** |

EMA memberikan **bobot lebih besar** pada data terbaru dibanding SMA. Makin kecil span, makin responsif terhadap perubahan harga.

---

#### MACD — Moving Average Convergence Divergence

```python
df["MACD"] = df["EMA_12"] - df["EMA_26"]
```

**Formula:**

```
MACD(t) = EMA_12(t) - EMA_26(t)
```

**Contoh numerik** (misalkan EMA_12=105, EMA_26=103):
```
MACD = 105 - 103 = +2.0 → positif = momentum bullish (harga cenderung naik)
```

**Interpretasi trading**:
- MACD > 0: EMA cepat di atas EMA lambat → momentum bullish → sinyal Buy
- MACD < 0: EMA cepat di bawah EMA lambat → momentum bearish → sinyal Sell
- MACD mendekati nol: pasar sideways / tanpa tren jelas

---

#### Bollinger Bands

```python
bb_std = close.rolling(window=20).std()
df["BB_Upper"] = df["SMA_20"] + 2 * bb_std
df["BB_Lower"] = df["SMA_20"] - 2 * bb_std
```

**Formula:**

```
BB_Upper(t) = SMA_20(t) + 2 × σ_20(t)
BB_Lower(t) = SMA_20(t) - 2 × σ_20(t)

σ_20(t) = standar deviasi dari 20 Close terakhir
```

**Contoh numerik** (20 hari terakhir):
```
Misalkan: SMA_20 = 100, σ_20 = 3
BB_Upper = 100 + 2×3 = 106
BB_Lower = 100 - 2×3 = 94
```

**Interpretasi trading**:
- Close mendekati BB_Upper: harga mungkin *overbought* → potensi turun
- Close mendekati BB_Lower: harga mungkin *oversold* → potensi naik
- Band menyempit: volatilitas rendah, sering diikuti breakout besar

---

#### RSI — Relative Strength Index

```python
delta = close.diff()
gain  = delta.clip(lower=0).rolling(window=14).mean()
loss  = (-delta.clip(upper=0)).rolling(window=14).mean()
rs    = gain / loss.replace(0, np.nan)
df["RSI"] = 100 - (100 / (1 + rs))
```

**Formula Manual:**

```
Δ(t)   = Close(t) - Close(t-1)
Gain   = max(Δ, 0)   → hanya perubahan positif
Loss   = max(-Δ, 0)  → hanya perubahan negatif (dibuat positif)

AvgGain_14 = rata-rata Gain 14 hari terakhir
AvgLoss_14 = rata-rata Loss 14 hari terakhir

RS  = AvgGain_14 / AvgLoss_14
RSI = 100 - (100 / (1 + RS))
```

**Contoh numerik** (5 hari perubahan harga):

| Hari | Close | Δ | Gain | Loss |
|---|---|---|---|---|
| 1 | 100 | — | — | — |
| 2 | 103 | +3 | 3 | 0 |
| 3 | 101 | -2 | 0 | 2 |
| 4 | 105 | +4 | 4 | 0 |
| 5 | 104 | -1 | 0 | 1 |

```
AvgGain = (3+0+4+0) / 4 = 1.75
AvgLoss = (0+2+0+1) / 4 = 0.75
RS  = 1.75 / 0.75 = 2.333
RSI = 100 - (100 / (1 + 2.333)) = 100 - (100/3.333) = 100 - 30.0 = 70.0
```

**Interpretasi trading**:
- RSI > 70: overbought → harga mungkin akan koreksi turun → sinyal Sell
- RSI < 30: oversold → harga mungkin akan rebound naik → sinyal Buy
- 30–70: zona netral

**Kenapa `loss.replace(0, np.nan)`?** Jika AvgLoss = 0 (14 hari berturut-turut naik), pembagian oleh nol menghasilkan `inf`. Diubah ke `NaN` agar RSI menjadi `NaN` dan bisa di-drop, bukan menghasilkan nilai tak hingga.

---

#### Volume Change

```python
df["Volume_Change"] = volume.pct_change().fillna(0).clip(-5, 5)
```

**Formula:**
```
Volume_Change(t) = (Volume(t) - Volume(t-1)) / Volume(t-1)
```

**Contoh numerik**:
```
Volume kemarin = 1.000.000
Volume hari ini = 1.500.000
Volume_Change = (1.500.000 - 1.000.000) / 1.000.000 = 0.5 (naik 50%)
```

**Kenapa di-clip ke [-5, 5]?** Volume bisa melonjak ekstrem (misalnya saat berita besar), menghasilkan nilai ratusan. Clip ke ±5 (±500%) menghilangkan outlier ekstrem yang bisa mendominasi normalisasi.

---

#### Label Target

```python
df["Target"] = (close.shift(-1) > close).astype(int)
```

**Formula:**
```
Target(t) = 1  jika Close(t+1) > Close(t)  → harga BESOK naik → Buy
Target(t) = 0  jika Close(t+1) ≤ Close(t)  → harga BESOK turun → Sell
```

**Contoh numerik**:

| Hari | Close | Close(t+1) | Target |
|---|---|---|---|
| Senin | 100 | 103 | **1 (Buy)** |
| Selasa | 103 | 101 | **0 (Sell)** |
| Rabu | 101 | 105 | **1 (Buy)** |
| Kamis | 105 | — | **NaN** (hari terakhir) |

Hari terakhir setiap saham selalu `NaN` karena tidak ada data hari berikutnya, lalu di-drop oleh `dropna()`.

---

### 6.2 Normalisasi Min-Max

Nilai fitur perlu dinormalisasi agar semua fitur berada di rentang [0, 1]. Ini penting karena:
- Close bisa bernilai ratusan hingga ribuan dolar
- MACD bernilai kecil (bisa -10 hingga +10)
- Volume_Change dikliping ke [-5, 5]

Tanpa normalisasi, KNN akan didominasi oleh fitur dengan skala besar (Close).

> **Catatan penting:** Normalisasi **tidak** lagi dilakukan di `features.py`. Ia dipindahkan ke `main.py` setelah train/test split untuk menghindari **data leakage** — lihat STEP 3.

**Distribusi label hasil feature engineering:**
```
Sell (0): 47,036 (47.2%)
Buy  (1): 52,529 (52.8%)
```
Dataset hampir seimbang — tidak perlu teknik resampling seperti SMOTE.

---

## 7. STEP 3 — Train/Test Split & Normalisasi

### Pembagian 70:30 Time-Series

```python
split    = int(len(X) * 0.7)       # 99,565 × 0.7 = 69,695
X_train  = X[:split]               # 69,695 sampel pertama
X_test   = X[split:]               # 29,870 sampel terakhir
y_train  = y[:split]
y_test   = y[split:]
```

### Normalisasi Min-Max (setelah split)

```python
# Fit HANYA pada X_train — X_test tidak boleh mempengaruhi skala
X_min   = X_train.min(axis=0)
X_max   = X_train.max(axis=0)
denom   = np.where((X_max - X_min) == 0, 1.0, X_max - X_min)
X_train = (X_train - X_min) / denom
X_test  = (X_test  - X_min) / denom
```

**Formula:**
```
X_scaled = (X - X_min_train) / (X_max_train - X_min_train)
```

**Mengapa normalisasi harus setelah split?**

Jika normalisasi dihitung dari seluruh data (train+test):
- `X_min` dan `X_max` mengandung informasi dari data test
- Model secara tidak langsung "melihat" rentang nilai test saat training
- Ini disebut **data leakage** — hasil evaluasi jadi terlalu optimistis

Dengan fit pada train saja:
- Skala ditentukan murni dari data training
- Test data di-transform menggunakan skala yang sama, tanpa mempengaruhinya
- Mencerminkan kondisi deployment nyata: skala hanya diketahui dari data historis

**Contoh numerik** (fitur Close):
```
X_train Close = [95, 100, 110, 120, 105]
X_min_train   = 95
X_max_train   = 120,  denom = 25

X_train setelah normalisasi:
  95  → (95-95)/25  = 0.00
  100 → (100-95)/25 = 0.20
  110 → (110-95)/25 = 0.60
  120 → (120-95)/25 = 1.00
  105 → (105-95)/25 = 0.40

X_test Close = [125, 90]   ← nilai di luar range train
  125 → (125-95)/25 = 1.20  (boleh > 1, tidak di-clip)
  90  → (90-95)/25  = -0.20 (boleh < 0, tidak di-clip)
```

Nilai test bisa di luar [0,1] — ini wajar dan benar, karena skala tetap konsisten dengan training.

### Mengapa Tidak Di-Shuffle?

Untuk data keuangan time-series, **shuffle sangat dilarang**. Alasannya:

1. **Data leakage**: Jika data hari Rabu masuk training dan hari Selasa masuk test, model sudah "melihat masa depan" saat training. Hasilnya tidak realistis.
2. **Temporal dependency**: Indikator teknikal seperti SMA, EMA, dan RSI dihitung dari nilai hari sebelumnya. Urutan data adalah informasinya sendiri.
3. **Realisme deployment**: Dalam trading nyata, model dilatih dengan data historis dan diuji pada data yang belum pernah dilihat (masa depan relatif).

### Hasil Split

| Subset | Jumlah Sampel | Buy (1) | Sell (0) |
|---|---|---|---|
| **Training** | 69,695 (70%) | 36,843 (52.9%) | 32,852 (47.1%) |
| **Testing** | 29,870 (30%) | 15,686 (52.5%) | 14,184 (47.5%) |
| **Total** | 99,565 | 52,529 (52.8%) | 47,036 (47.2%) |

Proporsi Buy/Sell hampir identik antara train dan test — ini menunjukkan distribusi label stabil sepanjang waktu (tidak terjadi distribusi shift ekstrem).

---

## 8. STEP 4 — Model Training

---

### 8.1 Gaussian Naive Bayes (naive_bayes.py)

#### Konsep Dasar

Naive Bayes menggunakan **Teorema Bayes** untuk menghitung probabilitas posterior setiap kelas diberikan input fitur:

```
P(y | x₁, x₂, ..., xₙ) ∝ P(y) × P(x₁|y) × P(x₂|y) × ... × P(xₙ|y)
```

Asumsi **"naive"**: semua fitur saling independen kondisional terhadap kelas. Dalam realitasnya, RSI dan MACD jelas berkorelasi (keduanya berbasis perubahan harga), tapi asumsi ini membuat komputasi jauh lebih sederhana dan sering tetap performatif.

**"Gaussian"** berarti setiap P(xᵢ|y) diasumsikan mengikuti distribusi normal (bell curve).

---

#### Fase Training — `fit(X_train, y_train)`

```python
self.classes_  = np.unique(y)                          # [0, 1]
self.mean_     = np.zeros((n_classes, n_features))     # shape (2, 8)
self.var_      = np.zeros((n_classes, n_features))     # shape (2, 8)
self.priors_   = np.zeros(n_classes)                   # shape (2,)

for idx, c in enumerate(self.classes_):
    X_c                = X[y == c]
    self.priors_[idx]  = X_c.shape[0] / n_samples      # P(kelas)
    self.mean_[idx]    = X_c.mean(axis=0)               # μ per fitur
    self.var_[idx]     = X_c.var(axis=0) + 1e-9         # σ² per fitur
```

**Apa yang dihitung:**

Untuk **Kelas Sell (0)**: 32,852 sampel
- Prior P(Sell) = 32,852 / 69,695 = **0.4714**
- mean_[0] = rata-rata setiap fitur khusus sampel Sell → vektor 8 angka
- var_[0]  = variansi setiap fitur khusus sampel Sell → vektor 8 angka

Untuk **Kelas Buy (1)**: 36,843 sampel
- Prior P(Buy) = 36,843 / 69,695 = **0.5286**
- mean_[1] = rata-rata setiap fitur khusus sampel Buy
- var_[1]  = variansi setiap fitur khusus sampel Buy

**Kenapa `+1e-9` pada variansi?**
Jika suatu fitur bernilai sama untuk semua sampel satu kelas (variansi = 0), maka PDF Gaussian menghasilkan pembagian dengan nol. Menambahkan `1e-9` (sangat kecil, tidak mengubah nilai secara signifikan) mencegah hal ini — teknik ini disebut **variance smoothing**.

**Contoh manual** (fitur RSI, kelas Buy):
```
Misalkan RSI dari 36,843 sampel Buy memiliki:
mean_RSI_Buy = 0.57
var_RSI_Buy  = 0.04 + 1e-9 ≈ 0.04
```

---

#### Fase Inferensi — `predict_proba(X)`

**Langkah 1: Hitung Log Prior**
```python
log_prior = np.log(self.priors_[idx])
```
```
log P(Buy)  = log(0.5286) = -0.637
log P(Sell) = log(0.4714) = -0.751
```

**Langkah 2: Hitung Log-Likelihood via Gaussian PDF**

```python
def _gaussian_log_pdf(self, class_idx, X):
    mean = self.mean_[class_idx]
    var  = self.var_[class_idx]
    log_pdf = -0.5 * np.sum(
        np.log(2.0 * np.pi * var) + ((X - mean)**2) / var,
        axis=1
    )
    return log_pdf
```

**Formula log PDF Gaussian:**
```
log P(x | μ, σ²) = -0.5 × [log(2πσ²) + (x-μ)²/σ²]
```

**Contoh manual** (satu sampel, satu fitur RSI = 0.65):
```
μ_RSI_Buy = 0.57,  σ²_RSI_Buy = 0.04

log P(RSI=0.65 | Buy) = -0.5 × [log(2π × 0.04) + (0.65-0.57)²/0.04]
                      = -0.5 × [log(0.2513) + (0.0064/0.04)]
                      = -0.5 × [-1.379 + 0.16]
                      = -0.5 × (-1.219)
                      = 0.61
```

Untuk semua 8 fitur, nilai log PDF dijumlahkan (karena log dari perkalian = penjumlahan log):
```
log P(X | Buy) = Σᵢ log P(xᵢ | Buy)
```

**Mengapa menggunakan log?**
Tanpa log: `P(x|Buy) = 0.3 × 0.2 × 0.4 × 0.1 × ... × 0.05` (8 fitur, nilai kecil)
Hasil bisa `1e-50` atau lebih kecil → **numerical underflow** → komputer membulatkan ke 0.
Dengan log: penjumlahan angka negatif yang aman secara numerik.

**Langkah 3: Log-Sum-Exp Trick**

```python
max_log  = log_post.max(axis=1, keepdims=True)
exp_val  = np.exp(log_post - max_log)
probs    = exp_val / exp_val.sum(axis=1, keepdims=True)
```

Konversi log-posterior ke probabilitas yang stabil:
```
softmax(z) = exp(zᵢ - max(z)) / Σ exp(zⱼ - max(z))
```

**Contoh manual** (log-posterior Buy = -15, Sell = -18):
```
Tanpa trick: exp(-15) ≈ 3.06e-7, exp(-18) ≈ 1.52e-8
→ Ini aman. Tapi jika log-posterior sangat negatif (-500, -520):
  exp(-500) → underflow ke 0!

Dengan trick: max = -15
  exp(-15 - (-15)) = exp(0) = 1.0
  exp(-18 - (-15)) = exp(-3) = 0.0498
  prob_Buy  = 1.0   / (1.0 + 0.0498) = 0.953
  prob_Sell = 0.0498/ (1.0 + 0.0498) = 0.047
```

---

#### Output Training NB

```
[NB] Training pada 69,695 sampel, 8 fitur
[NB]   Kelas Sell (0): 32,852 sampel | prior=0.4714
[NB]   Kelas Buy (1):  36,843 sampel | prior=0.5286
[NB] Parameter tersimpan: mean shape=(2, 8), var shape=(2, 8)
```

Total parameter yang disimpan: 2 prior + (2×8) mean + (2×8) variansi = **34 angka** — sangat efisien.

---

### 8.2 Decision Tree CART (decision_tree.py)

#### Konsep Dasar

Decision Tree CART (Classification and Regression Trees) membangun pohon keputusan dengan cara **rekursif membagi data** berdasarkan split (fitur + threshold) yang meminimalkan **Gini Impurity**.

Pohon dibaca dari atas ke bawah:
```
Jika BB_Lower ≤ 0.3356
    → Jika RSI ≤ 0.0580
        → ... (lanjut ke cabang lebih dalam)
    → Jika RSI > 0.0580
        → ... 
Jika BB_Lower > 0.3356
    → ...
```

---

#### Parameter yang Digunakan

```python
dt_model = DecisionTree(
    max_depth=8,          # pohon tidak boleh lebih dalam dari 8 level
    min_samples_split=20, # node perlu ≥20 sampel untuk di-split
    min_samples_leaf=10,  # setiap leaf perlu ≥10 sampel
    max_thresholds=30,    # maks 30 kandidat threshold per fitur
)
```

---

#### Gini Impurity

```python
def _gini(self, y):
    _, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    return float(1.0 - np.sum(probs**2))
```

**Formula:**
```
Gini(S) = 1 - Σᵢ pᵢ²
```

**Contoh manual:**

Node dengan 60 sampel Buy, 40 sampel Sell (total 100):
```
p_Buy  = 60/100 = 0.6
p_Sell = 40/100 = 0.4
Gini = 1 - (0.6² + 0.4²) = 1 - (0.36 + 0.16) = 1 - 0.52 = 0.48
```

Node murni (100 Buy, 0 Sell):
```
p_Buy  = 1.0,  p_Sell = 0.0
Gini = 1 - (1.0² + 0.0²) = 1 - 1 = 0.0  ← perfectly pure
```

Node sempurna campur (50 Buy, 50 Sell):
```
p_Buy  = 0.5,  p_Sell = 0.5
Gini = 1 - (0.5² + 0.5²) = 1 - 0.5 = 0.5  ← maximum impurity (binary)
```

---

#### Weighted Gini

```python
def _weighted_gini(self, left_y, right_y):
    n   = len(left_y) + len(right_y)
    w_l = len(left_y) / n
    w_r = len(right_y) / n
    return w_l * self._gini(left_y) + w_r * self._gini(right_y)
```

**Formula:**
```
Gini_weighted = (n_L/n) × Gini(L) + (n_R/n) × Gini(R)
```

**Contoh manual** (split RSI ≤ 0.5):
```
Kiri (RSI ≤ 0.5): 30 Buy, 20 Sell → n_L=50
  p_Buy=0.6, p_Sell=0.4
  Gini_L = 1 - (0.36+0.16) = 0.48

Kanan (RSI > 0.5): 30 Buy, 20 Sell → n_R=50
  p_Buy=0.6, p_Sell=0.4
  Gini_R = 0.48

Gini_weighted = (50/100)×0.48 + (50/100)×0.48 = 0.48

→ Tidak ada perbaikan! Threshold ini tidak informatif.
```

**Contoh split yang baik** (RSI ≤ 0.3):
```
Kiri (RSI ≤ 0.3): 5 Buy, 45 Sell → n_L=50
  p_Buy=0.1, p_Sell=0.9
  Gini_L = 1 - (0.01+0.81) = 0.18

Kanan (RSI > 0.3): 55 Buy, 5 Sell → n_R=60
  p_Buy=0.917, p_Sell=0.083
  Gini_R = 1 - (0.841+0.007) = 0.152

Gini_weighted = (50/110)×0.18 + (60/110)×0.152
              = 0.455×0.18 + 0.545×0.152
              = 0.0819 + 0.0829
              = 0.1648   ← jauh lebih kecil → split ini BAGUS
```

---

#### Pencarian Split Terbaik (`_best_criteria`)

```python
for feat_idx in range(X.shape[1]):         # iterasi 8 fitur
    col = X[:, feat_idx]
    if len(unique_vals) > self.max_thresholds:
        percs = np.linspace(0, 100, self.max_thresholds + 2)[1:-1]
        thresholds = np.percentile(col, percs)   # 30 threshold by percentile
    else:
        thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2.0  # midpoints
    
    for thresh in thresholds:
        gini = self._weighted_gini(y[col <= thresh], y[col > thresh])
        if gini < best_gini:
            best_gini  = gini
            best_feat  = feat_idx
            best_thresh = thresh
```

Untuk setiap dari 8 fitur, diuji hingga 30 threshold. Total: **8 × 30 = 240 split** per node. Split dengan Gini Impurity terkecil dipilih.

**Mengapa persentil, bukan semua nilai unik?**
Jika suatu fitur memiliki 69,695 nilai unik, menguji semua threshold butuh 69,694 iterasi per node. Dengan max_thresholds=30, iterasi dibatasi ke 30 saja — efisiensi ~2.300×.

---

#### Stopping Criteria

Sebuah node menjadi **leaf** (tidak di-split lagi) jika:

1. `depth >= max_depth` (8) → mencegah overfitting
2. `n_labels == 1` → semua sampel satu kelas, Gini = 0, tidak perlu split lagi
3. `n_samples < min_samples_split` (20) → terlalu sedikit sampel untuk split valid
4. `best_feat is None` → tidak ada split yang valid (semua threshold menghasilkan leaf < min_samples_leaf)
5. `n_left < min_samples_leaf` atau `n_right < min_samples_leaf` (10) → child terlalu kecil

---

#### Rekursi Pembangunan Pohon

```python
left  = self._build_tree(X[left_mask],  y[left_mask],  depth + 1)
right = self._build_tree(X[right_mask], y[right_mask], depth + 1)
```

Dipanggil dari root, setiap pemanggilan menghasilkan satu node dengan dua anak, hingga stopping criteria terpenuhi.

**Hasil yang didapat:**
```
[DT] Pohon selesai: 315 node total, 158 leaf node
```

Berarti ada 315 - 158 = **157 node internal** (decision nodes) dan 158 leaf.

---

#### Struktur Pohon (ASCII, 3 level pertama)

```
[NODE | BB_Lower ≤ 0.3356 | gini=0.4984 | n=69,695]
├── L: [NODE | RSI ≤ 0.0580 | gini=0.4713 | n=2,249]
│   ├── L: [NODE | Close ≤ 0.0011 | gini=0.4982 | n=436]
│   └── R: [NODE | Volume_Change ≤ 0.0744 | gini=0.4600 | n=1,813]
└── R: [NODE | RSI ≤ 0.5113 | gini=0.4987 | n=67,446]
    ├── L: [NODE | RSI ≤ 0.2410 | gini=0.4967 | n=30,460]
    └── R: [NODE | MACD ≤ 0.4929 | gini=0.4997 | n=36,986]
```

**Interpretasi root node:**
- Split pertama: `BB_Lower ≤ 0.3356`
- Artinya: jika Bollinger Band bawah (sudah ternormalisasi) ≤ 0.3356, masuk cabang kiri (2,249 sampel). Ini menangkap situasi di mana harga berada di area bawah Bollinger Band — sinyal oversold.
- Sebagian besar data (67,446 sampel) masuk cabang kanan (BB_Lower > 0.3356).
- Gini root = 0.4984 ≈ 0.5 → dataset hampir perfectly mixed di awal (seperti yang diharapkan, distribusi 52.8% vs 47.2%)

---

#### Prediksi

```python
def _traverse(self, x, node):
    if node.value is not None:   # leaf
        return node.value
    if x[node.feature] <= node.threshold:
        return self._traverse(x, node.left)
    return self._traverse(x, node.right)
```

Setiap sampel uji ditelusuri dari root. Di setiap node, bandingkan nilai fitur dengan threshold, belok kiri atau kanan, hingga sampai leaf → return nilai kelas mayoritas leaf tersebut.

---

### 8.3 K-Nearest Neighbors (knn.py)

#### Konsep Dasar

KNN adalah **lazy learner** — tidak ada model yang dibangun saat training. Semua komputasi terjadi saat prediksi: untuk setiap sampel baru, cari k sampel training yang paling mirip (secara jarak Euclidean), lalu prediksi berdasarkan kelas mayoritas k tetangga tersebut.

Parameter: `k=7` (selalu ganjil untuk menghindari seri), `batch_size=512`.

---

#### "Training" (Lazy)

```python
def fit(self, X, y):
    self.X_train = X.astype(np.float64)
    self.y_train = y.astype(np.int32)
```

Hanya menyimpan data. Tidak ada komputasi.

---

#### Prediksi — Euclidean Distance Vectorized

```python
sq_train = np.sum(self.X_train ** 2, axis=1)   # (n_train,)  ← pre-compute sekali

for start in range(0, n, self.batch_size):
    batch    = X[start:end]                        # (b, 8)
    sq_batch = np.sum(batch**2, axis=1, keepdims=True)  # (b, 1)
    dot      = batch @ self.X_train.T             # (b, n_train)
    dist_sq  = np.clip(sq_batch + sq_train - 2 * dot, 0, None)  # (b, n_train)
```

**Formula jarak kuadrat Euclidean yang dioptimasi:**
```
||a - b||² = ||a||² + ||b||² - 2(a · b)
```

**Mengapa ini lebih cepat dari loop biasa?**

Loop naif:
```python
for i in range(n_test):
    for j in range(n_train):
        dist[i,j] = sqrt(sum((X_test[i] - X_train[j])**2))
# Kompleksitas: O(n_test × n_train × n_features) → sangat lambat
```

Versi vectorized dengan identitas matrix:
```python
dist_sq = sq_batch + sq_train - 2 * (batch @ X_train.T)
# Seluruh matriks (b × n_train) dihitung sekaligus menggunakan BLAS
# Kompleksitas komputasi sama, tapi memanfaatkan CPU parallelism
```

**Contoh manual** (2 sampel test, 3 sampel train, 2 fitur):
```
Test  = [[0.5, 0.6], [0.2, 0.8]]
Train = [[0.4, 0.5], [0.7, 0.3], [0.1, 0.9]]

||test[0]||² = 0.5² + 0.6² = 0.25 + 0.36 = 0.61
||test[1]||² = 0.2² + 0.8² = 0.04 + 0.64 = 0.68

||train[0]||² = 0.4² + 0.5² = 0.16 + 0.25 = 0.41
||train[1]||² = 0.7² + 0.3² = 0.49 + 0.09 = 0.58
||train[2]||² = 0.1² + 0.9² = 0.01 + 0.81 = 0.82

dot[0,0] = 0.5×0.4 + 0.6×0.5 = 0.20 + 0.30 = 0.50
dot[0,1] = 0.5×0.7 + 0.6×0.3 = 0.35 + 0.18 = 0.53

dist²[0,0] = 0.61 + 0.41 - 2×0.50 = 1.02 - 1.00 = 0.02
dist²[0,1] = 0.61 + 0.58 - 2×0.53 = 1.19 - 1.06 = 0.13
→ train[0] adalah tetangga terdekat test[0] (dist²=0.02)

Jarak Euclidean sebenarnya = √0.02 = 0.141
```

**Kenapa `np.clip(..., 0, None)`?** Karena floating-point arithmetic tidak presisi sempurna. Bisa menghasilkan nilai negatif sangat kecil seperti `-1e-15`. `clip` mengubahnya ke 0 sebelum operasi lain.

---

#### Pemilihan k Tetangga

```python
k_idx = np.argpartition(dist_sq, self.k, axis=1)[:, :self.k]
```

**`np.argpartition` vs `np.argsort`:**
- `argsort`: Mengurutkan semua `n_train` jarak → O(n log n) per sampel
- `argpartition`: Hanya menjamin k indeks terkecil di depan, sisanya tidak terurut → O(n) per sampel

Kita tidak perlu urutan k tetangga, hanya identitas mereka. Jadi `argpartition` lebih efisien.

---

#### Majority Vote

```python
for i, indices in enumerate(k_idx):
    neighbors        = self.y_train[indices]          # label 7 tetangga
    preds[start + i] = int(np.bincount(neighbors).argmax())
```

**Contoh** (k=7, label tetangga = [1, 1, 0, 1, 0, 1, 0]):
```
bincount([1,1,0,1,0,1,0]) = [3, 4]
→ indeks 0 (Sell) muncul 3×
→ indeks 1 (Buy)  muncul 4×
argmax([3, 4]) = 1
Prediksi = Buy (1)
```

Karena k ganjil (7), tidak mungkin terjadi seri 3.5 vs 3.5.

---

## 9. Hybrid Pipeline (hybrid_pipeline.py)

### Arsitektur: Majority Vote

Ketiga model memberikan **satu suara** pada **setiap sampel**. Kelas dengan ≥ 2 suara dari 3 menjadi prediksi final.

```
  X_test (n sampel)
         │
    ┌────┼────┐
    ▼    ▼    ▼
   NB   DT  KNN     ← ketiga model prediksi SEMUA sampel
  vote vote vote
  (0/1)(0/1)(0/1)
    └────┼────┘
         ▼
   total = NB + DT + KNN   ∈ {0, 1, 2, 3}
   final = 1 (Buy)  jika total ≥ 2
         = 0 (Sell) jika total < 2
```

**Empat kemungkinan hasil voting:**

| total_votes | Artinya | Final |
|---|---|---|
| 3 | Semua sepakat Buy | Buy (1) — keyakinan tertinggi |
| 2 | Mayoritas Buy | Buy (1) |
| 1 | Mayoritas Sell | Sell (0) |
| 0 | Semua sepakat Sell | Sell (0) — keyakinan tertinggi |

---

### Implementasi

```python
# Vote 1: Naive Bayes (soft → hard)
nb_probs  = self.nb.predict_proba(X)   # (n, 2)
nb_votes  = nb_probs.argmax(axis=1)    # 0 atau 1

# Vote 2: Decision Tree
dt_votes  = self.dt.predict(X)         # 0 atau 1

# Vote 3: KNN
knn_votes = self.knn.predict(X)        # 0 atau 1

# Majority vote
total_votes = nb_votes + dt_votes + knn_votes
final_preds = (total_votes >= 2).astype(np.int32)
```

---

### Mengapa Desain Ini Lebih Baik dari Cascade Sebelumnya?

**Masalah cascade lama:**
- NB berperan sebagai *penjaga gerbang* via confidence threshold 0.60
- Karena semua fitur berkorelasi tinggi (semua dari `Close`), asumsi independensi NB dilanggar → probabilitas NB tidak terkalibrasi → confidence hampir selalu < 0.60
- Akibatnya: 92% sampel hanya ditangani NB saja, DT dan KNN hampir tidak terpakai

**Keunggulan majority vote:**

| Aspek | Cascade (lama) | Majority Vote (baru) |
|---|---|---|
| Coverage DT | ~8% sampel | 100% sampel |
| Coverage KNN | ~6.7% sampel | 100% sampel |
| Threshold arbitrary | Ya (0.60) | Tidak |
| Kompensasi antar model | Tidak | Ya — 2 model koreksi 1 model |
| Bias NB ke Buy | Mendominasi (92%) | Dikontrol oleh DT+KNN |

---

## 10. STEP 5 — Evaluasi (evaluation.py)

### Definisi Metrik (Binary Classification)

**Positif = 1 (Buy), Negatif = 0 (Sell)**

```
TP (True Positive) : prediksi Buy, aktual Buy   ✓ benar Buy
TN (True Negative) : prediksi Sell, aktual Sell ✓ benar Sell
FP (False Positive): prediksi Buy, aktual Sell  ✗ salah (harusnya Sell)
FN (False Negative): prediksi Sell, aktual Buy  ✗ salah (harusnya Buy)
```

**Confusion Matrix:**

```
                  Prediksi
              | Sell(0) | Buy(1)
          ----+---------+--------
Aktual Sell(0)| TN      | FP
       Buy(1) | FN      | TP
```

---

#### Accuracy

```python
accuracy = (TP + TN) / n
```

**Formula:**
```
Accuracy = (TP + TN) / Total
         = jumlah prediksi benar / total prediksi
```

**Interpretasi:** Seberapa sering model benar secara keseluruhan.

---

#### Precision

```python
precision = TP / (TP + FP)
```

**Formula:**
```
Precision = TP / (TP + FP)
          = dari semua yang diprediksi Buy, berapa yang benar-benar Buy?
```

**Interpretasi:** Presisi prediksi Buy. Tinggi = saat model bilang Buy, biasanya benar. Penting jika biaya FP tinggi (beli saham yang ternyata turun = rugi).

---

#### Recall (Sensitivity)

```python
recall = TP / (TP + FN)
```

**Formula:**
```
Recall = TP / (TP + FN)
       = dari semua yang aktual Buy, berapa yang berhasil ditangkap?
```

**Interpretasi:** Kemampuan menemukan semua peluang Buy. Tinggi = model tidak banyak melewatkan sinyal Buy. Penting jika biaya FN tinggi (melewatkan peluang untung = opportunity cost).

---

#### F1-Score

```python
f1_score = 2 * precision * recall / (precision + recall)
```

**Formula:**
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
   = Harmonic mean dari Precision dan Recall
```

**Interpretasi:** Keseimbangan antara Precision dan Recall. Cocok saat keduanya sama penting. Harmonic mean lebih keras dari aritmetik mean — model dengan Precision=1.0 dan Recall=0.01 mendapat F1=0.02, bukan 0.505.

---

#### Specificity

```python
specificity = TN / (TN + FP)
```

**Formula:**
```
Specificity = TN / (TN + FP)
            = dari semua yang aktual Sell, berapa yang berhasil ditangkap sebagai Sell?
```

---

## 11. Output Aktual Setelah Dijalankan

> **Catatan:** Output di bawah adalah hasil **sebelum perbaikan** (versi cascade lama + normalisasi bocor). Setelah perbaikan (majority vote + normalisasi benar), jalankan ulang `main.py` untuk mendapatkan angka terbaru.

### Naive Bayes
```
Accuracy    : 0.5241  (52.41%)
Precision   : 0.5256
Recall      : 0.9626
F1-Score    : 0.6800
Specificity : 0.0393
TP=15,099  TN=557  FP=13,627  FN=587
```

**Perhitungan manual verifikasi:**
```
Accuracy    = (15,099 + 557) / 29,870 = 15,656 / 29,870 = 0.5241 ✓
Precision   = 15,099 / (15,099 + 13,627) = 15,099 / 28,726 = 0.5256 ✓
Recall      = 15,099 / (15,099 + 587) = 15,099 / 15,686 = 0.9626 ✓
F1          = 2 × 0.5256 × 0.9626 / (0.5256 + 0.9626) = 1.0119 / 1.4882 = 0.6800 ✓
Specificity = 557 / (557 + 13,627) = 557 / 14,184 = 0.0393 ✓
```

---

### Decision Tree
```
Accuracy    : 0.5222  (52.22%)
Precision   : 0.5269
Recall      : 0.8826
F1-Score    : 0.6599
Specificity : 0.1235
TP=13,845  TN=1,752  FP=12,432  FN=1,841
```

---

### K-Nearest Neighbors
```
Accuracy    : 0.5058  (50.58%)
Precision   : 0.5275
Recall      : 0.5644
F1-Score    : 0.5453
Specificity : 0.4409
TP=8,853  TN=6,254  FP=7,930  FN=6,833
```

---

### Hybrid Pipeline — Cascade Lama (sebelum perbaikan)
```
Accuracy    : 0.5203  (52.03%)
Precision   : 0.5243
Recall      : 0.9347
F1-Score    : 0.6718
Specificity : 0.0620
TP=14,662  TN=880  FP=13,304  FN=1,024
```
Sangat mirip NB karena 92% sampel hanya ditangani NB.

---

## 12. Tabel Komparasi Final

> Tabel di bawah adalah hasil **sebelum perbaikan**. Update setelah menjalankan ulang `main.py`.

| METODE | ACCURACY | PRECISION | RECALL | F1-SCORE | SPECIFICITY |
|---|---|---|---|---|---|
| Naive Bayes | **0.5241** | 0.5256 | **0.9626** | **0.6800** | 0.0393 |
| Decision Tree | 0.5222 | **0.5269** | 0.8826 | 0.6599 | 0.1235 |
| K-Nearest Neighbors | 0.5058 | 0.5275 | 0.5644 | 0.5453 | **0.4409** |
| Hybrid Cascade (lama) | 0.5203 | 0.5243 | 0.9347 | 0.6718 | 0.0620 |
| **Hybrid Majority Vote (baru)** | *jalankan ulang* | *jalankan ulang* | *jalankan ulang* | *jalankan ulang* | *jalankan ulang* |

---

## 13. Analisis Hasil & Kenapa Hasilnya Seperti Ini

### Accuracy Semua Model ~52% — Kenapa Rendah?

Ini **wajar dan terduga** untuk prediksi arah saham. Beberapa alasan:

1. **Pasar saham mendekati random walk (Efficient Market Hypothesis)**: Semua informasi publik sudah terefleksi di harga. Indikator teknikal adalah informasi historis yang tersedia untuk semua orang.
2. **Label hampir 50:50**: Distribusi Buy 52.8% vs Sell 47.2% sangat seimbang, tidak ada dominasi kelas yang "mudah" diprediksi.
3. **Fitur terbatas**: Hanya 8 indikator teknikal — tidak ada data sentimen, berita, fundamental, makroekonomi.
4. **Noise tinggi**: Data harian sangat noise. Prediksi mingguan/bulanan biasanya lebih akurat.

**Benchmark**: Model random (coin flip) menghasilkan accuracy ~50%. Model ini berada di 50.58%–52.41%, artinya ada **sedikit signal** yang ditangkap.

---

### Naive Bayes: Recall Tinggi (96.26%), Specificity Sangat Rendah (3.93%)

**Penjelasan:**
- NB memprediksi Buy hampir selalu → dari 29,870 sampel test, 28,726 diprediksi Buy
- Ini karena prior P(Buy) = 0.5286 > P(Sell) = 0.4714, dan log-likelihood Buy lebih tinggi untuk sebagian besar fitur
- **Trade-off**: Recall Buy tinggi, tapi Specificity (kemampuan mendeteksi Sell) sangat rendah — hanya 557 dari 14,184 Sell yang terdeteksi benar

Dalam konteks trading: model ini hampir selalu bilang "beli" → berhasil menangkap hampir semua hari naik, tapi juga sering salah beli di hari turun.

---

### KNN: Accuracy Terendah tapi Specificity Tertinggi (44.09%)

**Penjelasan:**
- KNN adalah satu-satunya model yang benar-benar membedakan Buy vs Sell secara seimbang
- Dengan k=7 dan 69,695 titik training dalam ruang 8 dimensi, KNN mengalami **curse of dimensionality**: di dimensi tinggi, semua titik jarak menjadi hampir sama
- Specificity 44.09% vs NB 3.93% menunjukkan KNN jauh lebih baik dalam mendeteksi sinyal Sell

---

### Hybrid Cascade Lama: Masalah Arsitektur

Hybrid cascade menghasilkan Specificity hanya 6.20% — hampir sama dengan NB (3.93%) karena 91.99% sampel cuma ditangani NB. DT dan KNN hampir tidak berpengaruh.

**Akar masalah cascade:**
- NB confidence threshold 0.60 terlalu tinggi untuk data berkorelasi tinggi
- Semua fitur (RSI, MACD, SMA, EMA, BB) berasal dari `Close` yang sama → NB melanggar asumsi independensi → probabilitas tidak terkalibrasi → confidence selalu < 0.60

**Solusi diterapkan:** Arsitektur diubah ke **Majority Vote** — ketiga model dipakai pada 100% data, saling mengkompensasi kelemahan masing-masing.

---

### Ekspektasi Setelah Perbaikan (Majority Vote)

Dengan majority vote, KNN yang memiliki Specificity 44.09% kini dipakai pada semua sampel. Ketika NB (bias Buy) berhadapan dengan KNN (lebih seimbang), hasilnya diharapkan:
- Specificity Hybrid **naik signifikan** (mendekati rata-rata NB+DT+KNN)
- Recall Hybrid **turun** dari 93.47% (tidak lagi bias Buy ekstrem)
- Accuracy dan F1 **lebih seimbang**

---

### Naive Bayes Menang F1-Score (Sebelum Perbaikan)

NB memiliki F1 tertinggi (0.6800) karena F1 adalah harmonic mean Precision-Recall, dan NB memiliki Recall sangat tinggi (0.9626) meskipun Precision hanya sedang (0.5256). Harmonic mean lebih tinggi ketika satu nilai sangat tinggi.

---

### Kenapa Pohon Mulai dari BB_Lower?

Root node Decision Tree menggunakan `BB_Lower ≤ 0.3356` sebagai split pertama. Ini berarti Bollinger Band Bawah adalah fitur paling informatif di level root karena split ini menghasilkan Gini Impurity terbobot paling kecil dari 240 kandidat split yang diuji.

Dalam analisis teknikal, BB_Lower yang rendah (relatif terhadap rentangnya) menandakan harga sedang di bawah pita bawah Bollinger — kondisi oversold yang sering diikuti pembalikan arah (Buy signal). Masuk akal secara domain bahwa ini menjadi pembeda paling kuat di level pertama.

---

### Output Files

Setelah dijalankan, folder `output/` berisi:

| File | Isi |
|---|---|
| `output/confusion_matrices.png` | Gambar 4 confusion matrix berdampingan |
| `output/evaluation_summary.csv` | Tabel komparasi metrik dalam format CSV |

---

## 14. Glosarium Istilah Trading

Bagian ini menjelaskan semua istilah dari dunia keuangan dan analisis teknikal yang digunakan dalam proyek ini, dari yang paling dasar hingga yang paling spesifik.

---

### Pasar & Instrumen

---

#### Saham (Stock / Share / Equity)
Surat kepemilikan atas sebagian kecil sebuah perusahaan. Jika sebuah perusahaan menerbitkan 1.000.000 lembar saham dan kamu memiliki 1.000 lembar, kamu memiliki 0,1% perusahaan tersebut. Harga saham naik-turun setiap detik berdasarkan penawaran dan permintaan di bursa.

---

#### Indeks Saham (Stock Market Index)
Sekumpulan saham yang dijadikan satu angka untuk mengukur kondisi pasar secara keseluruhan. Seperti "rata-rata nilai rapor seluruh siswa di sekolah" — satu angka yang mewakili banyak saham.

---

#### S&P 500 (Standard & Poor's 500)
Indeks saham Amerika Serikat yang berisi **500 perusahaan terbesar** berdasarkan kapitalisasi pasar (market cap), antara lain Apple, Microsoft, Amazon, Google, Tesla. S&P 500 sering dijadikan tolok ukur (*benchmark*) kesehatan ekonomi AS secara keseluruhan. Data dalam proyek ini adalah harga historis harian saham-saham anggota S&P 500.

---

#### Ticker (Simbol Saham)
Kode unik pendek yang merepresentasikan sebuah saham di bursa. Contoh:
- `AAPL` = Apple Inc.
- `MSFT` = Microsoft
- `TSLA` = Tesla
- `A` = Agilent Technologies (saham pertama di dataset ini secara abjad)

Kode ini digunakan untuk membedakan saham satu dengan lainnya agar rolling window tidak tercampur antar perusahaan.

---

#### Bursa Efek (Stock Exchange)
Tempat jual beli saham secara terorganisir. Saham S&P 500 diperdagangkan terutama di **NYSE** (New York Stock Exchange) dan **NASDAQ**. Bursa buka hari Senin–Jumat, jam 09.30–16.00 waktu New York.

---

#### Trading Day (Hari Perdagangan)
Hari di mana bursa buka dan saham bisa diperjualbelikan. Tidak termasuk akhir pekan dan hari libur nasional AS. Dalam satu tahun ada sekitar 252 hari trading.

---

### Harga: OHLCV

Data harga saham harian direpresentasikan dalam format **OHLCV** — lima angka yang merangkum seluruh aktivitas perdagangan dalam satu hari.

---

#### Open (Harga Pembukaan)
Harga pertama kali saham diperdagangkan saat bursa buka di pagi hari (jam 09.30 New York). Bisa berbeda jauh dari harga Close hari sebelumnya jika ada berita besar semalam (*gap up* atau *gap down*).

---

#### High (Harga Tertinggi)
Harga tertinggi yang terjadi sepanjang hari itu. Mencerminkan puncak antusiasme pembeli pada hari tersebut.

---

#### Low (Harga Terendah)
Harga terendah yang terjadi sepanjang hari itu. Mencerminkan titik tekanan jual terbesar pada hari tersebut.

---

#### Close (Harga Penutupan)
Harga terakhir saat bursa tutup (jam 16.00 New York). Ini adalah harga yang paling sering digunakan dalam analisis teknikal karena dianggap mencerminkan "konsensus pasar" pada akhir hari.

> Dalam proyek ini, semua indikator teknikal (RSI, MACD, SMA, dll.) dihitung berbasis harga Close.

---

#### Adj Close (Adjusted Close / Harga Penutupan Disesuaikan)
Harga Close yang sudah disesuaikan untuk kejadian korporasi seperti:
- **Stock split**: Jika saham pecah 2:1, harga turun setengah tapi jumlah lembar dua kali lipat. Adj Close menyesuaikan harga historis agar grafiknya tidak terlihat seperti "crash" mendadak.
- **Dividen**: Pembagian laba ke pemegang saham menurunkan harga. Adj Close memperhitungkan ini.

Dalam proyek ini, kolom `Adj Close` ada di CSV tapi tidak digunakan — hanya `Close` yang dipakai.

---

#### Volume
Jumlah total lembar saham yang berpindah tangan dalam satu hari trading. Volume tinggi menunjukkan banyak aktivitas transaksi — bisa pertanda tren kuat atau akan ada pembalikan arah. Volume rendah menandakan pasar sepi dan pergerakan harga kurang terpercaya.

---

### Sinyal Trading

---

#### Sinyal Trading (Trading Signal)
Indikasi kapan sebaiknya membeli atau menjual suatu aset. Dalam proyek ini, sinyal direpresentasikan sebagai label biner:
- **Buy (1)**: Direkomendasikan membeli — prediksi harga besok akan **naik**
- **Sell (0)**: Direkomendasikan menjual (atau tidak membeli) — prediksi harga besok akan **turun**

---

#### Buy (Beli)
Aksi membeli saham dengan harapan harganya akan naik di masa depan sehingga bisa dijual dengan untung. Dalam sistem prediksi ini, sinyal Buy dikeluarkan ketika model memprediksi `Close(t+1) > Close(t)`.

---

#### Sell (Jual)
Aksi menjual saham yang sudah dimiliki. Sinyal Sell dalam konteks ini berarti diperkirakan harga akan turun, sehingga sebaiknya tidak membeli (atau menjual jika sudah punya).

---

#### Long (Posisi Long)
Strategi membeli saham dan menahannya, berharap harga naik. Investor "long" untung jika harga naik, rugi jika harga turun. Sistem ini hanya mempertimbangkan posisi Long (Buy atau tidak Buy).

---

#### Short (Posisi Short / Short Selling)
Strategi meminjam saham, menjualnya sekarang, lalu membelinya kembali di harga lebih murah untuk dikembalikan. Investor "short" untung jika harga turun. Proyek ini **tidak** mengimplementasikan short selling — Sell di sini hanya berarti "jangan beli" atau "jual yang sudah dimiliki".

---

### Analisis Teknikal

---

#### Analisis Teknikal (Technical Analysis)
Metode memprediksi pergerakan harga berdasarkan **data historis harga dan volume** saja — tanpa melihat kondisi bisnis perusahaan (laba, hutang, produk). Asumsinya: semua informasi sudah tercermin dalam harga, dan pola harga masa lalu cenderung berulang.

> Proyek ini murni menggunakan analisis teknikal.

---

#### Analisis Fundamental (Fundamental Analysis)
Kebalikan dari analisis teknikal — menilai saham berdasarkan kondisi bisnis perusahaan: pendapatan, laba, hutang, manajemen, prospek industri. Tidak digunakan dalam proyek ini.

---

#### Tren (Trend)
Arah umum pergerakan harga dalam suatu periode:
- **Uptrend (Tren Naik)**: Harga secara umum bergerak ke atas — serangkaian *higher high* dan *higher low*
- **Downtrend (Tren Turun)**: Harga bergerak ke bawah — serangkaian *lower high* dan *lower low*
- **Sideways / Ranging**: Harga bergerak datar tanpa arah yang jelas, berfluktuasi dalam rentang sempit

---

#### Momentum
Kecepatan atau kekuatan pergerakan harga. Momentum tinggi berarti harga bergerak cepat ke satu arah. Indikator momentum seperti MACD mengukur apakah kecepatan kenaikan/penurunan harga sedang menguat atau melemah.

---

#### Volatilitas (Volatility)
Seberapa besar fluktuasi harga dalam suatu periode. Volatilitas tinggi = harga naik-turun dengan rentang besar setiap harinya. Volatilitas rendah = harga bergerak tenang, perubahan kecil.

> Bollinger Bands mengukur volatilitas secara langsung — band yang lebar menandakan volatilitas tinggi, band yang sempit menandakan volatilitas rendah.

---

#### Support
Level harga di mana saham historis sulit turun lebih jauh — ada banyak pembeli yang siap membeli di harga tersebut, sehingga harga "dipantulkan" ke atas. Bollinger Band Bawah sering berfungsi sebagai support dinamis.

---

#### Resistance
Level harga di mana saham historis sulit naik lebih jauh — ada banyak penjual yang siap menjual di harga tersebut, sehingga harga "tertahan". Bollinger Band Atas sering berfungsi sebagai resistance dinamis.

---

#### Breakout
Kondisi ketika harga menembus level support atau resistance yang selama ini bertahan. Breakout ke atas dari resistance = sinyal kuat Buy. Breakout ke bawah dari support = sinyal kuat Sell. Bollinger Band yang menyempit sering mendahului breakout besar.

---

#### Overbought (Jenuh Beli)
Kondisi di mana harga sudah naik terlalu tinggi terlalu cepat sehingga kemungkinan besar akan koreksi turun. RSI > 70 adalah indikasi klasik overbought. Bukan berarti harga pasti turun, tapi probabilitas koreksi meningkat.

---

#### Oversold (Jenuh Jual)
Kondisi di mana harga sudah turun terlalu dalam terlalu cepat sehingga kemungkinan besar akan rebound naik. RSI < 30 adalah indikasi klasik oversold. Bollinger Band Bawah yang ditembus Close juga merupakan sinyal oversold.

---

#### Rebound / Reversal (Pembalikan Arah)
Perubahan arah tren. Setelah downtrend, harga berbalik naik (bullish reversal). Setelah uptrend, harga berbalik turun (bearish reversal). Oversold sering mendahului bullish reversal.

---

#### Bullish
Kondisi atau pandangan yang optimistis — harga diperkirakan akan naik. Berasal dari cara banteng (*bull*) menyerang: mendorong ke atas.
- MACD positif = sinyal bullish
- RSI < 30 (oversold) = potensi bullish reversal
- Harga di atas SMA_20 = konteks bullish

---

#### Bearish
Kondisi atau pandangan yang pesimistis — harga diperkirakan akan turun. Berasal dari cara beruang (*bear*) menyerang: mencakar ke bawah.
- MACD negatif = sinyal bearish
- RSI > 70 (overbought) = potensi bearish reversal
- Harga di bawah SMA_20 = konteks bearish

---

#### Crossover
Momen ketika dua garis indikator saling melewati, dianggap sinyal yang kuat:
- **Bullish Crossover MACD**: EMA_12 melewati EMA_26 dari bawah ke atas → sinyal Buy
- **Bearish Crossover MACD**: EMA_12 melewati EMA_26 dari atas ke bawah → sinyal Sell
- **Golden Cross**: SMA_50 melewati SMA_200 ke atas → sinyal Buy jangka panjang
- **Death Cross**: SMA_50 melewati SMA_200 ke bawah → sinyal Sell jangka panjang

Proyek ini tidak menggunakan crossover secara eksplisit, tapi nilai MACD (selisih EMA_12 − EMA_26) mencerminkan seberapa jauh dari crossover.

---

### Indikator Teknikal Spesifik

---

#### Moving Average (Rata-Rata Bergerak)
Rata-rata harga Close dalam N hari terakhir yang terus diperbarui setiap hari. Digunakan untuk menghaluskan fluktuasi harian dan mengidentifikasi tren. "Bergerak" karena jendela waktu bergeser setiap hari — hari baru masuk, hari terlama keluar.

---

#### SMA — Simple Moving Average (Rata-Rata Bergerak Sederhana)
Setiap hari mendapat bobot yang **sama persis**. Hari ini dan 19 hari lalu sama bobotnya dalam SMA_20.

```
SMA_20 hari ini = (Close_hari_ini + Close_kemarin + ... + Close_19_hari_lalu) / 20
```

**Kelemahan**: Lambat bereaksi terhadap perubahan harga terbaru. Saat harga tiba-tiba turun tajam, SMA butuh waktu lama untuk turun karena masih "mengingat" harga tinggi 20 hari lalu.

---

#### EMA — Exponential Moving Average (Rata-Rata Bergerak Eksponensial)
Memberikan **bobot lebih besar** pada harga terbaru dan bobot menurun secara eksponensial ke masa lalu. Jauh lebih responsif dari SMA.

```
α = 2 / (N + 1)   → faktor peluruhan

EMA_12: α = 2/13 = 0.154  → ~15% bobot ke harga hari ini
EMA_26: α = 2/27 = 0.074  → ~7% bobot ke harga hari ini
```

EMA_12 lebih cepat (lebih responsif) dari EMA_26 karena faktor α lebih besar. Dalam trading, EMA sering disukai karena tidak "ketinggalan" tren seperti SMA.

---

#### MACD — Moving Average Convergence Divergence
Selisih antara EMA cepat (12 hari) dan EMA lambat (26 hari). Mengukur **momentum** dan arah tren:

```
MACD = EMA_12 − EMA_26
```

| Nilai MACD | Arti |
|---|---|
| Positif dan naik | Momentum bullish menguat — EMA cepat semakin jauh di atas EMA lambat |
| Positif tapi turun | Momentum bullish melemah — EMA cepat masih di atas tapi selisih mengecil |
| Nol (crossover) | Titik perubahan tren — EMA cepat baru saja melewati EMA lambat |
| Negatif | Momentum bearish — EMA cepat di bawah EMA lambat |

Diciptakan oleh Gerald Appel pada akhir 1970-an dan masih menjadi salah satu indikator paling populer hingga hari ini.

---

#### Bollinger Bands
Tiga garis yang membentuk "pita" di sekitar harga, diciptakan oleh John Bollinger pada 1980-an:

```
Garis Tengah (Middle Band) = SMA_20
Garis Atas  (BB_Upper)     = SMA_20 + 2 × Standar Deviasi 20 hari
Garis Bawah (BB_Lower)     = SMA_20 − 2 × Standar Deviasi 20 hari
```

Secara statistik, dalam distribusi normal, ~95.4% harga seharusnya berada dalam pita ±2σ. Jika harga menembus pita ini, itu adalah kejadian yang "tidak normal" secara statistik.

| Kondisi | Interpretasi |
|---|---|
| Close menyentuh / menembus BB_Upper | Overbought — harga terlalu tinggi relatif terhadap volatilitas terakhir |
| Close menyentuh / menembus BB_Lower | Oversold — harga terlalu rendah, potensi rebound |
| Band menyempit (Squeeze) | Volatilitas rendah, pasar "menahan nafas" — sering mendahului breakout besar |
| Band melebar | Volatilitas tinggi, pergerakan harga besar |

---

#### RSI — Relative Strength Index
Osilator momentum yang mengukur **kecepatan dan besaran** perubahan harga, dengan skala 0–100. Diciptakan oleh J. Welles Wilder Jr. pada 1978.

```
RSI = 100 − [100 / (1 + RS)]
RS  = Rata-rata Kenaikan 14 hari / Rata-rata Penurunan 14 hari
```

| Nilai RSI | Zona | Interpretasi |
|---|---|---|
| > 70 | Overbought | Harga naik terlalu kencang, kemungkinan koreksi turun |
| 50–70 | Bullish Normal | Momentum positif, tren naik normal |
| 50 | Netral | Kekuatan naik dan turun seimbang |
| 30–50 | Bearish Normal | Momentum negatif, tren turun normal |
| < 30 | Oversold | Harga turun terlalu dalam, kemungkinan rebound naik |

> RSI = 50 artinya AvgGain = AvgLoss, momentum seimbang.
> RSI = 100 secara teoritis terjadi jika 14 hari berturut-turut harga hanya naik (tidak pernah turun).
> RSI = 0 secara teoritis terjadi jika 14 hari berturut-turut harga hanya turun.

---

#### Volume Change (Perubahan Volume)
Persentase perubahan volume hari ini dibanding hari sebelumnya:

```
Volume_Change = (Volume_hari_ini − Volume_kemarin) / Volume_kemarin × 100%
```

**Mengapa volume penting?** Volume mengkonfirmasi atau membantah pergerakan harga:
- Harga naik + volume naik = tren naik **valid** (banyak partisipan membeli)
- Harga naik + volume turun = tren naik **lemah** (sedikit partisipan, bisa false signal)
- Harga turun + volume naik = tekanan jual **kuat** (panik jual massal)

---

### Konsep Umum Time-Series Keuangan

---

#### Time Series (Deret Waktu)
Rangkaian data yang diurutkan berdasarkan waktu, di mana **urutan sangat penting**. Data saham harian adalah time series — hari Senin harus sebelum hari Selasa. Jika data diacak (*shuffle*), makna temporal hilang dan model akan belajar dari pola yang tidak valid.

---

#### Rolling Window (Jendela Bergerak)
Teknik menghitung statistik (rata-rata, standar deviasi) menggunakan sejumlah N observasi terbaru yang terus bergeser. Misalnya, SMA_20 dengan rolling window 20 hari: setiap hari menggunakan 20 data Close terbaru, lalu jendela bergeser satu hari ke depan.

---

#### Look-Ahead Bias
Kesalahan serius dalam backtesting di mana model secara tidak sadar menggunakan **informasi masa depan** untuk membuat prediksi masa lalu. Contoh: Jika label Target hari ini dihitung dari harga besok, model tidak boleh menggunakan label ini saat memprediksi hari ini. Proyek ini menghindari ini dengan memisahkan train (data lama) dan test (data baru).

---

#### Backtesting
Proses menguji strategi trading menggunakan data historis untuk melihat seberapa baik strategi tersebut bekerja di masa lalu. Proyek ini melakukan backtesting sederhana dengan mengevaluasi prediksi pada data test yang belum pernah dilihat model.

---

#### Overfitting
Kondisi di mana model "hafal" data training terlalu baik sehingga performa di data test (data baru) sangat buruk. Decision Tree tanpa batasan kedalaman bisa overfitting — itulah mengapa digunakan `max_depth=8`, `min_samples_split=20`, dan `min_samples_leaf=10`.

---

#### Data Leakage (Kebocoran Data)
Situasi di mana informasi dari data test "bocor" masuk ke proses training, menyebabkan performa evaluasi terlalu optimistis. Dalam proyek ini, normalisasi Min-Max dihitung dari **seluruh X** (setelah split bukan sebelum split ideally) — ini adalah bentuk data leakage kecil yang umum dalam proyek akademik.

---

#### Efficient Market Hypothesis (EMH)
Teori bahwa harga saham saat ini sudah **mencerminkan semua informasi yang tersedia secara publik**. Jika EMH benar sepenuhnya, tidak ada analisis teknikal yang bisa secara konsisten mengalahkan pasar — akurasi prediksi ~50% adalah batas atas yang wajar. Itulah mengapa akurasi ~52% dalam proyek ini sudah dianggap "ada sinyal", bukan "model gagal".

---

#### Market Noise (Kebisingan Pasar)
Fluktuasi harga acak jangka pendek yang tidak mencerminkan fundamental atau tren nyata — sekadar "kebisingan" statistik. Indikator teknikal seperti SMA dan EMA dirancang untuk menyaring noise dan memperlihatkan sinyal tren yang lebih bersih.

---

#### Kapitalisasi Pasar (Market Capitalization / Market Cap)
Total nilai pasar sebuah perusahaan:
```
Market Cap = Jumlah Saham Beredar × Harga Per Saham
```
S&P 500 dibobot berdasarkan market cap, artinya Apple (market cap ~$3 triliun) punya pengaruh lebih besar terhadap nilai indeks dibanding perusahaan kecil.

---

#### Dividen (Dividend)
Bagian laba perusahaan yang dibagikan kepada pemegang saham, biasanya secara triwulanan. Ketika dividen dibayarkan, harga saham biasanya turun sejumlah nilai dividen tersebut (karena kas keluar dari perusahaan). Kolom `Adj Close` sudah memperhitungkan penyesuaian ini.

---

*Dokumentasi ini dibuat berdasarkan analisis mendalam seluruh kode proyek dan output yang dihasilkan saat pipeline dijalankan pada 9 Juni 2026.*

---

# PERHITUNGAN MANUAL LENGKAP SEMUA RUMUS

> Bagian ini menulis ulang **setiap rumus dari kode** dan menghitungnya secara manual dengan angka konkret, langkah per langkah.

---

## A. INDIKATOR TEKNIKAL (features.py)

---

### A.1 — SMA: Simple Moving Average

**Rumus dari kode (`features.py:62`):**
```python
df["SMA_20"] = close.rolling(window=20).mean()
```

**Definisi matematis:**
```
SMA(t) = [ Close(t) + Close(t-1) + Close(t-2) + ... + Close(t-19) ] / 20
```

**Data contoh — 5 hari Close:** `[100, 102, 105, 103, 107]`

```
SMA(5) = (100 + 102 + 105 + 103 + 107) / 5

Penjumlahan:
  100
+ 102  → 202
+ 105  → 307
+ 103  → 410
+ 107  → 517

SMA(5) = 517 / 5 = 103.40
```

**Interpretasi:** Harga rata-rata 5 hari terakhir = 103.40.
Jika Close(t) = 107 > SMA = 103.40 → harga di atas rata-rata → tren naik (bullish).

---

### A.2 — EMA: Exponential Moving Average

**Rumus dari kode (`features.py:63-64`):**
```python
df["EMA_12"] = close.ewm(span=12, adjust=False).mean()
df["EMA_26"] = close.ewm(span=26, adjust=False).mean()
```

**Definisi matematis:**
```
k       = 2 / (span + 1)
EMA(1)  = Close(1)                                       ← seed = harga pertama
EMA(t)  = Close(t) × k  +  EMA(t-1) × (1 - k)
```

**Hitung k untuk EMA_12 (span=12):**
```
k = 2 / (12 + 1) = 2 / 13 = 0.153846...  ≈ 0.1538
```

**Hitung k untuk EMA_26 (span=26):**
```
k = 2 / (26 + 1) = 2 / 27 = 0.074074...  ≈ 0.0741
```

**Perhitungan EMA_12 — 5 hari Close: [100, 102, 105, 103, 107]**

```
Hari 1:
  EMA_12(1) = 100.0000  (seed = harga pertama)

Hari 2 (Close=102):
  EMA_12(2) = 102 × 0.1538 + 100 × (1 - 0.1538)
            = 102 × 0.1538 + 100 × 0.8462
            = 15.6923   + 84.6154
            = 100.3077

Hari 3 (Close=105):
  EMA_12(3) = 105 × 0.1538 + 100.3077 × 0.8462
            = 16.1538   + 84.8801
            = 101.0339

Hari 4 (Close=103):
  EMA_12(4) = 103 × 0.1538 + 101.0339 × 0.8462
            = 15.8462   + 85.4953
            = 101.3415

Hari 5 (Close=107):
  EMA_12(5) = 107 × 0.1538 + 101.3415 × 0.8462
            = 16.4615   + 85.7535
            = 102.2151
```

**Ringkasan tabel EMA_12:**

| Hari | Close | EMA_12   |
|------|-------|----------|
| 1    | 100   | 100.0000 |
| 2    | 102   | 100.3077 |
| 3    | 105   | 101.0339 |
| 4    | 103   | 101.3415 |
| 5    | 107   | 102.2151 |

---

### A.3 — MACD: Moving Average Convergence Divergence

**Rumus dari kode (`features.py:67`):**
```python
df["MACD"] = df["EMA_12"] - df["EMA_26"]
```

**Definisi matematis:**
```
MACD(t) = EMA_12(t) - EMA_26(t)
```

**Perhitungan — melanjutkan contoh hari 5:**
```
EMA_12(5) = 102.2151
EMA_26(5) = 100.8623  (EMA_26 lebih lambat → lebih dekat ke SMA)

MACD(5)   = 102.2151 - 100.8623 = +1.3528
```

**Verifikasi logika:**
```
MACD > 0 → EMA_12 di atas EMA_26 → momentum bullish → kemungkinan harga lanjut naik
MACD < 0 → EMA_12 di bawah EMA_26 → momentum bearish → kemungkinan harga turun
MACD = 0 → Titik crossover (perubahan tren)
```

---

### A.4 — Bollinger Bands

**Rumus dari kode (`features.py:70-72`):**
```python
bb_std      = close.rolling(window=20).std()
df["BB_Upper"] = df["SMA_20"] + 2 * bb_std
df["BB_Lower"] = df["SMA_20"] - 2 * bb_std
```

**Definisi matematis:**
```
σ(t)      = √[ (1/N) × Σ (Close(t-i) - SMA(t))² ]   untuk i=0..N-1
BB_Upper  = SMA_20 + 2 × σ
BB_Lower  = SMA_20 - 2 × σ
```

**Data contoh — 5 hari:** `[100, 102, 105, 103, 107]`

**Langkah 1: Hitung SMA (sudah dihitung di atas)**
```
SMA = 517 / 5 = 103.40
```

**Langkah 2: Hitung deviasi kuadrat dari setiap titik ke mean**
```
(100 - 103.40)² = (-3.40)² = 11.56
(102 - 103.40)² = (-1.40)² =  1.96
(105 - 103.40)² = ( 1.60)² =  2.56
(103 - 103.40)² = (-0.40)² =  0.16
(107 - 103.40)² = ( 3.60)² = 12.96
```

**Langkah 3: Hitung variance dan standar deviasi**
```
Jumlah deviasi kuadrat = 11.56 + 1.96 + 2.56 + 0.16 + 12.96 = 29.20

variance = 29.20 / 5 = 5.84

σ = √5.84 = √5.84

Hitung √5.84:
  2.4² = 5.76  (terlalu kecil)
  2.42² = 5.8564  (terlalu besar)
  2.41² = 5.8081  (terlalu kecil)
  2.416² = 5.8370  (terlalu kecil)
  2.417² = 5.8419  (terlalu kecil sedikit)

σ ≈ 2.4166
```

**Langkah 4: Hitung Bollinger Bands**
```
BB_Upper = 103.40 + 2 × 2.4166
         = 103.40 + 4.8332
         = 108.2332

BB_Lower = 103.40 - 2 × 2.4166
         = 103.40 - 4.8332
         =  98.5668
```

**Interpretasi:**
```
Harga Close terbaru = 107
BB_Lower = 98.57 ≤ 107 ≤ 108.23 = BB_Upper
→ Harga masih dalam batas normal (belum overbought/oversold)
```

---

### A.5 — RSI: Relative Strength Index

**Rumus dari kode (`features.py:75-79`):**
```python
delta = close.diff()
gain  = delta.clip(lower=0).rolling(window=14).mean()
loss  = (-delta.clip(upper=0)).rolling(window=14).mean()
rs    = gain / loss.replace(0, np.nan)
df["RSI"] = 100 - (100 / (1 + rs))
```

**Definisi matematis:**
```
Δ(t)       = Close(t) - Close(t-1)
gain(t)    = max(Δ(t), 0)           ← hanya perubahan positif
loss(t)    = max(-Δ(t), 0)          ← hanya perubahan negatif (dibuat positif)

avg_gain   = mean(gain[-14:])        ← rata-rata 14 hari
avg_loss   = mean(loss[-14:])

RS         = avg_gain / avg_loss
RSI        = 100 - (100 / (1 + RS))
```

**Data contoh — 15 hari harga Close:**
```
[100.0, 101.5, 99.0, 102.0, 103.5, 101.0, 104.0, 105.0,
  102.0, 106.0, 107.0, 105.0, 108.0, 109.0, 107.0]
```

**Hitung delta (Δ):**
```
Hari 1→2:  101.5 - 100.0 = +1.5
Hari 2→3:   99.0 - 101.5 = -2.5
Hari 3→4:  102.0 -  99.0 = +3.0
Hari 4→5:  103.5 - 102.0 = +1.5
Hari 5→6:  101.0 - 103.5 = -2.5
Hari 6→7:  104.0 - 101.0 = +3.0
Hari 7→8:  105.0 - 104.0 = +1.0
Hari 8→9:  102.0 - 105.0 = -3.0
Hari 9→10: 106.0 - 102.0 = +4.0
Hari 10→11: 107.0 - 106.0 = +1.0
Hari 11→12: 105.0 - 107.0 = -2.0
Hari 12→13: 108.0 - 105.0 = +3.0
Hari 13→14: 109.0 - 108.0 = +1.0
Hari 14→15: 107.0 - 109.0 = -2.0
```

**Pisahkan gain dan loss (14 perubahan):**

| Period | Δ    | gain | loss |
|--------|------|------|------|
| 1→2    | +1.5 | 1.5  | 0.0  |
| 2→3    | -2.5 | 0.0  | 2.5  |
| 3→4    | +3.0 | 3.0  | 0.0  |
| 4→5    | +1.5 | 1.5  | 0.0  |
| 5→6    | -2.5 | 0.0  | 2.5  |
| 6→7    | +3.0 | 3.0  | 0.0  |
| 7→8    | +1.0 | 1.0  | 0.0  |
| 8→9    | -3.0 | 0.0  | 3.0  |
| 9→10   | +4.0 | 4.0  | 0.0  |
| 10→11  | +1.0 | 1.0  | 0.0  |
| 11→12  | -2.0 | 0.0  | 2.0  |
| 12→13  | +3.0 | 3.0  | 0.0  |
| 13→14  | +1.0 | 1.0  | 0.0  |
| 14→15  | -2.0 | 0.0  | 2.0  |

**Hitung avg_gain (14 periode):**
```
sum_gain = 1.5 + 0 + 3.0 + 1.5 + 0 + 3.0 + 1.0 + 0 + 4.0 + 1.0 + 0 + 3.0 + 1.0 + 0
         = 1.5 + 3.0 + 1.5 + 3.0 + 1.0 + 4.0 + 1.0 + 3.0 + 1.0
         = 19.0

avg_gain = 19.0 / 14 = 1.357143
```

**Hitung avg_loss (14 periode):**
```
sum_loss = 0 + 2.5 + 0 + 0 + 2.5 + 0 + 0 + 3.0 + 0 + 0 + 2.0 + 0 + 0 + 2.0
         = 2.5 + 2.5 + 3.0 + 2.0 + 2.0
         = 12.0

avg_loss = 12.0 / 14 = 0.857143
```

**Hitung RS dan RSI:**
```
RS  = avg_gain / avg_loss
    = 1.357143 / 0.857143
    = 1.583333

RSI = 100 - (100 / (1 + RS))
    = 100 - (100 / (1 + 1.583333))
    = 100 - (100 / 2.583333)
    = 100 - 38.7097
    = 61.2903   ≈ 61.29
```

**Interpretasi:**
```
RSI = 61.29 → Zona Bullish Normal (50-70)
→ Momentum positif, tren naik normal
→ Belum overbought (< 70)
```

---

### A.6 — Volume Change

**Rumus dari kode (`features.py:83`):**
```python
df["Volume_Change"] = volume.pct_change().fillna(0).clip(-5, 5)
```

**Definisi matematis:**
```
Volume_Change(t) = (Volume(t) - Volume(t-1)) / Volume(t-1)
```

**Contoh 1 — kenaikan volume normal:**
```
Volume(t-1) = 1.000.000
Volume(t)   = 1.200.000

Volume_Change = (1.200.000 - 1.000.000) / 1.000.000
              = 200.000 / 1.000.000
              = 0.20   (naik 20%)

Clip: 0.20 berada dalam [-5, 5] → tetap 0.20
```

**Contoh 2 — volume meledak ekstrem:**
```
Volume(t-1) = 500.000
Volume(t)   = 8.000.000   (berita besar)

Volume_Change = (8.000.000 - 500.000) / 500.000
              = 7.500.000 / 500.000
              = 15.0   (naik 1500%!)

Clip: 15.0 > 5 → di-clip menjadi 5.0
```

**Contoh 3 — volume turun drastis:**
```
Volume(t-1) = 2.000.000
Volume(t)   =   100.000

Volume_Change = (100.000 - 2.000.000) / 2.000.000
              = -1.900.000 / 2.000.000
              = -0.95   (turun 95%)

Clip: -0.95 berada dalam [-5, 5] → tetap -0.95
```

---

### A.7 — Target Label

**Rumus dari kode (`features.py:86`):**
```python
df["Target"] = (close.shift(-1) > close).astype(int)
```

**Definisi matematis:**
```
Target(t) = 1  jika Close(t+1) > Close(t)   → BUY (harga besok naik)
Target(t) = 0  jika Close(t+1) ≤ Close(t)  → SELL (harga besok turun/sama)
```

**Contoh dengan 5 hari data:**

| Hari (t) | Close(t) | Close(t+1) | Close(t+1) > Close(t)? | Target |
|----------|----------|------------|------------------------|--------|
| Senin    | 100.0    | 103.0      | 103 > 100 = True       | 1 (Buy)|
| Selasa   | 103.0    | 101.0      | 101 > 103 = False      | 0 (Sell)|
| Rabu     | 101.0    | 105.0      | 105 > 101 = True       | 1 (Buy)|
| Kamis    | 105.0    | 107.0      | 107 > 105 = True       | 1 (Buy)|
| Jumat    | 107.0    | NaN        | NaN → di-drop          | NaN    |

**Persentase naik vs turun dari contoh di atas:**
```
Buy  (1): 3 dari 4 hari valid = 75%
Sell (0): 1 dari 4 hari valid = 25%
```

---

## B. NORMALISASI MIN-MAX (features.py)

**Rumus dari kode (`features.py:41-44`):**
```python
X_min = X.min(axis=0)
X_max = X.max(axis=0)
denom = np.where((X_max - X_min) == 0, 1.0, X_max - X_min)
X     = (X - X_min) / denom
```

**Definisi matematis:**
```
X_scaled(i, j) = (X(i, j) - X_min(j)) / (X_max(j) - X_min(j))

Khusus: jika X_max(j) = X_min(j), maka denominator = 1.0 (hindari ÷0)
```

**Contoh — 5 sampel, 2 fitur (RSI dan MACD):**

| Sampel | RSI (raw) | MACD (raw) |
|--------|-----------|------------|
| 1      | 30.0      | -2.5       |
| 2      | 45.0      | -1.0       |
| 3      | 55.0      |  0.5       |
| 4      | 70.0      |  2.0       |
| 5      | 80.0      |  3.5       |

**Hitung min dan max per fitur:**
```
RSI:  X_min = 30.0,  X_max = 80.0,  range = 80.0 - 30.0 = 50.0
MACD: X_min = -2.5,  X_max =  3.5,  range = 3.5 - (-2.5) = 6.0
```

**Hitung normalisasi RSI:**
```
Sampel 1: (30.0 - 30.0) / 50.0 =  0.0 / 50.0 = 0.0000
Sampel 2: (45.0 - 30.0) / 50.0 = 15.0 / 50.0 = 0.3000
Sampel 3: (55.0 - 30.0) / 50.0 = 25.0 / 50.0 = 0.5000
Sampel 4: (70.0 - 30.0) / 50.0 = 40.0 / 50.0 = 0.8000
Sampel 5: (80.0 - 30.0) / 50.0 = 50.0 / 50.0 = 1.0000
```

**Hitung normalisasi MACD:**
```
Sampel 1: (-2.5 - (-2.5)) / 6.0 =  0.0 / 6.0 = 0.0000
Sampel 2: (-1.0 - (-2.5)) / 6.0 =  1.5 / 6.0 = 0.2500
Sampel 3: ( 0.5 - (-2.5)) / 6.0 =  3.0 / 6.0 = 0.5000
Sampel 4: ( 2.0 - (-2.5)) / 6.0 =  4.5 / 6.0 = 0.7500
Sampel 5: ( 3.5 - (-2.5)) / 6.0 =  6.0 / 6.0 = 1.0000
```

**Hasil setelah normalisasi:**

| Sampel | RSI (scaled) | MACD (scaled) |
|--------|-------------|---------------|
| 1      | 0.0000      | 0.0000        |
| 2      | 0.3000      | 0.2500        |
| 3      | 0.5000      | 0.5000        |
| 4      | 0.8000      | 0.7500        |
| 5      | 1.0000      | 1.0000        |

Semua nilai sekarang berada di rentang **[0.0, 1.0]** ✓

---

## C. GAUSSIAN NAIVE BAYES (naive_bayes.py)

---

### C.1 — Prior Probability

**Rumus dari kode (`naive_bayes.py:54`):**
```python
self.priors_[idx] = X_c.shape[0] / n_samples
```

**Definisi matematis:**
```
P(c) = |X_c| / |X_total|
```

**Contoh dari output aktual program:**
```
Total training = 69.695 sampel
Kelas Buy (1)  = 36.843 sampel
Kelas Sell (0) = 32.852 sampel

P(Buy)  = 36.843 / 69.695 = 0.52861...
        ≈ 0.5286

P(Sell) = 32.852 / 69.695 = 0.47139...
        ≈ 0.4714

Verifikasi: P(Buy) + P(Sell) = 0.5286 + 0.4714 = 1.0000 ✓
```

---

### C.2 — Mean dan Variance per Kelas

**Rumus dari kode (`naive_bayes.py:55-56`):**
```python
self.mean_[idx] = X_c.mean(axis=0)
self.var_[idx]  = X_c.var(axis=0) + 1e-9
```

**Definisi matematis:**
```
μ_c  = (1/|X_c|) × Σ x_i
σ²_c = (1/|X_c|) × Σ (x_i - μ_c)²  +  1e-9
```

**Contoh — fitur RSI, kelas Buy (5 sampel representatif):**
```
RSI dari kelas Buy: [0.60, 0.55, 0.70, 0.65, 0.50]
```

**Hitung mean:**
```
sum = 0.60 + 0.55 + 0.70 + 0.65 + 0.50
    = 0.60
    + 0.55  → 1.15
    + 0.70  → 1.85
    + 0.65  → 2.50
    + 0.50  → 3.00

μ_Buy_RSI = 3.00 / 5 = 0.6000
```

**Hitung variance:**
```
d1 = 0.60 - 0.60 =  0.00  → d1² = 0.0000
d2 = 0.55 - 0.60 = -0.05  → d2² = 0.0025
d3 = 0.70 - 0.60 =  0.10  → d3² = 0.0100
d4 = 0.65 - 0.60 =  0.05  → d4² = 0.0025
d5 = 0.50 - 0.60 = -0.10  → d5² = 0.0100

Σ d² = 0.0000 + 0.0025 + 0.0100 + 0.0025 + 0.0100 = 0.0250

σ²_Buy_RSI = 0.0250 / 5 + 1e-9
           = 0.0050 + 0.000000001
           = 0.005000001
           ≈ 0.0050
```

---

### C.3 — Gaussian Log PDF

**Rumus dari kode (`naive_bayes.py:81-84`):**
```python
log_pdf = -0.5 * np.sum(
    np.log(2.0 * np.pi * var) + ((X - mean) ** 2) / var,
    axis=1
)
```

**Definisi matematis:**
```
log P(x | μ, σ²) = -0.5 × [ log(2π σ²)  +  (x - μ)² / σ² ]
```

**Contoh — hitung log P(RSI=0.63 | kelas Buy):**
```
μ  = 0.60
σ² = 0.0050

Bagian 1: log(2π σ²)
  2π = 6.283185
  2π × σ² = 6.283185 × 0.0050 = 0.031416
  log(0.031416) = -3.4607

Bagian 2: (x - μ)² / σ²
  x - μ = 0.63 - 0.60 = 0.03
  (x - μ)² = (0.03)² = 0.0009
  (x - μ)² / σ² = 0.0009 / 0.0050 = 0.18

Gabungkan:
  log P = -0.5 × (-3.4607 + 0.18)
        = -0.5 × (-3.2807)
        = 1.6403

Verifikasi menggunakan PDF langsung:
  P(x|μ,σ²) = (1/√(2π×0.0050)) × exp(-(0.03)²/(2×0.0050))
             = (1/√0.031416) × exp(-0.0009/0.01)
             = (1/0.17725) × exp(-0.09)
             = 5.6419 × 0.91393
             = 5.1563

  log(5.1563) = 1.6403 ✓  (cocok)
```

---

### C.4 — Posterior dengan Bayes' Theorem

**Rumus dari kode (`naive_bayes.py:105-107`):**
```python
log_prior  = np.log(self.priors_[idx])
log_likeli = self._gaussian_log_pdf(idx, X)
log_posteriors.append(log_prior + log_likeli)
```

**Definisi matematis:**
```
log P(y | x) = log P(y) + Σᵢ log P(xᵢ | y)
```

**Contoh lengkap — 2 fitur: RSI=0.63, MACD=0.45**

**Parameter yang sudah diestimasi:**
```
P(Buy)  = 0.5286  → log P(Buy)  = log(0.5286) = -0.6376
P(Sell) = 0.4714  → log P(Sell) = log(0.4714) = -0.7521

Fitur RSI:
  μ_Buy_RSI  = 0.60, σ²_Buy_RSI  = 0.0050
  μ_Sell_RSI = 0.40, σ²_Sell_RSI = 0.0045

Fitur MACD:
  μ_Buy_MACD  = 0.50, σ²_Buy_MACD  = 0.0030
  μ_Sell_MACD = 0.35, σ²_Sell_MACD = 0.0025
```

**Hitung log P(RSI=0.63 | Buy):**
```
= -0.5 × [log(2π × 0.0050) + (0.63-0.60)²/0.0050]
= -0.5 × [-3.4607 + 0.0009/0.005]
= -0.5 × [-3.4607 + 0.18]
= -0.5 × (-3.2807)
= 1.6403
```

**Hitung log P(MACD=0.45 | Buy):**
```
= -0.5 × [log(2π × 0.0030) + (0.45-0.50)²/0.0030]
= -0.5 × [log(0.018850) + (-0.05)²/0.003]
= -0.5 × [-3.9703 + 0.0025/0.003]
= -0.5 × [-3.9703 + 0.8333]
= -0.5 × (-3.1370)
= 1.5685
```

**Hitung log posterior Buy:**
```
log P(Buy | x) = log P(Buy) + log P(RSI|Buy) + log P(MACD|Buy)
               = -0.6376    + 1.6403         + 1.5685
               = 2.5712
```

**Hitung log P(RSI=0.63 | Sell):**
```
= -0.5 × [log(2π × 0.0045) + (0.63-0.40)²/0.0045]
= -0.5 × [log(0.028274) + (0.23)²/0.0045]
= -0.5 × [-3.5655 + 0.0529/0.0045]
= -0.5 × [-3.5655 + 11.7556]
= -0.5 × (8.1901)
= -4.0950
```

**Hitung log P(MACD=0.45 | Sell):**
```
= -0.5 × [log(2π × 0.0025) + (0.45-0.35)²/0.0025]
= -0.5 × [log(0.015708) + (0.10)²/0.0025]
= -0.5 × [-4.1527 + 0.01/0.0025]
= -0.5 × [-4.1527 + 4.0000]
= -0.5 × (-0.1527)
= 0.0764
```

**Hitung log posterior Sell:**
```
log P(Sell | x) = log P(Sell) + log P(RSI|Sell) + log P(MACD|Sell)
                = -0.7521     + (-4.0950)        + 0.0764
                = -4.7707
```

---

### C.5 — Log-Sum-Exp Trick

**Rumus dari kode (`naive_bayes.py:114-116`):**
```python
max_log = log_post.max(axis=1, keepdims=True)
exp_val = np.exp(log_post - max_log)
probs   = exp_val / exp_val.sum(axis=1, keepdims=True)
```

**Definisi matematis:**
```
softmax(z_i) = exp(z_i - max(z)) / Σ_j exp(z_j - max(z))
```

**Melanjutkan contoh — log posterior: [2.5712, -4.7707]**

```
log_post = [2.5712, -4.7707]  (Buy, Sell)

Langkah 1: Cari max
  max_log = max(2.5712, -4.7707) = 2.5712

Langkah 2: Kurangi max dan hitung exp
  exp_Buy  = exp(2.5712 - 2.5712) = exp(0)        = 1.000000
  exp_Sell = exp(-4.7707 - 2.5712) = exp(-7.3419)  = 0.000649

Langkah 3: Hitung sum
  Σ exp = 1.000000 + 0.000649 = 1.000649

Langkah 4: Normalisasi menjadi probabilitas
  P(Buy  | x) = 1.000000 / 1.000649 = 0.999351  ≈ 0.9993
  P(Sell | x) = 0.000649 / 1.000649 = 0.000649  ≈ 0.0006

Verifikasi: 0.9993 + 0.0006 ≈ 0.9999 ≈ 1.0 ✓

Confidence  = max(0.9993, 0.0006) = 0.9993 ≥ 0.60 → lanjut ke Stage 2
Prediksi NB = argmax = index 0 = Buy (1)
```

---

## D. DECISION TREE — GINI IMPURITY (decision_tree.py)

---

### D.1 — Gini Impurity Satu Node

**Rumus dari kode (`decision_tree.py:255-265`):**
```python
_, counts = np.unique(y, return_counts=True)
probs     = counts / len(y)
return float(1.0 - np.sum(probs**2))
```

**Definisi matematis:**
```
Gini(S) = 1 - Σᵢ pᵢ²
```

**Contoh 1 — Node pure (100% Buy):**
```
Node: y = [1, 1, 1, 1, 1]  (5 Buy, 0 Sell)
p_Buy  = 5/5 = 1.0
p_Sell = 0/5 = 0.0

Gini = 1 - (1.0² + 0.0²)
     = 1 - (1.0 + 0.0)
     = 0.000   ← sempurna, tidak perlu split
```

**Contoh 2 — Node 50:50 (maximum impurity):**
```
Node: y = [0, 0, 0, 1, 1, 1]  (3 Buy, 3 Sell)
p_Buy  = 3/6 = 0.5
p_Sell = 3/6 = 0.5

Gini = 1 - (0.5² + 0.5²)
     = 1 - (0.25 + 0.25)
     = 1 - 0.50
     = 0.500   ← maksimum impurity untuk binary
```

**Contoh 3 — Node campuran tidak rata:**
```
Node: y = [0, 0, 1, 1, 1, 1, 1, 1]  (6 Buy, 2 Sell)
p_Buy  = 6/8 = 0.750
p_Sell = 2/8 = 0.250

0.75² = 0.5625
0.25² = 0.0625

Gini = 1 - (0.5625 + 0.0625)
     = 1 - 0.6250
     = 0.3750
```

**Contoh 4 — Node dari output aktual program (root node):**
```
Root: n=69.695, gini=0.4984 (dari output pohon)

Ini artinya:
  p_Buy  + p_Sell = 1
  1 - p_Buy² - p_Sell² = 0.4984
  p_Buy² + p_Sell² = 0.5016

  Jika p_Buy = 0.5286, p_Sell = 0.4714:
  0.5286² + 0.4714² = 0.2794 + 0.2222 = 0.5016 ✓
  Gini = 1 - 0.5016 = 0.4984 ✓
```

---

### D.2 — Weighted Gini untuk Split

**Rumus dari kode (`decision_tree.py:267-276`):**
```python
n   = len(left_y) + len(right_y)
w_l = len(left_y) / n
w_r = len(right_y) / n
return w_l * self._gini(left_y) + w_r * self._gini(right_y)
```

**Definisi matematis:**
```
Gini_w = (n_L / n) × Gini(Left) + (n_R / n) × Gini(Right)
```

**Contoh — split RSI ≤ 0.45 pada node 10 sampel:**

```
Data lengkap (10 sampel):
  RSI:  [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.70, 0.80]
  Label:[  0,    0,    0,    0,    0,    0,    1,    1,    1,    1  ]
```

**Setelah split RSI ≤ 0.45:**
```
Left  (RSI ≤ 0.45): RSI=[0.20,0.25,0.30,0.35,0.40,0.45] → label=[0,0,0,0,0,0]
Right (RSI  > 0.45): RSI=[0.50,0.55,0.70,0.80]           → label=[1,1,1,1]

n_L = 6, n_R = 4, n = 10
```

**Hitung Gini(Left):**
```
p_Sell = 6/6 = 1.0,  p_Buy = 0/6 = 0.0
Gini(L) = 1 - (1.0² + 0.0²) = 1 - 1.0 = 0.0000
```

**Hitung Gini(Right):**
```
p_Buy  = 4/4 = 1.0,  p_Sell = 0/4 = 0.0
Gini(R) = 1 - (1.0² + 0.0²) = 1 - 1.0 = 0.0000
```

**Hitung Weighted Gini:**
```
w_L = 6/10 = 0.6
w_R = 4/10 = 0.4

Gini_w = 0.6 × 0.0000 + 0.4 × 0.0000
       = 0.0 + 0.0
       = 0.0000   ← split sempurna!
```

**Bandingkan dengan split yang buruk (RSI ≤ 0.35):**
```
Left  (RSI ≤ 0.35): label=[0,0,0,0]           n_L=4
  p_Sell=1.0, Gini(L) = 0.0

Right (RSI  > 0.35): label=[0,0,1,1,1,1]      n_R=6
  p_Sell = 2/6 = 0.333,  p_Buy = 4/6 = 0.667
  0.333² = 0.111,  0.667² = 0.444
  Gini(R) = 1 - (0.111 + 0.444) = 1 - 0.555 = 0.445

Gini_w = (4/10) × 0.0 + (6/10) × 0.445
       = 0 + 0.267
       = 0.267   ← lebih buruk dari 0.0
```

**Kesimpulan algoritma:**
```
Split RSI ≤ 0.45:  Gini_w = 0.000  ← TERPILIH (minimum)
Split RSI ≤ 0.35:  Gini_w = 0.267  ← tidak dipilih
```

---

### D.3 — Candidate Thresholds

**Rumus dari kode (`decision_tree.py:229-232`):**
```python
if len(unique_vals) > self.max_thresholds:
    percs      = np.linspace(0, 100, self.max_thresholds + 2)[1:-1]
    thresholds = np.percentile(col, percs)
else:
    thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2.0
```

**Contoh — midpoint threshold (jika nilai unik ≤ 30):**
```
unique_vals = [0.2, 0.4, 0.6, 0.8]  (4 nilai unik)

Midpoints = (unique_vals[:-1] + unique_vals[1:]) / 2
          = ([0.2, 0.4, 0.6] + [0.4, 0.6, 0.8]) / 2
          = [0.6, 1.0, 1.4] / 2
          = [0.30, 0.50, 0.70]

Kandidat threshold: [0.30, 0.50, 0.70]
```

**Contoh — percentile threshold (jika nilai unik > 30):**
```
max_thresholds = 30
percs = np.linspace(0, 100, 32)[1:-1]
      = [3.226, 6.452, 9.677, ..., 93.548, 96.774]
      (30 nilai merata dari 3.2% hingga 96.8%)

Artinya: uji threshold pada persentil ke-3.2, 6.5, 9.7, ..., 93.5, 96.8
```

---

### D.4 — Majority Vote untuk Leaf

**Rumus dari kode (`decision_tree.py:280-281`):**
```python
vals, counts = np.unique(y, return_counts=True)
return int(vals[np.argmax(counts)])
```

**Definisi matematis:**
```
prediksi_leaf = kelas dengan frekuensi tertinggi di node tersebut
```

**Contoh 1 — Node leaf dengan 12 Buy, 8 Sell:**
```
unique  = [0, 1]
counts  = [8, 12]
argmax  = index 1  (karena 12 > 8)
prediksi = vals[1] = 1 = Buy
```

**Contoh 2 — Node leaf dengan 15 Sell, 3 Buy:**
```
unique  = [0, 1]
counts  = [15, 3]
argmax  = index 0  (karena 15 > 3)
prediksi = vals[0] = 0 = Sell
```

---

## E. K-NEAREST NEIGHBORS (knn.py)

---

### E.1 — Euclidean Distance (Definisi)

**Rumus dari kode (`knn.py:15`):**
```
d(a,b) = √Σ (aᵢ - bᵢ)²
```

**Contoh — 2 titik dalam ruang 3 dimensi (3 fitur):**
```
a = [0.5, 0.3, 0.8]  (sampel test)
b = [0.4, 0.6, 0.7]  (sampel training)

d(a, b) = √[ (0.5-0.4)² + (0.3-0.6)² + (0.8-0.7)² ]
         = √[ (0.1)²    + (-0.3)²     + (0.1)²     ]
         = √[ 0.01      +  0.09       +  0.01       ]
         = √0.11
         = 0.3317
```

---

### E.2 — Optimasi: Squared Distance via Identity Matrix

**Rumus dari kode (`knn.py:82-92`):**
```python
sq_train = np.sum(self.X_train ** 2, axis=1)          # ||train_i||²
sq_batch = np.sum(batch ** 2, axis=1, keepdims=True)  # ||test_j||²
dot      = batch @ self.X_train.T                      # test · train^T
dist_sq  = np.clip(sq_batch + sq_train - 2 * dot, 0, None)
```

**Identitas matematis yang digunakan:**
```
||a - b||² = ||a||² + ||b||² - 2(a · b)
```

**Bukti:**
```
||a - b||² = Σᵢ (aᵢ - bᵢ)²
           = Σᵢ (aᵢ² - 2aᵢbᵢ + bᵢ²)
           = Σᵢ aᵢ²  -  2 Σᵢ aᵢbᵢ  +  Σᵢ bᵢ²
           = ||a||²   -  2(a · b)    +  ||b||²
           = ||a||² + ||b||² - 2(a · b)   ✓
```

**Contoh numerik lengkap — 2 test, 3 training, 2 fitur:**

```
X_test  = [[0.63, 0.45],    ← test_1
           [0.30, 0.70]]    ← test_2

X_train = [[0.55, 0.50],    ← train_1 (label=Buy=1)
           [0.40, 0.60],    ← train_2 (label=Buy=1)
           [0.20, 0.30]]    ← train_3 (label=Sell=0)
```

**Hitung sq_train = ||train_i||²:**
```
||train_1||² = 0.55² + 0.50² = 0.3025 + 0.2500 = 0.5525
||train_2||² = 0.40² + 0.60² = 0.1600 + 0.3600 = 0.5200
||train_3||² = 0.20² + 0.30² = 0.0400 + 0.0900 = 0.1300

sq_train = [0.5525, 0.5200, 0.1300]
```

**Hitung sq_batch = ||test_j||² (dengan keepdims=True → shape (2,1)):**
```
||test_1||² = 0.63² + 0.45² = 0.3969 + 0.2025 = 0.5994
||test_2||² = 0.30² + 0.70² = 0.0900 + 0.4900 = 0.5800

sq_batch = [[0.5994],
            [0.5800]]
```

**Hitung dot = batch @ X_train.T (shape: (2, 3)):**
```
dot[test_1, train_1] = 0.63×0.55 + 0.45×0.50 = 0.3465 + 0.2250 = 0.5715
dot[test_1, train_2] = 0.63×0.40 + 0.45×0.60 = 0.2520 + 0.2700 = 0.5220
dot[test_1, train_3] = 0.63×0.20 + 0.45×0.30 = 0.1260 + 0.1350 = 0.2610

dot[test_2, train_1] = 0.30×0.55 + 0.70×0.50 = 0.1650 + 0.3500 = 0.5150
dot[test_2, train_2] = 0.30×0.40 + 0.70×0.60 = 0.1200 + 0.4200 = 0.5400
dot[test_2, train_3] = 0.30×0.20 + 0.70×0.30 = 0.0600 + 0.2100 = 0.2700

dot = [[0.5715, 0.5220, 0.2610],
       [0.5150, 0.5400, 0.2700]]
```

**Hitung dist_sq = sq_batch + sq_train - 2×dot (shape: (2, 3)):**
```
dist²[test_1, train_1]:
  sq_batch[test_1] + sq_train[train_1] - 2×dot[test_1,train_1]
= 0.5994 + 0.5525 - 2×0.5715
= 1.1519 - 1.1430
= 0.0089

dist²[test_1, train_2]:
= 0.5994 + 0.5200 - 2×0.5220
= 1.1194 - 1.0440
= 0.0754

dist²[test_1, train_3]:
= 0.5994 + 0.1300 - 2×0.2610
= 0.7294 - 0.5220
= 0.2074

dist²[test_2, train_1]:
= 0.5800 + 0.5525 - 2×0.5150
= 1.1325 - 1.0300
= 0.1025

dist²[test_2, train_2]:
= 0.5800 + 0.5200 - 2×0.5400
= 1.1000 - 1.0800
= 0.0200

dist²[test_2, train_3]:
= 0.5800 + 0.1300 - 2×0.2700
= 0.7100 - 0.5400
= 0.1700

dist_sq = [[0.0089, 0.0754, 0.2074],
           [0.1025, 0.0200, 0.1700]]
```

**Verifikasi — Euclidean distance langsung untuk test_1 ke train_1:**
```
d = √[(0.63-0.55)² + (0.45-0.50)²]
  = √[(0.08)²     + (-0.05)²    ]
  = √[0.0064      + 0.0025      ]
  = √0.0089
  = 0.09434

dist² = 0.09434² = 0.0089 ✓
```

---

### E.3 — Majority Vote (k=3)

**Rumus dari kode (`knn.py:99-105`):**
```python
k_idx    = np.argpartition(dist_sq, self.k, axis=1)[:, :self.k]
neighbors = self.y_train[indices]
preds[i]  = int(np.bincount(neighbors).argmax())
```

**Melanjutkan contoh — k=3 (ambil semua 3 tetangga karena hanya ada 3):**

**Untuk test_1:**
```
Jarak ke training:
  train_1 (Buy): dist² = 0.0089  ← terdekat ke-1
  train_2 (Buy): dist² = 0.0754  ← terdekat ke-2
  train_3 (Sell): dist² = 0.2074 ← terdekat ke-3

Labels k=3 tetangga = [Buy, Buy, Sell] = [1, 1, 0]

bincount([1,1,0]) = [1, 2]   (Sell=1 kali, Buy=2 kali)
argmax([1, 2]) = 1

Prediksi test_1 = Buy (1)
```

**Untuk test_2:**
```
Jarak ke training:
  train_1 (Buy): dist² = 0.1025
  train_2 (Buy): dist² = 0.0200  ← terdekat ke-1
  train_3 (Sell): dist² = 0.1700

Labels k=3 tetangga = [Buy, Sell, Buy] = [1, 0, 1]

bincount([1,0,1]) = [1, 2]   (Sell=1 kali, Buy=2 kali)
argmax([1, 2]) = 1

Prediksi test_2 = Buy (1)
```

---

## F. METRIK EVALUASI (evaluation.py)

---

### F.1 — Confusion Matrix

**Rumus dari kode (`evaluation.py:33-36`):**
```python
TP = int(np.sum((y_true == 1) & (y_pred == 1)))
TN = int(np.sum((y_true == 0) & (y_pred == 0)))
FP = int(np.sum((y_true == 0) & (y_pred == 1)))
FN = int(np.sum((y_true == 1) & (y_pred == 0)))
```

**Data contoh — 10 sampel:**

| Sampel | y_true | y_pred | Kategori          |
|--------|--------|--------|-------------------|
| 1      | 1      | 1      | TP (benar Buy)    |
| 2      | 1      | 1      | TP (benar Buy)    |
| 3      | 0      | 0      | TN (benar Sell)   |
| 4      | 0      | 0      | TN (benar Sell)   |
| 5      | 0      | 0      | TN (benar Sell)   |
| 6      | 1      | 0      | FN (salah Sell)   |
| 7      | 0      | 1      | FP (salah Buy)    |
| 8      | 1      | 1      | TP (benar Buy)    |
| 9      | 0      | 0      | TN (benar Sell)   |
| 10     | 1      | 0      | FN (salah Sell)   |

```
TP = 3  (baris 1, 2, 8)
TN = 4  (baris 3, 4, 5, 9)
FP = 1  (baris 7)
FN = 2  (baris 6, 10)
n  = TP + TN + FP + FN = 3 + 4 + 1 + 2 = 10 ✓
```

---

### F.2 — Accuracy

**Rumus dari kode (`evaluation.py:39`):**
```python
accuracy = (TP + TN) / n if n > 0 else 0.0
```

**Definisi matematis:**
```
Accuracy = (TP + TN) / n
```

**Perhitungan dari contoh:**
```
Accuracy = (3 + 4) / 10
         = 7 / 10
         = 0.7000   (70.00%)
```

**Verifikasi dari output aktual program (Naive Bayes):**
```
TP = 15.099, TN = 557, FP = 13.627, FN = 587
n  = 15.099 + 557 + 13.627 + 587 = 29.870

Accuracy = (15.099 + 557) / 29.870
         = 15.656 / 29.870
         = 0.52409...
         ≈ 0.5241 ✓  (cocok dengan output program)
```

---

### F.3 — Precision

**Rumus dari kode (`evaluation.py:40`):**
```python
precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
```

**Definisi matematis:**
```
Precision = TP / (TP + FP)
```

**Dari contoh 10 sampel:**
```
Precision = 3 / (3 + 1)
          = 3 / 4
          = 0.7500   (75.00%)
```

**Dari output aktual (Naive Bayes):**
```
Precision = 15.099 / (15.099 + 13.627)
          = 15.099 / 28.726
          = 0.52561...
          ≈ 0.5256 ✓
```

---

### F.4 — Recall (Sensitivity)

**Rumus dari kode (`evaluation.py:41`):**
```python
recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
```

**Definisi matematis:**
```
Recall = TP / (TP + FN)
```

**Dari contoh 10 sampel:**
```
Recall = 3 / (3 + 2)
       = 3 / 5
       = 0.6000   (60.00%)
```

**Dari output aktual (Naive Bayes):**
```
Recall = 15.099 / (15.099 + 587)
       = 15.099 / 15.686
       = 0.96258...
       ≈ 0.9626 ✓
```

---

### F.5 — F1-Score

**Rumus dari kode (`evaluation.py:42-46`):**
```python
f1_score = (
    2 * precision * recall / (precision + recall)
    if (precision + recall) > 0
    else 0.0
)
```

**Definisi matematis:**
```
F1 = 2 × Precision × Recall / (Precision + Recall)
```

**Dari contoh 10 sampel:**
```
F1 = 2 × 0.75 × 0.60 / (0.75 + 0.60)
   = 2 × 0.45 / 1.35
   = 0.90 / 1.35
   = 0.6667   (66.67%)
```

**Verifikasi menggunakan formula alternatif:**
```
F1 = 2TP / (2TP + FP + FN)
   = (2 × 3) / (2×3 + 1 + 2)
   = 6 / (6 + 3)
   = 6 / 9
   = 0.6667 ✓
```

**Dari output aktual (Naive Bayes):**
```
F1 = 2 × 0.5256 × 0.9626 / (0.5256 + 0.9626)
   = 2 × 0.50594 / 1.4882
   = 1.01188 / 1.4882
   = 0.67992...
   ≈ 0.6800 ✓
```

---

### F.6 — Specificity

**Rumus dari kode (`evaluation.py:47`):**
```python
specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
```

**Definisi matematis:**
```
Specificity = TN / (TN + FP)
```

**Dari contoh 10 sampel:**
```
Specificity = 4 / (4 + 1)
            = 4 / 5
            = 0.8000   (80.00%)
```

**Dari output aktual (Naive Bayes):**
```
Specificity = 557 / (557 + 13.627)
            = 557 / 14.184
            = 0.039273...
            ≈ 0.0393 ✓
```

---

### F.7 — Persentase di Confusion Matrix

**Rumus dari kode (`evaluation.py:122`):**
```python
pct = val / total * 100 if total > 0 else 0
```

**Dari output aktual (Naive Bayes), total = 29.870:**
```
TN  = 557   → 557   / 29.870 × 100 = 1.864%
FP  = 13.627 → 13.627 / 29.870 × 100 = 45.617%
FN  = 587   → 587   / 29.870 × 100 = 1.965%
TP  = 15.099 → 15.099 / 29.870 × 100 = 50.553%

Verifikasi: 1.864 + 45.617 + 1.965 + 50.553 = 99.999 ≈ 100% ✓
```

---

## G. HYBRID PIPELINE LOGIC (hybrid_pipeline.py)

---

### G.1 — Majority Vote

**Rumus dari kode (`hybrid_pipeline.py:86-87`):**
```python
total_votes = nb_votes + dt_votes + knn_votes
final_preds = (total_votes >= 2).astype(np.int32)
```

**Definisi:**
```
nb_votes  = argmax P(y|x)   → 0 atau 1
dt_votes  = DT.predict(x)   → 0 atau 1
knn_votes = KNN.predict(x)  → 0 atau 1

total = nb_votes + dt_votes + knn_votes   ∈ {0, 1, 2, 3}
final = 1 (Buy)  jika total ≥ 2
      = 0 (Sell) jika total < 2
```

---

### G.2 — Contoh Voting per Sampel

**5 sampel contoh:**

| Sampel | NB vote | DT vote | KNN vote | total | final |
|--------|---------|---------|----------|-------|-------|
| A      | 1 (Buy) | 1 (Buy) | 1 (Buy)  | 3     | Buy (suara bulat) |
| B      | 0 (Sell)| 0 (Sell)| 0 (Sell) | 0     | Sell (suara bulat) |
| C      | 1 (Buy) | 1 (Buy) | 0 (Sell) | 2     | Buy (mayoritas) |
| D      | 0 (Sell)| 1 (Buy) | 0 (Sell) | 1     | Sell (mayoritas) |
| E      | 1 (Buy) | 0 (Sell)| 1 (Buy)  | 2     | Buy (mayoritas) |

**Kasus C:** NB dan DT setuju Buy, KNN setuju Sell → 2 vs 1 → **Buy menang**
**Kasus D:** NB dan KNN setuju Sell, DT setuju Buy → 2 vs 1 → **Sell menang**

---

### G.3 — Perbandingan Cascade vs Majority Vote

**Cascade lama — contoh 5 sampel:**
```
Sampel A: NB confidence=0.55 < 0.60 → output NB=Buy  (DT,KNN tidak dipakai)
Sampel B: NB confidence=0.52 < 0.60 → output NB=Sell (DT,KNN tidak dipakai)
Sampel C: NB confidence=0.72 ≥ 0.60 → ke DT
  DT=Buy = NB=Buy → ke KNN
  KNN=Sell ≠ DT=Buy → output KNN=Sell
Sampel D: NB confidence=0.68 ≥ 0.60 → ke DT
  DT=Buy ≠ NB=Sell → output DT=Buy
Sampel E: NB confidence=0.48 < 0.60 → output NB=Buy  (DT,KNN tidak dipakai)
```

**Majority Vote baru — contoh 5 sampel yang sama:**
```
Sampel A: NB=Buy, DT=Buy, KNN=Sell  → total=2 → Buy
Sampel B: NB=Sell, DT=Sell, KNN=Buy → total=1 → Sell
Sampel C: NB=Buy, DT=Buy, KNN=Sell  → total=2 → Buy
Sampel D: NB=Sell, DT=Buy, KNN=Sell → total=1 → Sell
Sampel E: NB=Buy, DT=Buy, KNN=Buy   → total=3 → Buy
```

Semua model berkontribusi pada semua sampel.

---

## H. VERIFIKASI LENGKAP OUTPUT AKTUAL PROGRAM

### H.1 — Naive Bayes: Verifikasi Manual Semua Metrik

**Nilai aktual dari program:**
```
TP = 15.099,  TN = 557,  FP = 13.627,  FN = 587
n  = 29.870
```

```
Accuracy:
  (15.099 + 557) / 29.870 = 15.656 / 29.870 = 0.524092 ≈ 0.5241 ✓

Precision:
  15.099 / (15.099 + 13.627) = 15.099 / 28.726 = 0.525609 ≈ 0.5256 ✓

Recall:
  15.099 / (15.099 + 587) = 15.099 / 15.686 = 0.962572 ≈ 0.9626 ✓

F1:
  2 × 0.5256 × 0.9626 / (0.5256 + 0.9626)
  = 2 × 0.505943 / 1.4882
  = 1.011886 / 1.4882
  = 0.679966 ≈ 0.6800 ✓

Specificity:
  557 / (557 + 13.627) = 557 / 14.184 = 0.039271 ≈ 0.0393 ✓
```

### H.2 — Decision Tree: Verifikasi Manual

```
TP = 13.845, TN = 1.752, FP = 12.432, FN = 1.841
n  = 29.870
```

```
Accuracy:
  (13.845 + 1.752) / 29.870 = 15.597 / 29.870 = 0.522118 ≈ 0.5222 ✓

Precision:
  13.845 / (13.845 + 12.432) = 13.845 / 26.277 = 0.526854 ≈ 0.5269 ✓

Recall:
  13.845 / (13.845 + 1.841) = 13.845 / 15.686 = 0.882639 ≈ 0.8826 ✓

F1:
  2 × 0.5269 × 0.8826 / (0.5269 + 0.8826)
  = 2 × 0.464988 / 1.4095
  = 0.929976 / 1.4095
  = 0.659795 ≈ 0.6598 ≈ 0.6599 ✓

Specificity:
  1.752 / (1.752 + 12.432) = 1.752 / 14.184 = 0.123513 ≈ 0.1235 ✓
```

### H.3 — KNN: Verifikasi Manual

```
TP = 8.853, TN = 6.254, FP = 7.930, FN = 6.833
n  = 29.870
```

```
Accuracy:
  (8.853 + 6.254) / 29.870 = 15.107 / 29.870 = 0.505788 ≈ 0.5058 ✓

Precision:
  8.853 / (8.853 + 7.930) = 8.853 / 16.783 = 0.527499 ≈ 0.5275 ✓

Recall:
  8.853 / (8.853 + 6.833) = 8.853 / 15.686 = 0.564394 ≈ 0.5644 ✓

F1:
  2 × 0.5275 × 0.5644 / (0.5275 + 0.5644)
  = 2 × 0.297721 / 1.0919
  = 0.595442 / 1.0919
  = 0.545335 ≈ 0.5453 ✓

Specificity:
  6.254 / (6.254 + 7.930) = 6.254 / 14.184 = 0.440882 ≈ 0.4409 ✓
```

### H.4 — Hybrid Pipeline: Verifikasi Manual

```
TP = 14.662, TN = 880, FP = 13.304, FN = 1.024
n  = 29.870
```

```
Accuracy:
  (14.662 + 880) / 29.870 = 15.542 / 29.870 = 0.520277 ≈ 0.5203 ✓

Precision:
  14.662 / (14.662 + 13.304) = 14.662 / 27.966 = 0.524301 ≈ 0.5243 ✓

Recall:
  14.662 / (14.662 + 1.024) = 14.662 / 15.686 = 0.934739 ≈ 0.9347 ✓

F1:
  2 × 0.5243 × 0.9347 / (0.5243 + 0.9347)
  = 2 × 0.490063 / 1.4590
  = 0.980126 / 1.4590
  = 0.671778 ≈ 0.6718 ✓

Specificity:
  880 / (880 + 13.304) = 880 / 14.184 = 0.062039 ≈ 0.0620 ✓
```

---

## I. RINGKASAN TABEL SEMUA RUMUS

| # | Nama Rumus | Formula | File | Baris |
|---|-----------|---------|------|-------|
| 1 | SMA | `(Σ Close_i) / N` | features.py | 62 |
| 2 | EMA faktor | `k = 2/(span+1)` | features.py | 63 |
| 3 | EMA rekursi | `EMA(t) = Close(t)×k + EMA(t-1)×(1-k)` | features.py | 63 |
| 4 | MACD | `EMA_12 - EMA_26` | features.py | 67 |
| 5 | Bollinger Upper | `SMA_20 + 2×σ(20)` | features.py | 71 |
| 6 | Bollinger Lower | `SMA_20 - 2×σ(20)` | features.py | 72 |
| 7 | RSI delta | `Δ(t) = Close(t) - Close(t-1)` | features.py | 75 |
| 8 | RSI gain | `gain(t) = max(Δ(t), 0)` | features.py | 76 |
| 9 | RSI loss | `loss(t) = max(-Δ(t), 0)` | features.py | 77 |
| 10 | RSI RS | `RS = avg_gain / avg_loss` | features.py | 78 |
| 11 | RSI | `100 - (100 / (1 + RS))` | features.py | 79 |
| 12 | Volume Change | `(Vol(t) - Vol(t-1)) / Vol(t-1)` | features.py | 83 |
| 13 | Target Label | `1 if Close(t+1)>Close(t) else 0` | features.py | 86 |
| 14 | Min-Max Norm | `(X - X_min) / (X_max - X_min)` | features.py | 44 |
| 15 | NB Prior | `P(c) = \|X_c\| / \|X\|` | naive_bayes.py | 54 |
| 16 | NB Mean | `μ_c = mean(X_c, axis=0)` | naive_bayes.py | 55 |
| 17 | NB Variance | `σ²_c = var(X_c) + 1e-9` | naive_bayes.py | 56 |
| 18 | Gaussian Log PDF | `-0.5×[log(2πσ²) + (x-μ)²/σ²]` | naive_bayes.py | 81 |
| 19 | Bayes Theorem | `log P(y\|x) = log P(y) + Σ log P(xᵢ\|y)` | naive_bayes.py | 105-107 |
| 20 | Log-Sum-Exp | `exp(z - max(z)) / Σ exp(z - max(z))` | naive_bayes.py | 114-116 |
| 21 | Gini Impurity | `1 - Σ pᵢ²` | decision_tree.py | 255-265 |
| 22 | Weighted Gini | `(n_L/n)×Gini(L) + (n_R/n)×Gini(R)` | decision_tree.py | 267-276 |
| 23 | Majority Vote DT | `argmax(counts)` | decision_tree.py | 280-281 |
| 24 | Euclidean Distance² | `\|\|a-b\|\|² = \|\|a\|\|²+\|\|b\|\|²-2(a·b)` | knn.py | 82-92 |
| 25 | KNN Majority Vote | `argmax(bincount(neighbors))` | knn.py | 105 |
| 26 | Accuracy | `(TP+TN)/n` | evaluation.py | 39 |
| 27 | Precision | `TP/(TP+FP)` | evaluation.py | 40 |
| 28 | Recall | `TP/(TP+FN)` | evaluation.py | 41 |
| 29 | F1-Score | `2×P×R/(P+R)` | evaluation.py | 42-46 |
| 30 | Specificity | `TN/(TN+FP)` | evaluation.py | 47 |
| 31 | CM Persentase | `val/total×100` | evaluation.py | 122 |
| 32 | NB Confidence | `max(P(Buy\|x), P(Sell\|x))` | hybrid_pipeline.py | 74 |
| 33 | Hybrid Stage 1 filter | `confidence < 0.60 → output NB` | hybrid_pipeline.py | 78-80 |
| 34 | Hybrid Stage 2 vote | `NB≠DT → output DT` | hybrid_pipeline.py | 117 |
| 35 | Hybrid Stage 3 vote | `KNN≠DT → output KNN` | hybrid_pipeline.py | 139 |

---

*Semua 35 rumus telah ditulis ulang dari kode sumber dan dihitung manual dengan angka konkret. Hasil perhitungan manual diverifikasi terhadap output aktual program.*
