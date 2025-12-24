import pickle
import pandas as pd
import numpy as np
import statsmodels.api as sm

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ==============================
# 1) AYARLAR (veri setine göre değiştir)
# ==============================
DATA_PATH = "data/insurance.csv"   # <- kendi dosyanın yolu
TARGET_COL = "charges"            # <- hedef değişken
RAW_FEATURES = ["age", "sex", "bmi", "children", "smoker", "region"]  # <= 10 özellik şartı
CATEGORICAL_COLS = ["sex", "smoker", "region"]
NUMERIC_COLS = ["age", "bmi", "children"]

SIGNIFICANCE_LEVEL = 0.05


def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Eksik değerleri doldur: sayısal -> median, kategorik -> mode."""
    df = df.copy()
    for c in df.columns:
        if df[c].isna().sum() == 0:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].fillna(df[c].median())
        else:
            df[c] = df[c].fillna(df[c].mode(dropna=True)[0])
    return df


def backward_elimination(X: pd.DataFrame, y: pd.Series, sl: float = 0.05):
    """p-value > sl olan değişkenleri tek tek eleyerek backward elimination uygular."""
    cols = list(X.columns)
    X_ = sm.add_constant(X, has_constant="add")
    while True:
        model_ols = sm.OLS(y, X_).fit()
        pvals = model_ols.pvalues.drop("const", errors="ignore")
        max_p = pvals.max()
        if max_p <= sl:
            break
        worst_feature = pvals.idxmax()
        # ilgili kolonu çıkar
        cols.remove(worst_feature)
        X_ = sm.add_constant(X[cols], has_constant="add")
    final_model = sm.OLS(y, X_).fit()
    return cols, final_model


def main():
    # ==============================
    # 2) VERİYİ OKU + ÖN İŞLEME
    # ==============================
    df = pd.read_csv(DATA_PATH)

    # Sadece seçtiğimiz özellikler + hedef
    df = df[RAW_FEATURES + [TARGET_COL]].copy()

    # Eksik değer kontrol/doldurma
    df = fill_missing_values(df)

    # X/y
    X_raw = df[RAW_FEATURES]
    y = df[TARGET_COL].astype(float)

    # Train/test split (Backward Elimination'ı train üzerinde yapmak daha doğru)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.2, random_state=42
    )

    # ==============================
    # 3) KATEGORİK ENCODING (One-Hot via get_dummies)
    # ==============================
    # Dummies'i float olarak üretelim ki statsmodels OLS'e giderken dtype=object sorun çıkarmasın
    X_train = pd.get_dummies(
        X_train_raw, columns=CATEGORICAL_COLS, drop_first=True, dtype=float
    )
    X_test = pd.get_dummies(
        X_test_raw, columns=CATEGORICAL_COLS, drop_first=True, dtype=float
    )

    # Test kolonlarını train'e hizala (eksik kolonları 0 ile doldur)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    # ==============================
    # 4) BACKWARD ELIMINATION
    # ==============================
    selected_cols, ols_model = backward_elimination(X_train, y_train, sl=SIGNIFICANCE_LEVEL)

    print("\n=== Backward Elimination Sonrası Seçilen Kolonlar ===")
    print(selected_cols)
    print("\n=== OLS Özeti (kısa) ===")
    print(ols_model.summary())

    # Seçilen kolonlarla yeniden veri setini hazırla
    X_train_sel = X_train[selected_cols]
    X_test_sel = X_test[selected_cols]

    # ==============================
    # 5) SKLEARN MODELİ EĞİT + DEĞERLENDİR
    # ==============================
    lr = LinearRegression()
    lr.fit(X_train_sel, y_train)

    y_pred = lr.predict(X_test_sel)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    print("\n=== Değerlendirme (Test) ===")
    print(f"R2  : {r2:.4f}")
    print(f"MAE : {mae:.4f}")
    print(f"MSE : {mse:.4f}")

    # ==============================
    # 6) MODELİ KAYDET (Flask için gerekli her şeyi kaydediyoruz)
    # ==============================
    bundle = {
        "model": lr,
        "raw_features": RAW_FEATURES,
        "categorical_cols": CATEGORICAL_COLS,
        "numeric_cols": NUMERIC_COLS,
        "model_columns": selected_cols,   # encode+elimination sonrası kolonlar
        "significance_level": SIGNIFICANCE_LEVEL,
    }

    with open("model.pkl", "wb") as f:
        pickle.dump(bundle, f)

    print("\nmodel.pkl kaydedildi ✅")


if __name__ == "__main__":
    main()
