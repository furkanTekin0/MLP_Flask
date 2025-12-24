# Multiple Linear Regression + Flask GUI

Bu repo, **çoklu doğrusal regresyon** modeli eğitip (`model.pkl`) bir **Flask** arayüzü ile tahmin sunmak için örnek bir şablondur.

## 1) Veri seti
Bu şablon varsayılan olarak `data/insurance.csv` bekler (Kaggle: *Medical Cost Personal Dataset* gibi).
- Hedef (target): `charges`
- Giriş değişkenleri (features): `age, sex, bmi, children, smoker, region`  (**<=10 özellik şartına uygun**)

> Kendi veri setinizi kullanacaksanız `model.py` içindeki `TARGET_COL` ve `RAW_FEATURES` alanlarını güncelleyin,
> ayrıca `templates/index.html` form alanlarını da aynı sıraya göre düzenleyin.

## 2) Kurulum
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## 3) Modeli eğit (model.pkl üret)
```bash
python model.py
```
Çıktı:
- `model.pkl` (model + kolon bilgileri)

## 4) Flask uygulamasını çalıştır
```bash
python app.py
```
Sonra tarayıcıdan: `http://127.0.0.1:5000/`

## 5) Dosya Yapısı
```
.
├─ app.py
├─ model.py
├─ model.pkl                # eğitimden sonra oluşur
├─ data/
│  └─ insurance.csv         # kendi veri setiniz
├─ templates/
│  └─ index.html
├─ static/
│  └─ style.css
└─ requirements.txt
```

## Notlar
- Backward Elimination p-value yaklaşımı `statsmodels` ile yapılır.
- Flask tarafında, kullanıcı girdisi önce `pandas.get_dummies(drop_first=True)` ile encode edilir, sonra eğitimdeki kolonlara hizalanır.
