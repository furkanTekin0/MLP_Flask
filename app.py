from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Eğitimde kaydedilen bundle'ı yükle
with open("model.pkl", "rb") as f:
    bundle = pickle.load(f)

model = bundle["model"]
raw_features = bundle["raw_features"]
categorical_cols = bundle["categorical_cols"]
numeric_cols = bundle["numeric_cols"]
model_columns = bundle["model_columns"]


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Formdan gelen değerleri al
        raw_input = {col: request.form.get(col) for col in raw_features}
        df_input = pd.DataFrame([raw_input])

        # Tip dönüşümleri
        for col in numeric_cols:
            df_input[col] = pd.to_numeric(df_input[col], errors="coerce")

        # Basit doğrulama: numeric alanlarda NaN kalmasın
        if df_input[numeric_cols].isna().any().any():
            return render_template("index.html", error_text="Sayısal alanlara geçerli değer girin.")

        # Eğitimdeki ile aynı encoding
        df_encoded = pd.get_dummies(df_input, columns=categorical_cols, drop_first=True)

        # Eğitimde seçilen kolonlara hizala
        df_encoded = df_encoded.reindex(columns=model_columns, fill_value=0)

        pred = float(model.predict(df_encoded)[0])

        # Burada hedefin birimi neyse ona göre biçimlendir
        return render_template("index.html", prediction_text=f"Tahmin: {pred:,.2f}")

    except Exception as e:
        return render_template("index.html", error_text=str(e))


if __name__ == "__main__":
    app.run(debug=True)
