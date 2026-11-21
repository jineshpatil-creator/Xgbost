"""CoverCast: Flask front end for the uploaded XGBoost regressor (model.pkl).

Feature names/order are read from the model itself. The model stores categorical
columns as integers but NOT the label->integer mapping, so the mapping lives in
ENCODINGS below. Confirm it matches the encoding used in your training notebook.
"""
import os
import pickle

import pandas as pd
from flask import Flask, jsonify, render_template, request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(BASE_DIR, "model.pkl"), "rb") as fh:
    model = pickle.load(fh)

FEATURES = list(model.get_booster().feature_names)  # training order

# ASSUMPTION: alphabetical label encoding (sklearn LabelEncoder default).
ENCODINGS = {
    "sex": {"female": 0, "male": 1},
    "smoker": {"no": 0, "yes": 1},
    "region": {"northeast": 0, "northwest": 1, "southeast": 2, "southwest": 3},
}
# name: (type, min, max). Bounds are input sanity limits, not learned from the model.
NUMERIC = {"age": (int, 18, 100), "bmi": (float, 10.0, 60.0), "children": (int, 0, 10)}

app = Flask(__name__)


def parse(payload):
    row, errors = {}, {}
    for name in FEATURES:
        raw = payload.get(name)
        if raw is None or str(raw).strip() == "":
            errors[name] = "This field is required."
            continue
        if name in ENCODINGS:
            key = str(raw).strip().lower()
            if key not in ENCODINGS[name]:
                errors[name] = "Choose one of: " + ", ".join(ENCODINGS[name]) + "."
            else:
                row[name] = ENCODINGS[name][key]
        else:
            cast, lo, hi = NUMERIC[name]
            try:
                val = cast(float(raw)) if cast is int and float(raw).is_integer() else cast(raw)
            except (TypeError, ValueError):
                errors[name] = "Enter a valid number."
                continue
            if not lo <= val <= hi:
                errors[name] = f"Enter a value between {lo} and {hi}."
            else:
                row[name] = val
    return row, errors


def predict(row):
    df = pd.DataFrame([row], columns=FEATURES)
    return float(model.predict(df)[0])


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/health")
def health():
    return jsonify(status="ok")


@app.post("/predict")
def predict_route():
    row, errors = parse(request.get_json(silent=True) or {})
    if errors:
        return jsonify(errors=errors), 400
    result = predict(row)
    flipped = dict(row, smoker=1 - row["smoker"]) if "smoker" in row else None
    return jsonify(
        charges=round(result, 2),
        smoker=bool(row.get("smoker")),
        alternative=round(predict(flipped), 2) if flipped else None,
    )


if __name__ == "__main__":
    app.run(debug=True, port=int(os.environ.get("PORT", 5000)))
