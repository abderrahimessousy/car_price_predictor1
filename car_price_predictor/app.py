import os, joblib, pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)
bundle = joblib.load("model.pkl")
model, meta = bundle["model"], bundle["meta"]
FIELDS = ["brand", "vehicle_age", "km_driven", "seller_type", "fuel_type", "transmission_type", "mileage", "engine", "max_power", "seats"]

# Exchange rate: 1 INR = 0.097 MAD (approx., mid-market, mid-2026). Update it here whenever you want.
INR_TO_MAD = 0.097

def dh(inr):
    """Convert an INR amount to DH, rounded to the nearest 100 DH."""
    return f"{round(inr * INR_TO_MAD, -2):,.0f}"

@app.route("/", methods=["GET", "POST"])
def index():
    result, error, form = None, None, {}
    if request.method == "POST":
        form = request.form
        try:
            row = pd.DataFrame([{
                "brand": form["brand"], "vehicle_age": float(form["vehicle_age"]), "km_driven": float(form["km_driven"]),
                "seller_type": form["seller_type"], "fuel_type": form["fuel_type"], "transmission_type": form["transmission_type"],
                "mileage": float(form["mileage"]), "engine": float(form["engine"]), "max_power": float(form["max_power"]),
                "seats": float(form["seats"])}])[FIELDS]
            price = float(model.predict(row)[0]); mae = meta["test_mae"]
            result = {"price": dh(price), "low": dh(max(price - mae, 0)), "high": dh(price + mae)}
        except (KeyError, ValueError):
            error = "Please fill in all fields with valid values."
    return render_template("index.html", meta=meta, r=meta["ranges"], f=form, result=result, error=error)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=os.environ.get("FLASK_DEBUG") == "1")
