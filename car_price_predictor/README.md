# Car Price Predictor

Predicts the resale price (shown in DH, model trained on INR prices) of a used car. Flask web app + scikit-learn model.
Data: CarDekho used-car listings (15,411 rows). Target: `selling_price`.

## Structure
```
app.py              Flask application
train.py            data prep, EDA, model comparison, exports model.pkl
cars.csv            dataset
model.pkl           trained pipeline (Gradient Boosting) + metadata
templates/          index.html (prediction form)
static/             style.css, charts/
Procfile            gunicorn entry point (Heroku / Render)
requirements.txt
```

## Run locally
```
python -m venv venv
venv\Scripts\activate        # Windows  (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt
python app.py                # http://127.0.0.1:5000
```
Retrain (optional): `python train.py`

## Method
- Cleaning: 167 duplicates removed, impossible values (mileage/engine/power = 0) removed, extreme prices (outer 1%) trimmed, `car_name` and `model` dropped (high cardinality).
- Price modelled on a log scale; numeric features scaled, categoricals one-hot encoded (inside one pipeline).
- 6 models compared with 5-fold CV + held-out test set; best chosen on CV R2.

| Model | CV R2 | Test R2 | Test MAE (INR) |
|---|---|---|---|
| Gradient Boosting | 0.931 | 0.945 | 87,299 |
| Random Forest | 0.923 | 0.938 | 89,161 |
| KNN | 0.907 | 0.912 | 102,780 |
| Decision Tree | 0.893 | 0.914 | 101,010 |
| Linear Regression | 0.766 | 0.829 | 133,301 |
| Ridge | 0.762 | 0.831 | 133,435 |

Most important features: max_power, vehicle_age, engine, brand, km_driven.

## Currency
The model predicts in INR; `app.py` converts to DH with `INR_TO_MAD = 0.097` (approx. mid-2026). Change that constant to update the rate.
