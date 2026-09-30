"""Car Price Predictor: data prep, EDA, model comparison, export of the best model (model.pkl)."""
import pandas as pd, numpy as np, joblib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.inspection import permutation_importance

df = pd.read_csv("cars.csv", index_col=0)
print("Shape:", df.shape)
print("Missing values:", int(df.isna().sum().sum()), "| duplicates:", int(df.duplicated().sum()))

# ---------- Data preparation
df = df.drop_duplicates()
df = df[(df["mileage"] > 0) & (df["engine"] > 0) & (df["max_power"] > 0)]       # impossible values -> removed
q = df["selling_price"].quantile([.005, .995]); df = df[df["selling_price"].between(*q)]  # extreme outliers
df = df.drop(columns=["car_name", "model"])      # high-cardinality text, information already carried by brand + specs
print("After cleaning:", df.shape)
NUM = ["vehicle_age", "km_driven", "mileage", "engine", "max_power", "seats"]
CAT = ["brand", "seller_type", "fuel_type", "transmission_type"]
X, y = df[NUM + CAT], df["selling_price"]

# ---------- EDA
sns.set_theme(style="whitegrid")
fig, ax = plt.subplots(2, 2, figsize=(12, 8))
sns.histplot(np.log10(y), bins=40, ax=ax[0,0]); ax[0,0].set_title("Selling price distribution"); ax[0,0].set_xlabel("log10(price in INR)")
top = df.groupby("brand").selling_price.median().sort_values(ascending=False).head(10)
sns.barplot(x=top.values/1e5, y=top.index, ax=ax[0,1], color="#1f5fbf"); ax[0,1].set_title("Median price by brand (top 10)"); ax[0,1].set_xlabel("lakh INR")
sns.scatterplot(data=df.sample(3000, random_state=1), x="max_power", y="selling_price", hue="fuel_type", alpha=.5, ax=ax[1,0]); ax[1,0].set_title("Price vs engine power (bhp)")
sns.boxplot(data=df, x="vehicle_age", y="selling_price", ax=ax[1,1], color="#9ec1f0", showfliers=False); ax[1,1].set_title("Price vs vehicle age")
plt.tight_layout(); plt.savefig("static/charts/eda.png", dpi=110); plt.close()
plt.figure(figsize=(7, 5.5)); sns.heatmap(df[NUM + ["selling_price"]].corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation matrix"); plt.tight_layout(); plt.savefig("static/charts/correlation.png", dpi=110); plt.close()

# ---------- Pipelines / models (price is modelled on log scale)
pre = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), NUM),
    ("cat", OneHotEncoder(handle_unknown="ignore"), CAT)])
models = {
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "KNN": KNeighborsRegressor(n_neighbors=7, weights="distance"),
    "Decision Tree": DecisionTreeRegressor(max_depth=12, min_samples_leaf=5, random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=150, min_samples_leaf=2, n_jobs=-1, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=300, max_depth=5, learning_rate=0.08, random_state=42),
}
def make(m): return TransformedTargetRegressor(Pipeline([("pre", pre), ("model", m)]), func=np.log1p, inverse_func=np.expm1)

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
cv = KFold(5, shuffle=True, random_state=42); rows, fitted = [], {}
for name, m in models.items():
    est = make(m)
    cv_r2 = cross_val_score(est, Xtr, ytr, cv=cv, scoring="r2").mean()
    est.fit(Xtr, ytr); fitted[name] = est; p = est.predict(Xte)
    rows.append({"Model": name, "CV R2": cv_r2, "Test R2": r2_score(yte, p),
                 "Test MAE (INR)": mean_absolute_error(yte, p), "Test RMSE (INR)": mean_squared_error(yte, p) ** .5})
    print(rows[-1])
res = pd.DataFrame(rows).sort_values("CV R2", ascending=False).round(4)
res.to_csv("model_comparison.csv", index=False); print("\n", res.to_string(index=False))

plt.figure(figsize=(8, 4)); sns.barplot(data=res, y="Model", x="Test R2", color="#1f5fbf"); plt.xlim(0.6, 1)
plt.title("Model comparison (R2 on test set)"); plt.tight_layout(); plt.savefig("static/charts/model_comparison.png", dpi=110); plt.close()

# ---------- Best model
best = res.iloc[0]["Model"]; est = fitted[best]; p = est.predict(Xte); print("\nBest:", best)
plt.figure(figsize=(5.5, 5.5)); plt.scatter(yte/1e5, p/1e5, s=6, alpha=.35); lim = [0, yte.max()/1e5]
plt.plot(lim, lim, "r--"); plt.xlabel("Actual price (lakh INR)"); plt.ylabel("Predicted price (lakh INR)")
plt.title(f"{best}: actual vs predicted"); plt.tight_layout(); plt.savefig("static/charts/actual_vs_pred.png", dpi=110); plt.close()

pi = permutation_importance(est, Xte, yte, n_repeats=5, random_state=42, scoring="r2", n_jobs=-1)
imp = pd.Series(pi.importances_mean, index=X.columns).sort_values()
imp.plot.barh(figsize=(7, 4), color="#1f5fbf", title="Feature importance (drop in R2 when shuffled)")
plt.tight_layout(); plt.savefig("static/charts/feature_importance.png", dpi=110); plt.close()
print("\nImportance:\n", imp.sort_values(ascending=False).round(3))

final = make(models[best]).fit(X, y)
meta = {"best_model": best, "test_r2": float(res.iloc[0]["Test R2"]), "test_mae": float(res.iloc[0]["Test MAE (INR)"]),
        "brands": sorted(df.brand.unique()), "n_rows": int(len(df)),
        "ranges": {c: [float(df[c].min()), float(df[c].max()), float(df[c].median())] for c in NUM},
        "sellers": sorted(df.seller_type.unique()), "fuels": sorted(df.fuel_type.unique()), "trans": sorted(df.transmission_type.unique())}
joblib.dump({"model": final, "meta": meta}, "model.pkl", compress=3)
import os; print("model.pkl size (MB):", round(os.path.getsize("model.pkl")/1e6, 1))
