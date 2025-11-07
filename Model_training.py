import numpy as np
import joblib
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import math

# Load Data
X = np.load("X_processed.npy")
y = np.load("y_processed.npy")

# Load target scaler
y_scaler = joblib.load("scaler_y.pkl")

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# XGBoost Model
model = XGBRegressor(
    n_estimators=450,
    max_depth=6,
    learning_rate=0.07,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42
)

print("⏳ Training... Please wait...")
model.fit(X_train, y_train)

# Predict (scaled)
y_pred_scaled = model.predict(X_test)

# ✅ Convert back to real values (tons/ha)
y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
y_test_real = y_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

rmse = math.sqrt(mean_squared_error(y_test_real, y_pred))
r2 = r2_score(y_test_real, y_pred)

print("\n✅ Training Completed Successfully!")
print(f"RMSE: {rmse:.3f} tons/hectare")
print(f"R² Score: {r2:.4f}")

joblib.dump(model, "yield_model.pkl")
print("✅ Model saved as yield_model.pkl ✅")
