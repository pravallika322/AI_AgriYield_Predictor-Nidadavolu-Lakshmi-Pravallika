import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import StandardScaler
import joblib

# Load dataset
CSV_PATH = r"C:\Users\nidad\OneDrive\Desktop\crop yield predictor\Merged_Crop_Yield_Dataset.csv"
df = pd.read_csv(CSV_PATH)

print("✅ Dataset Loaded:", df.shape)

# ❌ Do NOT convert to tons (keep as kilograms)
# df['Yield'] = df['Yield'] / 1000   <-- remove this line

INPUT_COLS = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
Y_COL = 'Yield'

# ✅ One-hot encode crops
df = pd.get_dummies(df, columns=['label'])
crop_cols = [c for c in df.columns if c.startswith('label_')]

# ✅ Separate numeric + crop features
X_numeric = df[INPUT_COLS].values
X_crops = df[crop_cols].values

# ✅ Scale numeric inputs
input_scaler = StandardScaler()
X_scaled_numeric = input_scaler.fit_transform(X_numeric)

# ✅ Combine numeric + crop columns
X_processed = np.hstack([X_scaled_numeric, X_crops])

# ✅ Scale output
y = df[Y_COL].values.reshape(-1, 1)
y_scaler = StandardScaler()
y_processed = y_scaler.fit_transform(y).flatten()

# ✅ Save everything
np.save("X_processed.npy", X_processed)
np.save("y_processed.npy", y_processed)
joblib.dump(input_scaler, "scaler_input.pkl")
joblib.dump(y_scaler, "scaler_y.pkl")

# ✅ Crop mapping
crop_mapping = {i: col.replace("label_", "") for i, col in enumerate(crop_cols)}
with open("crop_mapping.json", "w") as f:
    json.dump(crop_mapping, f, indent=2)

print("\n✅ Preprocessing Completed!")
print("X_processed:", X_processed.shape)
print("y_processed:", y_processed.shape)
print("Crops:", len(crop_mapping))
