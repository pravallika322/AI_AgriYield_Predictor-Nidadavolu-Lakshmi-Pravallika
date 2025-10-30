# 📄 eda.py — Exploratory Data Analysis for Crop Yield Dataset

import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')  # ✅ Prevents GUI errors
import matplotlib.pyplot as plt

# 1️⃣ Load dataset
CSV_PATH = r"C:\Users\nidad\OneDrive\Desktop\crop yield predictor\Merged_Crop_Yield_Dataset.csv"
df = pd.read_csv(CSV_PATH)

print("✅ Dataset Loaded Successfully!")
print("Shape:", df.shape)
print("\n📊 Columns:\n", df.columns.tolist())

# 2️⃣ Display first few rows
print("\n🔹 Sample Data:")
print(df.head())

# 3️⃣ Check for missing values
print("\n❌ Missing Values per Column:")
print(df.isnull().sum())

# 4️⃣ Summary statistics
print("\n📈 Statistical Summary:")
print(df.describe())

# 5️⃣ Unique crop labels
if 'label' in df.columns:
    print("\n🌾 Unique Crops:")
    print(df['label'].unique())

# 6️⃣ Correlation matrix heatmap (for numeric features)
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

if len(numeric_cols) > 1:
    plt.figure(figsize=(10,6))
    sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='YlGnBu')
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig("correlation_heatmap.png")
    print("\n📸 Saved correlation heatmap as 'correlation_heatmap.png'")
else:
    print("\n⚠️ Not enough numeric columns for correlation heatmap.")

# 7️⃣ Optional: Save summary report
with open("eda_report.txt", "w") as f:
    f.write("EDA Summary Report\n")
    f.write("=====================\n\n")
    f.write(str(df.describe()))
    f.write("\n\nMissing Values:\n")
    f.write(str(df.isnull().sum()))
    f.write("\n\nUnique Crops:\n")
    if 'label' in df.columns:
        f.write(str(df['label'].unique()))
print("\n✅ EDA report saved as 'eda_report.txt'")
