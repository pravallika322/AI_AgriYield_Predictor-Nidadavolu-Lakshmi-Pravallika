import pandas as pd
df = pd.read_csv(r"C:\Users\nidad\OneDrive\Desktop\crop yield predictor\Merged_Crop_Yield_Dataset.csv")

# show unique labels
print("Unique labels sample:", df['label'].unique()[:50])

# filter maize rows (case-insensitive)
maize_df = df[df['label'].str.lower() == 'maize']
print("\nMaize rows count:", len(maize_df))
if len(maize_df) > 0:
    print("Maize Yield stats (kg/ha):")
    print(maize_df['Yield'].describe())
    print("\nFirst 10 maize rows (Yield):")
    print(maize_df[['N','P','K','temperature','humidity','ph','rainfall','Yield']].head(10))
else:
    print("No rows with label == 'Maize' found. Maybe label is spelled differently.")
