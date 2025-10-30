import pandas as pd
pd.options.display.float_format = '{:,.2f}'.format

df = pd.read_csv(r"C:\Users\nidad\OneDrive\Desktop\crop yield predictor\Merged_Crop_Yield_Dataset.csv")
# compute mean yield by label (in kg/ha)
group = df.groupby('label')['Yield'].agg(['count','mean','min','max']).reset_index()
group['mean_tons'] = group['mean'] / 1000.0
group = group.sort_values('mean', ascending=False)
print(group.head(40))
