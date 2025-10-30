import json
with open("crop_mapping.json") as f:
    mapping = json.load(f)

print("Crops and their indices (crop_mapping.json):")
for k,v in mapping.items():
    print(k, "->", v)

# find maize index if present (case-insensitive)
maize_idx = None
for k,v in mapping.items():
    if v.lower() == "maize":
        maize_idx = int(k)
        break

print("\nMaize index found:", maize_idx)
