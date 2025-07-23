import pandas as pd

# Path to the preprocessed dataset
csv_path = '/Users/hj/MLAdditive/data/preprocessed.csv'

# Load the dataset
df = pd.read_csv(csv_path)

# Drop target column if present
if 'Lifetime_years' in df.columns:
    df = df.drop(columns=['Lifetime_years'])

# Save feature names
with open('expected_columns.txt', 'w') as f:
    for col in df.columns:
        f.write(f"{col}\n")

print(f"✅ Saved {len(df.columns)} feature names to 'expected_columns.txt'")
