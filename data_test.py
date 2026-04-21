import pandas as pd

# Check the Madelon file
df = pd.read_csv(r'C:\My_projects\AutoML\phpfLuQE4.csv')

print("First few rows:")
print(df.head())

print("\nColumn names:")
print(df.columns.tolist())

print(f"\nShape: {df.shape}")
print(f"Last column (likely target): '{df.columns[-1]}'")

# Check unique values in last column (should be 2 classes)
print(f"Unique values in last column: {df[df.columns[-1]].unique()}")