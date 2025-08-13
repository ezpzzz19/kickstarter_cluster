import pandas as pd

# Load the original data
df = pd.read_excel('data/Kickstarter_2025.xlsx')

print("=" * 60)
print("KICKSTARTER DATASET COLUMNS")
print("=" * 60)

print(f"Total number of columns: {len(df.columns)}")
print(f"Total number of rows: {len(df)}")
print()

print("All column names:")
print("-" * 30)
for i, col in enumerate(df.columns, 1):
    print(f"{i:2d}. {col}")

print()
print("=" * 60)
print("COLUMN DATA TYPES")
print("=" * 60)
print(df.dtypes)

print()
print("=" * 60)
print("SAMPLE DATA (first 3 rows)")
print("=" * 60)
print(df.head(3))
