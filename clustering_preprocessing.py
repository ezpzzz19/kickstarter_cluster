import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt

# Load the data
df = pd.read_excel('data/Kickstarter_2025.xlsx')

print("=" * 60)
print("UNSUPERVISED CLUSTERING PREPROCESSING")
print("=" * 60)

# Check for missing values and drop columns with 90%+ missing
print("\nMISSING VALUES ANALYSIS:")
print("-" * 30)
missing_values = df.isnull().sum()
missing_percentage = (missing_values / len(df)) * 100
missing_summary = pd.DataFrame({
    'Column': missing_values.index,
    'Missing_Count': missing_values.values,
    'Missing_Percentage': missing_percentage.values
})

# Identify columns with 90%+ missing values
high_missing_columns = missing_summary[missing_summary['Missing_Percentage'] >= 90]
if len(high_missing_columns) > 0:
    print(f"Found {len(high_missing_columns)} columns with 90%+ missing values (will be dropped):")
    print(high_missing_columns.sort_values('Missing_Percentage', ascending=False))
    
    # Drop columns with 90%+ missing values
    columns_to_drop = high_missing_columns['Column'].tolist()
    df = df.drop(columns=columns_to_drop)
    print(f"Dropped {len(columns_to_drop)} columns: {columns_to_drop}")
    print(f"Remaining columns: {df.shape[1]}")

# Show remaining columns with missing values
remaining_missing = missing_summary[missing_summary['Missing_Count'] > 0]
remaining_missing = remaining_missing[~remaining_missing['Column'].isin(high_missing_columns['Column'])]
if len(remaining_missing) > 0:
    print(f"\nRemaining columns with missing values:")
    print(remaining_missing.sort_values('Missing_Percentage', ascending=False))

print("\nFEATURE ENGINEERING:")
print("-" * 30)

# Convert all monetary values to USD
df['goal_usd'] = df['goal'] * df['static_usd_rate']
df = df.drop(columns=['goal'])
df['pledged_usd'] = df['pledged'] * df['static_usd_rate']
df = df.drop(columns=['pledged'])

# Calculate funding percentage
df['funding_percentage'] = (df['pledged_usd'] / df['goal_usd']) * 100

# project duration in days
df['deadline'] = pd.to_datetime(df['deadline'])
df['created'] = pd.to_datetime(df['launched_at'])
df['project_duration'] = df['deadline'] - df['launched_at']
df['project_duration_days'] = df['project_duration'].dt.days

df = df.drop(columns=['deadline', 'launched_at', 'project_duration'])

# feature engineering season from launched_at_month
df['season'] = df['launched_at_month'].apply(lambda x: 'winter' if x in [12, 1, 2] else 'spring' if x in [3, 4, 5] else 'summer' if x in [6, 7, 8] else 'fall')
df = df.drop(columns=['launched_at_month'])
df = pd.get_dummies(df, columns=['main_category', 'season'], drop_first=True)

# for visualization purposes, we will drop the following columns:
for_clustering = df.drop(columns=['name', 'state', 'country', 'currency', 'state_changed_at', 'created_at', 'static_usd_rate', 'category', 'name_len', 'blurb_len', 'deadline_month', 'deadline_day',
 'deadline_yr', 'deadline_hr', 'state_changed_at_month', 'state_changed_at_day', 'state_changed_at_yr', 'state_changed_at_hr', 'created_at_month', 'created_at_day', 'created_at_yr', 'created_at_hr',
  'launched_at_day', 'launched_at_yr', 'launched_at_hr', 'deadline_weekday', 'state_changed_at_weekday', 'created_at_weekday', 'launched_at_weekday', 'deadline_weekday', 'state_changed_at_weekday',
   'created_at_weekday', 'launched_at_weekday', 'created'])

# write to csv
# for_clustering.to_csv('data/kickstarter_preprocessed_for_clustering.csv', index=False)

scaler = MinMaxScaler() # i tried standard scaler and it was not good, something is really skewed
scaled_data = scaler.fit_transform(for_clustering)

# visualize the scaled data in 2d using PCA
pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_data)

# plot the PCA data
plt.scatter(pca_data[:, 0], pca_data[:, 1])
plt.show()

