import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

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

# missing main category with mode
df['main_category'] = df['main_category'].fillna(df['main_category'].mode()[0])

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
# df['pledged_usd'] = df['pledged'] * df['static_usd_rate']
df = df.drop(columns=['pledged'])

# # Calculate funding percentage
# df['funding_percentage'] = (df['pledged_usd'] / df['goal_usd']) * 100

# project duration in days
df['deadline'] = pd.to_datetime(df['deadline'])
df['created'] = pd.to_datetime(df['launched_at'])
df['project_duration'] = df['deadline'] - df['launched_at']
df['project_duration_days'] = df['project_duration'].dt.days


df = df.drop(columns=['deadline', 'launched_at', 'project_duration'])

# feature engineering season from launched_at_month
df['season'] = df['launched_at_month'].apply(lambda x: 'winter' if x in [12, 1, 2] else 'spring' if x in [3, 4, 5] else 'summer' if x in [6, 7, 8] else 'fall')
df = df.drop(columns=['launched_at_month'])
# df = pd.get_dummies(df, columns=['main_category', 'season'], drop_first=True)

# df['avg_pledge_per_backer'] = df['usd_pledged'] / df['backers_count']
# df['avg_pledge_per_day'] = df['usd_pledged'] / df['project_duration_days']
#df['avg_duration_of_campaign'] = df['project_duration_days'] / df['backers_count']

# for visualization purposes, we will drop the following columns:
for_clustering = df.drop(columns=['name', 'state', 'country', 'currency', 'state_changed_at', 'created_at', 'static_usd_rate', 'category', 'name_len', 'blurb_len', 'deadline_month', 'deadline_day',
 'deadline_yr', 'deadline_hr', 'state_changed_at_month', 'state_changed_at_day', 'state_changed_at_yr', 'state_changed_at_hr', 'created_at_month', 'created_at_day', 'created_at_yr', 'created_at_hr',
  'launched_at_day', 'launched_at_yr', 'launched_at_hr', 'deadline_weekday', 'state_changed_at_weekday', 'created_at_weekday', 'launched_at_weekday', 'deadline_weekday', 'state_changed_at_weekday',
   'created_at_weekday', 'launched_at_weekday', 'created', 'main_category', 'season'])

# for_clustering['avg_pledge_per_backer'] = for_clustering['avg_pledge_per_backer'].replace([np.inf, -np.inf], 0).fillna(0)
for_clustering = for_clustering.drop(columns=['usd_pledged', 'disable_communication', 'backers_count'])

# count of values for the columns
print(f'disable_communication: {df["disable_communication"].value_counts()}')
print(f'staff_pick.1 : {df["staff_pick.1"].value_counts()}')


# avg pledge per backer
# avg pledge per day 
# avg duration of campaign 
# season 
# main category

# # write to csv
for_clustering.to_csv('data/kickstarter_preprocessed_for_clustering.csv', index=False)

scaler = MinMaxScaler() # i tried standard scaler and it was not good, something is really skewed
scaled_data = scaler.fit_transform(for_clustering)

# Run PCA
pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_data)

# Create figure with 2 subplots
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# PCA by State
sns.scatterplot(
    x=pca_data[:, 0], 
    y=pca_data[:, 1], 
    hue=df['state'],
    palette='tab10',
    alpha=0.7,
    ax=axes[0]
)
axes[0].set_title('PCA Projection by State')
axes[0].set_xlabel('PC1')
axes[0].set_ylabel('PC2')
axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# PCA by Country
sns.scatterplot(
    x=pca_data[:, 0], 
    y=pca_data[:, 1], 
    hue=df['country'],
    palette='tab20',
    alpha=0.7,
    ax=axes[1]
)
axes[1].set_title('PCA Projection by Country')
axes[1].set_xlabel('PC1')
axes[1].set_ylabel('PC2')
axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.show()

