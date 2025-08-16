import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from matplotlib.patches import Ellipse


import matplotlib.pyplot as plt
import pycountry_convert as pc

def country_code_to_continent(country_code):
    # Convert alpha-2 to continent code
    continent_code = pc.country_alpha2_to_continent_code(country_code)
    # Map to continent name
    continents = {
        'AF': 'Africa',
        'AS': 'Asia',
        'EU': 'Europe',
        'NA': 'North America',
        'OC': 'Oceania',
        'SA': 'South America',
        'AN': 'Antarctica'
    }
    return continents[continent_code]


# Load the data
df = pd.read_excel('data/Kickstarter_2025.xlsx')

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
    print("\nRemaining columns with missing values:")
    print(remaining_missing.sort_values('Missing_Percentage', ascending=False))

print("\nFEATURE ENGINEERING:")
print("-" * 30)

# Convert all monetary values to USD
df['goal_usd'] = df['goal'] * df['static_usd_rate']

# We decided to focus on the benefits for creators and backers
# pledge amount and the gap between the goal has a direct correlation to a successful project
# we will drop the pledged column and goal column to see if we can find the parameters that lead to successful projects without these
df = df.drop(columns=['goal'])
df = df.drop(columns=['pledged'])
# df['funding_percentage'] = (df['pledged_usd'] / df['goal_usd']) * 100

# project duration in days
df['deadline'] = pd.to_datetime(df['deadline'])
df['created'] = pd.to_datetime(df['launched_at'])
df['project_duration'] = df['deadline'] - df['launched_at']
df['project_duration_days'] = df['project_duration'].dt.days
# drop the now useless columns
df = df.drop(columns=['deadline', 'launched_at', 'project_duration'])

# feature engineering season from launched_at_month
# this was tried for a few iterations but later found that it was not useful for clustering 
# all seasons are good for certain projects
df['season'] = df['launched_at_month'].apply(lambda x: 'winter' if x in [12, 1, 2] else 'spring' if x in [3, 4, 5] else 'summer' if x in [6, 7, 8] else 'fall')
df = df.drop(columns=['launched_at_month'])
df['continent'] = df['country'].apply(country_code_to_continent)

# for clustering purposes, we will drop the following columns as some of them are redundant with the featured engineered
# others limit are ability to cluster the information we want to find (clustered by state)
for_clustering = df.drop(columns=['name', 'state', 'country', 'currency', 'state_changed_at', 'created_at', 'static_usd_rate', 'category', 'name_len', 'blurb_len', 'deadline_month', 'deadline_day',
 'deadline_yr', 'deadline_hr', 'state_changed_at_month', 'state_changed_at_day', 'state_changed_at_yr', 'state_changed_at_hr', 'created_at_month', 'created_at_day', 'created_at_yr', 'created_at_hr',
  'launched_at_day', 'launched_at_yr', 'launched_at_hr', 'deadline_weekday', 'state_changed_at_weekday', 'created_at_weekday', 'launched_at_weekday', 'deadline_weekday', 'state_changed_at_weekday',
   'created_at_weekday', 'launched_at_weekday', 'created', 'main_category', 'season', 'continent', 'id'])

# for_clustering['avg_pledge_per_backer'] = for_clustering['avg_pledge_per_backer'].replace([np.inf, -np.inf], 0).fillna(0)
for_clustering = for_clustering.drop(columns=['usd_pledged', 'disable_communication', 'backers_count'])

# # write to csv
for_clustering.to_csv('data/kickstarter_preprocessed_for_clustering.csv', index=False)

scaler = MinMaxScaler() # i tried standard scaler and it was not good. This gave us better results
scaled_data = scaler.fit_transform(for_clustering)

# evaluation metric for dbscan and kmeans
def eval_metrics(X, labels):
    """Compute silhouette, pseudo-F (Calinski–Harabasz), noise ratio, n_clusters."""
    unique_labels = set(labels)
    clusters_count = len([z for z in unique_labels if z != -1])
    noise_percentage = np.mean(labels == -1)*100
    mask = labels != -1
    if clusters_count > 1:
        try:
            sil = silhouette_score(X[mask], labels[mask])
        except Exception as e:
            print(f"Error computing silhouette score, returning NaN. {e}")
            sil = np.nan
    else:
        sil = np.nan

    # Calinski–Harabasz (a.k.a. pseudo-F) needs >= 2 clusters
    if clusters_count >= 2:
        try:
            ch = calinski_harabasz_score(X, labels)
        except Exception:
            ch = np.nan
    else:
        ch = np.nan

    return {
        "n_clusters": clusters_count,
        "silhouette": sil,
        "pseudo_f": ch,     # Calinski–Harabasz
        "noise_ratio": noise_percentage
    }

# KMEANS GRID SEARCH
k_values   = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12]
n_init_val = 10  
random_state = 42
max_iter = 300
kmeans_results = {}

# evaluation method for kmeans --> shows kmeans 5 is the a great option
def elbow_method(X, k_range=range(2, 16), random_state=42, n_init=20, max_iter=500):
    inertias = []
    for k in k_range:
        km = KMeans(
            n_clusters=k,
            n_init=n_init,
            max_iter=max_iter,
            random_state=random_state
        )
        km.fit(X)
        results = eval_metrics(X, km.labels_)
        print(f"Evaluating KMeans with k={k}: silhouette: {results['silhouette']}, pseudo_f: {results['pseudo_f']}")

        inertias.append(km.inertia_)  # sum of squared distances (WCSS)
        print(f"k={k}, inertia={km.inertia_:.2f}")
    
    return inertias

print("\nKMEANS ELBOW METHOD:")
elbow_method(scaled_data)
# DBSCAN GRID SEARCH
# (MinMax-scaled features --> keep eps small)
eps_values = [0.03, 0.05, 0.1, 0.2, 0.3, 0.5]
min_samples_values = [3, 5, 10, 20]

dbscan_result = {}
for eps in eps_values:
    for ms in min_samples_values:
        db = DBSCAN(eps=eps, min_samples=ms)
        labels = db.fit_predict(scaled_data)
        metrics = eval_metrics(scaled_data, labels)
        dbscan_result[(eps, ms)] = {
            "model": db,
            "metrics": metrics,
            "labels": labels
        }
        print(f"DBSCAN with eps={eps}, min_samples={ms} - n_clusters: {metrics['n_clusters']}, silhouette: {metrics['silhouette']}, pseudo_f: {metrics['pseudo_f']}, noise_ratio: {metrics['noise_ratio']}")

# Kmeans 4 
km = KMeans(
    n_clusters=5,
      n_init=20, max_iter=500,
    random_state=random_state,
)
labels = km.fit_predict(scaled_data)
df['cluster'] = labels
state_norm = df['state'].str.lower().str.strip()
df = df.assign(state_norm=state_norm)

valid_states = ['successful', 'failed', 'cancelled']
df_sub = df[df['state_norm'].isin(valid_states)].copy()

# Overall success rate (excluding 'live') 
counts = df_sub['state_norm'].value_counts()
subset_total = counts.get('successful', 0) + counts.get('failed', 0) + counts.get('cancelled', 0)
success_rate_overall = (counts.get('successful', 0) / subset_total * 100.0) if subset_total else np.nan

print("State counts (excluding 'live'):\n", counts)
print(f"\nOverall success rate (excluding 'live'): {success_rate_overall:.2f}%")

# Per-cluster success (excluding 'live') 
success_pct = (
    df_sub.groupby("cluster")['state_norm']
    .apply(lambda s: (s.eq('successful').sum() / len(s) * 100.0) if len(s) else np.nan)
    .rename("success_pct")
    .sort_index()
)

#  Cluster feature means (all rows; keep your original behavior) 
cluster_means = (
    df.groupby("cluster")[for_clustering.columns]
      .mean()
      .sort_index()
)

#  Combine into a clean cluster profile table 
cluster_profile = (
    cluster_means
    .join(success_pct, how="left")
    .fillna(0)
)

cluster_profile_rounded = cluster_profile.copy()
cluster_profile_rounded[for_clustering.columns] = cluster_profile_rounded[for_clustering.columns].round(3)
cluster_profile_rounded['success_pct'] = cluster_profile_rounded['success_pct'].round(2)

# write clustered data to csv but cluster as column
cluster_profile_rounded.to_csv('data/kickstarter_clustered.csv', index=False)
print("\nCluster Profiles (means and success rates):")
print(cluster_profile_rounded)
# PCA for visualization
pca = PCA(n_components=2, random_state=42)
pca_result = pca.fit_transform(scaled_data)
pca_df = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2'])
pca_df['cluster'] = labels
explained_variance = pca.explained_variance_ratio_.sum() * 100
print(f"\nPCA explained variance (2 components): {explained_variance:.2f}%")