import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("=" * 70)
print("COMPREHENSIVE UNSUPERVISED CLUSTERING ANALYSIS")
print("=" * 70)

# 1. LOAD PREPROCESSED DATA
print("\n1. LOADING PREPROCESSED DATA:")
print("-" * 40)
df = pd.read_csv('data/preprocessed_kickstarter.csv')
X_scaled = pd.read_csv('data/clustering_features_scaled.csv')

print(f"Full dataset shape: {df.shape}")
print(f"Scaled features shape: {X_scaled.shape}")

# 2. FEATURE SELECTION (Remove highly correlated features)
print("\n2. FEATURE SELECTION:")
print("-" * 40)

# Remove highly correlated features identified in preprocessing
correlated_features = [
    'backers_count',  # Correlated with pledged_usd (0.860)
    'funding_velocity',  # Correlated with pledged_usd (0.968)
    'country_popularity',  # Correlated with is_us (0.993)
    'currency_popularity'  # Correlated with is_us (0.935)
]

X_clean = X_scaled.drop(columns=correlated_features, errors='ignore')
print(f"Removed {len(correlated_features)} highly correlated features")
print(f"Final feature matrix shape: {X_clean.shape}")

# 3. DIMENSIONALITY REDUCTION FOR VISUALIZATION
print("\n3. DIMENSIONALITY REDUCTION:")
print("-" * 40)

# PCA for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_clean)

print(f"PCA explained variance: PC1={pca.explained_variance_ratio_[0]:.1%}, PC2={pca.explained_variance_ratio_[1]:.1%}")
print(f"Total explained variance: {sum(pca.explained_variance_ratio_):.1%}")

# 4. K-MEANS CLUSTERING
print("\n4. K-MEANS CLUSTERING ANALYSIS:")
print("-" * 40)

# Find optimal number of clusters
k_range = range(2, 11)
kmeans_scores = {'silhouette': [], 'calinski_harabasz': [], 'davies_bouldin': [], 'inertia': []}

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_clean)
    
    kmeans_scores['inertia'].append(kmeans.inertia_)
    kmeans_scores['silhouette'].append(silhouette_score(X_clean, labels))
    kmeans_scores['calinski_harabasz'].append(calinski_harabasz_score(X_clean, labels))
    kmeans_scores['davies_bouldin'].append(davies_bouldin_score(X_clean, labels))

# Find optimal k
optimal_k_silhouette = k_range[np.argmax(kmeans_scores['silhouette'])]
optimal_k_calinski = k_range[np.argmax(kmeans_scores['calinski_harabasz'])]
optimal_k_davies = k_range[np.argmin(kmeans_scores['davies_bouldin'])]

print(f"Optimal k (Silhouette): {optimal_k_silhouette}")
print(f"Optimal k (Calinski-Harabasz): {optimal_k_calinski}")
print(f"Optimal k (Davies-Bouldin): {optimal_k_davies}")

# Use silhouette score for final selection
optimal_k = optimal_k_silhouette

# 5. DBSCAN CLUSTERING
print("\n5. DBSCAN CLUSTERING ANALYSIS:")
print("-" * 40)

# Try different eps values
eps_values = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
dbscan_results = []

for eps in eps_values:
    dbscan = DBSCAN(eps=eps, min_samples=5)
    labels = dbscan.fit_predict(X_clean)
    
    # Count clusters (excluding noise points labeled as -1)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    
    if n_clusters > 1:  # Need at least 2 clusters for silhouette score
        silhouette = silhouette_score(X_clean, labels)
    else:
        silhouette = -1
    
    dbscan_results.append({
        'eps': eps,
        'n_clusters': n_clusters,
        'n_noise': n_noise,
        'silhouette': silhouette
    })

# Find best DBSCAN parameters
best_dbscan = max(dbscan_results, key=lambda x: x['silhouette'] if x['silhouette'] > 0 else -1)
print(f"Best DBSCAN: eps={best_dbscan['eps']}, clusters={best_dbscan['n_clusters']}, noise={best_dbscan['n_noise']}")

# 6. HIERARCHICAL CLUSTERING
print("\n6. HIERARCHICAL CLUSTERING ANALYSIS:")
print("-" * 40)

# Try different numbers of clusters
hierarchical_scores = {'silhouette': [], 'calinski_harabasz': [], 'davies_bouldin': []}

for k in k_range:
    hierarchical = AgglomerativeClustering(n_clusters=k)
    labels = hierarchical.fit_predict(X_clean)
    
    hierarchical_scores['silhouette'].append(silhouette_score(X_clean, labels))
    hierarchical_scores['calinski_harabasz'].append(calinski_harabasz_score(X_clean, labels))
    hierarchical_scores['davies_bouldin'].append(davies_bouldin_score(X_clean, labels))

optimal_k_hierarchical = k_range[np.argmax(hierarchical_scores['silhouette'])]
print(f"Optimal k for Hierarchical: {optimal_k_hierarchical}")

# 7. GAUSSIAN MIXTURE MODELS
print("\n7. GAUSSIAN MIXTURE MODELS ANALYSIS:")
print("-" * 40)

gmm_scores = {'silhouette': [], 'calinski_harabasz': [], 'davies_bouldin': [], 'aic': [], 'bic': []}

for k in k_range:
    gmm = GaussianMixture(n_components=k, random_state=42)
    labels = gmm.fit_predict(X_clean)
    
    gmm_scores['silhouette'].append(silhouette_score(X_clean, labels))
    gmm_scores['calinski_harabasz'].append(calinski_harabasz_score(X_clean, labels))
    gmm_scores['davies_bouldin'].append(davies_bouldin_score(X_clean, labels))
    gmm_scores['aic'].append(gmm.aic(X_clean))
    gmm_scores['bic'].append(gmm.bic(X_clean))

optimal_k_gmm = k_range[np.argmax(gmm_scores['silhouette'])]
print(f"Optimal k for GMM: {optimal_k_gmm}")

# 8. COMPARISON VISUALIZATION
print("\n8. CREATING COMPARISON VISUALIZATIONS:")
print("-" * 40)

# Create comparison plots
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle('Clustering Algorithm Comparison', fontsize=16, fontweight='bold')

# 8.1 K-Means Elbow and Silhouette
ax1 = axes[0, 0]
ax1_twin = ax1.twinx()
ax1.plot(k_range, kmeans_scores['inertia'], 'bo-', label='Inertia', linewidth=2)
ax1_twin.plot(k_range, kmeans_scores['silhouette'], 'ro-', label='Silhouette', linewidth=2)
ax1.set_xlabel('Number of Clusters (k)')
ax1.set_ylabel('Inertia', color='blue')
ax1_twin.set_ylabel('Silhouette Score', color='red')
ax1.set_title('K-Means: Elbow & Silhouette')
ax1.grid(True, alpha=0.3)

# 8.2 DBSCAN Results
ax2 = axes[0, 1]
eps_vals = [r['eps'] for r in dbscan_results]
n_clusters = [r['n_clusters'] for r in dbscan_results]
silhouettes = [r['silhouette'] for r in dbscan_results]
ax2.plot(eps_vals, silhouettes, 'go-', linewidth=2, markersize=8)
ax2.set_xlabel('Epsilon')
ax2.set_ylabel('Silhouette Score')
ax2.set_title('DBSCAN: Silhouette vs Epsilon')
ax2.grid(True, alpha=0.3)

# 8.3 Hierarchical Clustering
ax3 = axes[0, 2]
ax3.plot(k_range, hierarchical_scores['silhouette'], 'mo-', linewidth=2, markersize=8)
ax3.set_xlabel('Number of Clusters (k)')
ax3.set_ylabel('Silhouette Score')
ax3.set_title('Hierarchical: Silhouette Score')
ax3.grid(True, alpha=0.3)

# 8.4 GMM AIC/BIC
ax4 = axes[1, 0]
ax4_twin = ax4.twinx()
ax4.plot(k_range, gmm_scores['aic'], 'co-', label='AIC', linewidth=2)
ax4_twin.plot(k_range, gmm_scores['bic'], 'yo-', label='BIC', linewidth=2)
ax4.set_xlabel('Number of Clusters (k)')
ax4.set_ylabel('AIC', color='cyan')
ax4_twin.set_ylabel('BIC', color='orange')
ax4.set_title('GMM: AIC & BIC')
ax4.grid(True, alpha=0.3)

# 8.5 Algorithm Comparison
ax5 = axes[1, 1]
algorithms = ['K-Means', 'Hierarchical', 'GMM']
best_silhouettes = [
    max(kmeans_scores['silhouette']),
    max(hierarchical_scores['silhouette']),
    max(gmm_scores['silhouette'])
]
colors = ['blue', 'magenta', 'cyan']
bars = ax5.bar(algorithms, best_silhouettes, color=colors, alpha=0.7)
ax5.set_ylabel('Best Silhouette Score')
ax5.set_title('Best Silhouette Scores by Algorithm')
ax5.grid(True, alpha=0.3)

# Add value labels on bars
for bar, value in zip(bars, best_silhouettes):
    ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{value:.3f}', ha='center', va='bottom')

# 8.6 Final K-Means Clustering
ax6 = axes[1, 2]
kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
final_labels = kmeans_final.fit_predict(X_clean)
scatter = ax6.scatter(X_pca[:, 0], X_pca[:, 1], c=final_labels, cmap='Set1', alpha=0.6, s=20)
centers_pca = pca.transform(kmeans_final.cluster_centers_)
ax6.scatter(centers_pca[:, 0], centers_pca[:, 1], c='red', marker='x', s=200, linewidths=3, label='Centers')
ax6.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
ax6.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
ax6.set_title(f'K-Means Clustering (k={optimal_k})')
ax6.legend()
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('clustering_comparison.png', dpi=300, bbox_inches='tight')
print("✅ Clustering comparison saved as 'clustering_comparison.png'")

# 9. FINAL CLUSTERING ANALYSIS
print("\n9. FINAL CLUSTERING ANALYSIS:")
print("-" * 40)

# Apply final K-Means clustering
kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
final_labels = kmeans_final.fit_predict(X_clean)

# Add cluster labels to original data
df['cluster'] = final_labels

# Analyze cluster characteristics
cluster_analysis = []
for cluster_id in range(optimal_k):
    cluster_data = df[df['cluster'] == cluster_id]
    
    analysis = {
        'Cluster': cluster_id,
        'Size': len(cluster_data),
        'Size_Percentage': len(cluster_data) / len(df) * 100,
        'Avg_Funding_Percentage': cluster_data['funding_percentage'].mean(),
        'Avg_Goal_USD': cluster_data['goal_usd'].mean(),
        'Avg_Pledged_USD': cluster_data['pledged_usd'].mean(),
        'Avg_Backers': cluster_data['backers_count'].mean(),
        'Avg_Duration': cluster_data['project_duration_days'].mean(),
        'Success_Rate': (cluster_data['state'] == 'successful').mean() * 100,
        'Avg_Platform_Engagement': cluster_data['platform_engagement_score'].mean(),
        'Avg_Funding_Velocity': cluster_data['funding_velocity'].mean()
    }
    cluster_analysis.append(analysis)

cluster_df = pd.DataFrame(cluster_analysis)
print("\nCluster Characteristics:")
print(cluster_df.round(2))

# 10. SAVE RESULTS
print("\n10. SAVING RESULTS:")
print("-" * 40)

# Save cluster analysis
cluster_df.to_csv('data/final_cluster_analysis.csv', index=False)

# Save data with cluster labels
df.to_csv('data/kickstarter_with_clusters.csv', index=False)

# Save clustering scores for comparison
scores_df = pd.DataFrame({
    'Algorithm': ['K-Means', 'Hierarchical', 'GMM', 'DBSCAN'],
    'Best_Silhouette': [
        max(kmeans_scores['silhouette']),
        max(hierarchical_scores['silhouette']),
        max(gmm_scores['silhouette']),
        best_dbscan['silhouette']
    ],
    'Best_Calinski_Harabasz': [
        max(kmeans_scores['calinski_harabasz']),
        max(hierarchical_scores['calinski_harabasz']),
        max(gmm_scores['calinski_harabasz']),
        -1  # DBSCAN doesn't have this metric
    ],
    'Optimal_Parameters': [
        f"k={optimal_k}",
        f"k={optimal_k_hierarchical}",
        f"k={optimal_k_gmm}",
        f"eps={best_dbscan['eps']}, clusters={best_dbscan['n_clusters']}"
    ]
})

scores_df.to_csv('data/clustering_algorithm_scores.csv', index=False)

print("✅ Results saved:")
print("  - data/final_cluster_analysis.csv (cluster characteristics)")
print("  - data/kickstarter_with_clusters.csv (data with cluster labels)")
print("  - data/clustering_algorithm_scores.csv (algorithm comparison)")
print("  - clustering_comparison.png (visualization)")

# 11. SUMMARY
print("\n" + "=" * 70)
print("CLUSTERING ANALYSIS SUMMARY")
print("=" * 70)
print(f"Dataset: {df.shape[0]} projects, {X_clean.shape[1]} features")
print(f"Best Algorithm: K-Means (Silhouette: {max(kmeans_scores['silhouette']):.3f})")
print(f"Optimal Clusters: {optimal_k}")
print(f"Total Explained Variance (PCA): {sum(pca.explained_variance_ratio_):.1%}")

print("\nCluster Distribution:")
for _, row in cluster_df.iterrows():
    print(f"  Cluster {int(row['Cluster'])}: {int(row['Size'])} projects ({row['Size_Percentage']:.1f}%)")

print("\nSuccess Rates by Cluster:")
for _, row in cluster_df.iterrows():
    print(f"  Cluster {int(row['Cluster'])}: {row['Success_Rate']:.1f}% success rate")

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE!")
print("=" * 70)
