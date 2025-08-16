import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import matplotlib.pyplot as plt
import pycountry_convert as pc


def country_code_to_continent(country_code):
    """Convert country code to continent name"""
    try:
        continent_code = pc.country_alpha2_to_continent_code(country_code)
        continents = {
            "AF": "Africa",
            "AS": "Asia",
            "EU": "Europe",
            "NA": "North America",
            "OC": "Oceania",
            "SA": "South America",
            "AN": "Antarctica",
        }
        return continents[continent_code]
    except Exception as e:
        print(f"Error converting country code {country_code}: {e}")
        return "Unknown"


def resample_to_realistic_distribution(df, target_success_rate=0.42, random_state=42):
    """
    Resample the dataset to reflect realistic Kickstarter success rates.

    Args:
        df: Original dataframe
        target_success_rate: Target proportion of successful projects (default: 30%)
        target_failure_rate: Target proportion of failed projects (default: 61%)
        random_state: Random seed for reproducibility

    Returns:
        Resampled dataframe with realistic success/failure distribution
    """
    print("\nResampling dataset to realistic Kickstarter distribution:")
    print(
        f"Target: {target_success_rate * 100:.0f}% success, {(1 - target_success_rate) * 100:.0f}% failure"
    )

    # Get current distribution
    current_dist = df["state"].value_counts(normalize=True)
    print(f"Current distribution: {current_dist.to_dict()}")

    # Calculate target counts
    total_projects = len(df)
    target_success_count = int(total_projects * target_success_rate)
    target_failure_count = int(total_projects * (1 - target_success_rate))

    print(
        f"Target counts: {target_success_count} success, {target_failure_count} failure"
    )

    # Get current counts
    current_success = df[df["state"] == "successful"]
    current_failure = df[df["state"] == "failed"]

    # Resample each category
    np.random.seed(random_state)

    # For successful projects: downsample if too many, upsample if too few
    if len(current_success) > target_success_count:
        # Downsample successful projects
        success_sample = current_success.sample(
            n=target_success_count, random_state=random_state
        )
    else:
        # Upsample successful projects with replacement
        success_sample = current_success.sample(
            n=target_success_count, replace=True, random_state=random_state
        )

    # For failed projects: downsample if too many, upsample if too few
    if len(current_failure) > target_failure_count:
        # Downsample failed projects
        failure_sample = current_failure.sample(
            n=target_failure_count, random_state=random_state
        )
    else:
        # Upsample failed projects with replacement
        failure_sample = current_failure.sample(
            n=target_failure_count, replace=True, random_state=random_state
        )

    # Combine resampled data
    resampled_df = pd.concat([success_sample, failure_sample], ignore_index=True)

    # Shuffle the data
    resampled_df = resampled_df.sample(frac=1, random_state=random_state).reset_index(
        drop=True
    )

    # Verify new distribution
    new_dist = resampled_df["state"].value_counts(normalize=True)
    print(f"New distribution: {new_dist.to_dict()}")
    print(f"Resampled dataset shape: {resampled_df.shape}")

    return resampled_df


def analyze_pre_campaign_features():
    """
    Analyze and cluster Kickstarter projects using only pre-campaign features
    that are good indicators of success vs failure, avoiding post-campaign data leakage.
    """

    print("=" * 80)
    print("PRE-CAMPAIGN CLUSTERING ANALYSIS")
    print("=" * 80)

    # Load the data
    df = pd.read_excel("data/Kickstarter_2025.xlsx")

    print(f"Original dataset shape: {df.shape}")
    print("State distribution:")
    print(df["state"].value_counts())
    # remove all other states but 'successful', 'failed'
    df = df[df["state"].isin(["successful", "failed"])]
    print(f"Filtered dataset shape: {df.shape}")

    # Resample to realistic distribution
    df = resample_to_realistic_distribution(
        df, target_success_rate=0.42
    )  # citation from kickstater.com

    # ============================================================================
    # PRE-CAMPAIGN FEATURE SELECTION
    # ============================================================================

    print("\n" + "=" * 50)
    print("PRE-CAMPAIGN FEATURE SELECTION")
    print("=" * 50)

    # Features that are available BEFORE the campaign starts (no data leakage):

    # ============================================================================
    # FEATURE ENGINEERING
    # ============================================================================

    print("\nFeature Engineering:")
    print("-" * 30)

    # Convert goal to USD for consistency
    df["goal_usd"] = df["goal"] * df["static_usd_rate"]

    # Calculate project duration
    df["deadline"] = pd.to_datetime(df["deadline"], errors="coerce")
    df["launched_at"] = pd.to_datetime(df["launched_at"], errors="coerce")
    df["project_duration_days"] = (df["deadline"] - df["launched_at"]).dt.days

    # Add continent feature
    df["continent"] = df["country"].apply(country_code_to_continent)

    # Coalesce staff_pick fields - use staff_pick if True, otherwise use staff_pick.1
    df["staff_pick_coalesced"] = df["staff_pick"] | df["staff_pick.1"]
    print("\nStaff pick analysis:")
    print(
        f"staff_pick True: {df['staff_pick'].sum()} ({df['staff_pick'].mean() * 100:.1f}%)"
    )
    print(
        f"staff_pick.1 True: {df['staff_pick.1'].sum()} ({df['staff_pick.1'].mean() * 100:.1f}%)"
    )
    print(
        f"Coalesced True: {df['staff_pick_coalesced'].sum()} ({df['staff_pick_coalesced'].mean() * 100:.1f}%)"
    )

    # Create goal tiers (useful for clustering)
    df["goal_tier"] = pd.cut(
        df["goal_usd"],
        bins=[0, 1000, 5000, 10000, 50000, 100000, np.inf],
        labels=["Micro", "Small", "Medium", "Large", "Very Large", "Mega"],
    )

    # Create duration tiers
    df["duration_tier"] = pd.cut(
        df["project_duration_days"],
        bins=[0, 15, 30, 45, 60, np.inf],
        labels=["Very Short", "Short", "Medium", "Long", "Very Long"],
    )

    # Seasonality features
    df["launch_season"] = df["launched_at_month"].apply(
        lambda x: "winter"
        if x in [12, 1, 2]
        else "spring"
        if x in [3, 4, 5]
        else "summer"
        if x in [6, 7, 8]
        else "fall"
    )

    # ============================================================================
    # DATA PREPARATION FOR CLUSTERING
    # ============================================================================

    print("\nPreparing data for clustering:")
    print("-" * 30)

    # Select features for clustering (excluding post-campaign features)
    clustering_features = [
        "goal_usd",  # Funding goal in USD
        "name_len_clean",  # Cleaned name length
        "blurb_len_clean",  # Cleaned description length
        "project_duration_days",  # Campaign duration
        "staff_pick_coalesced",  # Coalesced staff pick status
        "show_feature_image",  # Has feature image
        "video",  # Has video
    ]

    # Create clustering dataset
    clustering_df = df[clustering_features].copy()

    # Handle missing values
    print("Missing values in clustering features:")
    missing_summary = clustering_df.isnull().sum()
    print(missing_summary[missing_summary > 0])

    # Fill missing values with appropriate strategies
    clustering_df["goal_usd"] = clustering_df["goal_usd"].fillna(
        clustering_df["goal_usd"].median()
    )
    clustering_df["name_len_clean"] = clustering_df["name_len_clean"].fillna(
        clustering_df["name_len_clean"].median()
    )
    clustering_df["blurb_len_clean"] = clustering_df["blurb_len_clean"].fillna(
        clustering_df["blurb_len_clean"].median()
    )
    clustering_df["project_duration_days"] = clustering_df[
        "project_duration_days"
    ].fillna(clustering_df["project_duration_days"].median())

    # Boolean features should be False if missing
    clustering_df["staff_pick_coalesced"] = clustering_df[
        "staff_pick_coalesced"
    ].fillna(False)
    clustering_df["show_feature_image"] = clustering_df["show_feature_image"].fillna(
        False
    )
    clustering_df["video"] = clustering_df["video"].fillna(False)

    print(f"Final clustering dataset shape: {clustering_df.shape}")

    # ============================================================================
    # FEATURE SCALING
    # ============================================================================

    print("\nScaling features...")
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(clustering_df)

    # ============================================================================
    # CLUSTERING ANALYSIS
    # ============================================================================

    print("\n" + "=" * 50)
    print("CLUSTERING ANALYSIS")
    print("=" * 50)

    # Test different numbers of clusters
    k_values = range(2, 11)
    inertias = []
    silhouette_scores = []
    calinski_scores = []

    print("\nEvaluating different numbers of clusters:")
    print("-" * 40)

    for k in k_values:
        kmeans = KMeans(n_clusters=k, n_init=20, max_iter=500, random_state=42)
        labels = kmeans.fit_predict(scaled_data)

        inertia = kmeans.inertia_
        inertias.append(inertia)

        # Calculate metrics
        if k > 1:
            sil_score = silhouette_score(scaled_data, labels)
            cal_score = calinski_harabasz_score(scaled_data, labels)
        else:
            sil_score = 0
            cal_score = 0

        silhouette_scores.append(sil_score)
        calinski_scores.append(cal_score)

        print(
            f"k={k:2d}: Inertia={inertia:10.2f}, Silhouette={sil_score:6.3f}, Calinski={cal_score:8.1f}"
        )

    # Find optimal k based on silhouette score
    optimal_k_silhouette = silhouette_scores.index(max(silhouette_scores)) + 2
    print(f"\nOptimal k based on silhouette score: {optimal_k_silhouette}")

    # ============================================================================
    # ELBOW METHOD VISUALIZATION
    # ============================================================================

    print("\n" + "=" * 50)
    print("ELBOW METHOD VISUALIZATION")
    print("=" * 50)

    # Create elbow method plot
    plt.figure(figsize=(12, 5))

    # Subplot 1: Elbow Method (Inertia)
    plt.subplot(1, 2, 1)
    plt.plot(k_values, inertias, "bo-", linewidth=2, markersize=8)
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Inertia (Within-Cluster Sum of Squares)")
    plt.title("Elbow Method for Optimal k")
    plt.grid(True, alpha=0.3)

    # Add annotation for optimal k
    plt.annotate(
        f"Optimal k={optimal_k_silhouette}",
        xy=(optimal_k_silhouette, inertias[optimal_k_silhouette - 2]),
        xytext=(optimal_k_silhouette + 1, inertias[optimal_k_silhouette - 2] * 0.8),
        arrowprops=dict(arrowstyle="->", color="red", lw=2),
        fontsize=12,
        color="red",
        weight="bold",
    )

    # Subplot 2: Silhouette Score
    plt.subplot(1, 2, 2)
    plt.plot(k_values, silhouette_scores, "ro-", linewidth=2, markersize=8)
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Silhouette Score")
    plt.title("Silhouette Score vs Number of Clusters")
    plt.grid(True, alpha=0.3)

    # Add annotation for optimal k
    plt.annotate(
        f"Optimal k={optimal_k_silhouette}",
        xy=(optimal_k_silhouette, max(silhouette_scores)),
        xytext=(optimal_k_silhouette + 1, max(silhouette_scores) * 0.9),
        arrowprops=dict(arrowstyle="->", color="red", lw=2),
        fontsize=12,
        color="red",
        weight="bold",
    )

    plt.tight_layout()
    plt.savefig("elbow_method_analysis.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("Elbow method visualization saved as 'elbow_method_analysis.png'")

    # Use the optimal k based on silhouette score
    optimal_k = optimal_k_silhouette
    print(
        f"Using k={optimal_k} for final clustering (optimal based on silhouette score)"
    )

    # ============================================================================
    # FINAL CLUSTERING
    # ============================================================================

    print(f"\nPerforming final clustering with k={optimal_k}...")

    final_kmeans = KMeans(
        n_clusters=optimal_k, n_init=20, max_iter=500, random_state=42
    )
    cluster_labels = final_kmeans.fit_predict(scaled_data)

    # Add cluster labels to original dataframe
    df["cluster"] = cluster_labels

    # ============================================================================
    # CLUSTER ANALYSIS
    # ============================================================================

    print("\n" + "=" * 50)
    print("CLUSTER ANALYSIS")
    print("=" * 50)

    # Analyze success rates by cluster
    success_analysis = (
        df.groupby("cluster")["state"].value_counts().unstack(fill_value=0)
    )
    success_analysis["total"] = success_analysis.sum(axis=1)
    success_analysis["success_rate"] = (
        success_analysis["successful"] / success_analysis["total"] * 100
    ).round(2)
    success_analysis["failure_rate"] = (
        success_analysis["failed"] / success_analysis["total"] * 100
    ).round(2)

    print("\nCluster Success Analysis:")
    print(success_analysis)

    # Analyze cluster characteristics
    cluster_means = df.groupby("cluster")[clustering_features].mean()
    cluster_means["size"] = df.groupby("cluster").size()
    cluster_means["success_rate"] = success_analysis["success_rate"]

    print("\nCluster Characteristics (means):")
    print(cluster_means.round(2))

    # ============================================================================
    # VISUALIZATION
    # ============================================================================

    print("\n" + "=" * 50)
    print("VISUALIZATION")
    print("=" * 50)

    # PCA for visualization
    pca = PCA(n_components=2, random_state=42)
    pca_data = pca.fit_transform(scaled_data)

    # Create visualization dataframe
    viz_df = pd.DataFrame(pca_data, columns=["PC1", "PC2"])
    viz_df["cluster"] = cluster_labels
    viz_df["state"] = df["state"].values

    # Plot clusters
    plt.figure(figsize=(15, 6))

    # Subplot 1: Clusters
    plt.subplot(1, 2, 1)
    for cluster in range(optimal_k):
        mask = viz_df["cluster"] == cluster
        plt.scatter(
            viz_df.loc[mask, "PC1"],
            viz_df.loc[mask, "PC2"],
            alpha=0.6,
            label=f"Cluster {cluster}",
        )

    plt.title(f"Kickstarter Projects Clusters (k={optimal_k})")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Subplot 2: Success rates by cluster
    plt.subplot(1, 2, 2)
    success_rates = success_analysis["success_rate"].values
    cluster_sizes = success_analysis["total"].values

    bars = plt.bar(range(optimal_k), success_rates, alpha=0.7)
    plt.title("Success Rate by Cluster")
    plt.xlabel("Cluster")
    plt.ylabel("Success Rate (%)")
    plt.xticks(range(optimal_k))

    # Add size annotations
    for i, (bar, size) in enumerate(zip(bars, cluster_sizes)):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"n={size}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("pre_campaign_clusters.png", dpi=300, bbox_inches="tight")
    plt.close()  # Close the plot to free memory and continue execution

    # ============================================================================
    # FEATURE IMPORTANCE ANALYSIS
    # ============================================================================

    print("\n" + "=" * 50)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("=" * 50)

    # Analyze which features are most important for success
    feature_importance = {}

    for feature in clustering_features:
        # Calculate correlation with success
        success_binary = (df["state"] == "successful").astype(int)
        correlation = np.corrcoef(df[feature], success_binary)[0, 1]
        feature_importance[feature] = abs(correlation)

    # Sort by importance
    feature_importance = dict(
        sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    )

    print("\nFeature importance for success prediction:")
    for feature, importance in feature_importance.items():
        print(f"{feature:25s}: {importance:.4f}")

    # ============================================================================
    # COMPREHENSIVE CLUSTER SUMMARY
    # ============================================================================

    print("\n" + "=" * 50)
    print("COMPREHENSIVE CLUSTER SUMMARY")
    print("=" * 50)

    # Create comprehensive cluster summary
    cluster_summary = {}

    for cluster in range(optimal_k):
        cluster_data = df[df["cluster"] == cluster]

        # Basic statistics
        cluster_summary[f"cluster_{cluster}"] = {
            "cluster_id": cluster,
            "size": len(cluster_data),
            "percentage_of_total": len(cluster_data) / len(df) * 100,
            # Success metrics
            "success_count": len(cluster_data[cluster_data["state"] == "successful"]),
            "failure_count": len(cluster_data[cluster_data["state"] == "failed"]),
            "success_rate": len(cluster_data[cluster_data["state"] == "successful"])
            / len(cluster_data)
            * 100,
            "failure_rate": len(cluster_data[cluster_data["state"] == "failed"])
            / len(cluster_data)
            * 100,
            # Feature means
            "avg_goal_usd": cluster_data["goal_usd"].mean(),
            "avg_name_length": cluster_data["name_len_clean"].mean(),
            "avg_blurb_length": cluster_data["blurb_len_clean"].mean(),
            "avg_duration_days": cluster_data["project_duration_days"].mean(),
            # Boolean feature percentages
            "pct_staff_pick": cluster_data["staff_pick_coalesced"].mean() * 100,
            "pct_has_feature_image": cluster_data["show_feature_image"].mean() * 100,
            "pct_has_video": cluster_data["video"].mean() * 100,
            # Goal tiers distribution
            "pct_micro_goal": len(cluster_data[cluster_data["goal_tier"] == "Micro"])
            / len(cluster_data)
            * 100,
            "pct_small_goal": len(cluster_data[cluster_data["goal_tier"] == "Small"])
            / len(cluster_data)
            * 100,
            "pct_medium_goal": len(cluster_data[cluster_data["goal_tier"] == "Medium"])
            / len(cluster_data)
            * 100,
            "pct_large_goal": len(cluster_data[cluster_data["goal_tier"] == "Large"])
            / len(cluster_data)
            * 100,
            "pct_very_large_goal": len(
                cluster_data[cluster_data["goal_tier"] == "Very Large"]
            )
            / len(cluster_data)
            * 100,
            "pct_mega_goal": len(cluster_data[cluster_data["goal_tier"] == "Mega"])
            / len(cluster_data)
            * 100,
            # Duration tiers distribution
            "pct_very_short_duration": len(
                cluster_data[cluster_data["duration_tier"] == "Very Short"]
            )
            / len(cluster_data)
            * 100,
            "pct_short_duration": len(
                cluster_data[cluster_data["duration_tier"] == "Short"]
            )
            / len(cluster_data)
            * 100,
            "pct_medium_duration": len(
                cluster_data[cluster_data["duration_tier"] == "Medium"]
            )
            / len(cluster_data)
            * 100,
            "pct_long_duration": len(
                cluster_data[cluster_data["duration_tier"] == "Long"]
            )
            / len(cluster_data)
            * 100,
            "pct_very_long_duration": len(
                cluster_data[cluster_data["duration_tier"] == "Very Long"]
            )
            / len(cluster_data)
            * 100,
            # Season distribution
            "pct_winter_launch": len(
                cluster_data[cluster_data["launch_season"] == "winter"]
            )
            / len(cluster_data)
            * 100,
            "pct_spring_launch": len(
                cluster_data[cluster_data["launch_season"] == "spring"]
            )
            / len(cluster_data)
            * 100,
            "pct_summer_launch": len(
                cluster_data[cluster_data["launch_season"] == "summer"]
            )
            / len(cluster_data)
            * 100,
            "pct_fall_launch": len(
                cluster_data[cluster_data["launch_season"] == "fall"]
            )
            / len(cluster_data)
            * 100,
            # Continent distribution
            "pct_north_america": len(
                cluster_data[cluster_data["continent"] == "North America"]
            )
            / len(cluster_data)
            * 100,
            "pct_europe": len(cluster_data[cluster_data["continent"] == "Europe"])
            / len(cluster_data)
            * 100,
            "pct_asia": len(cluster_data[cluster_data["continent"] == "Asia"])
            / len(cluster_data)
            * 100,
            "pct_oceania": len(cluster_data[cluster_data["continent"] == "Oceania"])
            / len(cluster_data)
            * 100,
            "pct_south_america": len(
                cluster_data[cluster_data["continent"] == "South America"]
            )
            / len(cluster_data)
            * 100,
            "pct_africa": len(cluster_data[cluster_data["continent"] == "Africa"])
            / len(cluster_data)
            * 100,
        }

    # Convert to DataFrame
    cluster_summary_df = pd.DataFrame(cluster_summary).T

    # Round numeric columns
    numeric_columns = cluster_summary_df.select_dtypes(include=[np.number]).columns
    cluster_summary_df[numeric_columns] = cluster_summary_df[numeric_columns].round(2)

    print("\nComprehensive Cluster Summary:")
    print(
        cluster_summary_df[
            [
                "cluster_id",
                "size",
                "success_rate",
                "failure_rate",
                "avg_goal_usd",
                "avg_duration_days",
                "pct_staff_pick",
                "pct_has_video",
                "pct_north_america",
                "pct_europe",
            ]
        ]
    )

    # ============================================================================
    # SAVE RESULTS
    # ============================================================================

    print("\n" + "=" * 50)
    print("SAVING RESULTS")
    print("=" * 50)

    # Save clustered data
    output_df = df[["id", "name", "state", "cluster"] + clustering_features].copy()
    output_df.to_csv("data/kickstarter_pre_campaign_clustered.csv", index=False)

    # Save cluster profiles
    cluster_profiles = cluster_means.copy()
    cluster_profiles.to_csv("data/pre_campaign_cluster_profiles.csv")

    # Save comprehensive cluster summary
    cluster_summary_df.to_csv("data/comprehensive_cluster_summary.csv")

    print("Results saved to:")
    print("- data/kickstarter_pre_campaign_clustered.csv")
    print("- data/pre_campaign_cluster_profiles.csv")
    print("- data/comprehensive_cluster_summary.csv")
    print("- pre_campaign_clusters.png")
    print("- elbow_method_analysis.png")

    # ============================================================================
    # SUMMARY
    # ============================================================================

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(f"✓ Analyzed {len(df)} Kickstarter projects using pre-campaign features only")
    print("✓ Resampled dataset to reflect realistic success rates (~30% success)")
    print(f"✓ Identified {optimal_k} distinct clusters")
    print(
        "✓ Avoided post-campaign data leakage (no pledged, backers_count, spotlight, etc.)"
    )
    print("✓ Used features available before campaign launch:")
    for feature in clustering_features:
        print(f"  - {feature}")

    print(
        f"\nBest performing cluster: Cluster {success_analysis['success_rate'].idxmax()} "
        f"({success_analysis['success_rate'].max():.1f}% success rate)"
    )
    print(
        f"Worst performing cluster: Cluster {success_analysis['success_rate'].idxmin()} "
        f"({success_analysis['success_rate'].min():.1f}% success rate)"
    )

    return df, clustering_df, cluster_means, success_analysis


if __name__ == "__main__":
    # Run the analysis
    df, clustering_df, cluster_means, success_analysis = analyze_pre_campaign_features()
