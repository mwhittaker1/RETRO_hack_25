# Customer Return Clustering Analysis
# Hybrid DBSCAN -> K-means -> sub-DBSCAN approach for customer segmentation

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import warnings
import json
from pathlib import Path

warnings.filterwarnings('ignore')
plt.style.use('default')
sns.set_palette("husl")

from db import get_connection

# %%
# =============================================================================
# PHASE 4: SUB-CLUSTERING WITH DBSCAN
# =============================================================================

def run_subclustering_dbscan(X, main_labels, config):
    """Run DBSCAN within each K-means cluster for refinement"""
    
    print("Phase 4: Running sub-clustering DBSCAN within K-means clusters...")
    print(f"Sub-DBSCAN parameters: eps={config['eps']}, min_samples={config['min_samples']}")
    
    # Initialize sub-cluster labels
    sub_labels = np.copy(main_labels)
    sub_cluster_info = {}
    
    # Get unique K-means clusters (excluding noise)
    kmeans_clusters = np.unique(main_labels[main_labels != -1])
    
    for cluster_id in kmeans_clusters:
        print(f"\n  Processing K-means cluster {cluster_id}...")
        
        # Get points in this cluster
        cluster_mask = main_labels == cluster_id
        cluster_indices = np.where(cluster_mask)[0]
        X_cluster = X[cluster_mask]
        
        print(f"    Cluster size: {len(X_cluster):,} points")
        
        # Skip if cluster is too small
        min_size_for_subclustering = config['min_samples'] * 3
        if len(X_cluster) < min_size_for_subclustering:
            print(f"    Skipping (too small for sub-clustering)")
            sub_cluster_info[cluster_id] = {
                'sub_clusters': 0,
                'noise_points': 0,
                'original_size': len(X_cluster)
            }
            continue
        
        # Run DBSCAN on this cluster
        dbscan_sub = DBSCAN(
            eps=config['eps'],
            min_samples=config['min_samples'],
            algorithm=config['algorithm'],
            n_jobs=config['n_jobs']
        )
        
        sub_cluster_labels = dbscan_sub.fit_predict(X_cluster)
        
        # Count sub-clusters and noise
        n_sub_clusters = len(set(sub_cluster_labels)) - (1 if -1 in sub_cluster_labels else 0)
        n_sub_noise = list(sub_cluster_labels).count(-1)
        
        print(f"    Sub-clusters found: {n_sub_clusters}")
        print(f"    Noise points: {n_sub_noise}")
        
        # Update labels with sub-cluster information
        if n_sub_clusters > 1:  # Only update if we found meaningful sub-clusters
            for i, sub_label in enumerate(sub_cluster_labels):
                if sub_label != -1:
                    # Create unique sub-cluster ID: cluster_id * 100 + sub_label
                    sub_labels[cluster_indices[i]] = cluster_id * 100 + sub_label
                # Keep noise points with original cluster label
        
        sub_cluster_info[cluster_id] = {
            'sub_clusters': n_sub_clusters,
            'noise_points': n_sub_noise,
            'original_size': len(X_cluster),
            'sub_cluster_sizes': dict(zip(*np.unique(sub_cluster_labels[sub_cluster_labels != -1], return_counts=True))) if n_sub_clusters > 0 else {}
        }
    
    # Print summary
    total_sub_clusters = sum(info['sub_clusters'] for info in sub_cluster_info.values())
    total_original_clusters = len(kmeans_clusters)
    
    print(f"\nSub-clustering Summary:")
    print(f"  Original K-means clusters: {total_original_clusters}")
    print(f"  Total sub-clusters created: {total_sub_clusters}")
    print(f"  Final cluster count: {len(np.unique(sub_labels[sub_labels != -1]))}")
    
    return sub_labels, sub_cluster_info

# Run sub-clustering
sub_cluster_labels, sub_cluster_info = run_subclustering_dbscan(
    X_cluster, 
    final_labels, 
    CLUSTERING_CONFIG['dbscan_subclusters']
)

# Visualize sub-clustering results
def plot_subclustering_results(X_2d, main_labels, sub_labels, title, method_name):
    """Plot comparison between main clustering and sub-clustering"""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Original K-means clusters
    axes[0].set_title(f'K-means Clusters - {method_name}')
    unique_main = np.unique(main_labels)
    colors_main = plt.cm.tab10(np.linspace(0, 1, len(unique_main)))
    
    for label, color in zip(unique_main, colors_main):
        mask = main_labels == label
        if label == -1:
            axes[0].scatter(X_2d[mask, 0], X_2d[mask, 1], c='lightgray', 
                          marker='x', s=20, alpha=0.6, label='Noise')
        else:
            axes[0].scatter(X_2d[mask, 0], X_2d[mask, 1], c=[color], 
                          s=30, alpha=0.7, label=f'Cluster {label}')
    
    axes[0].set_xlabel(f'{method_name} Component 1')
    axes[0].set_ylabel(f'{method_name} Component 2')
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Sub-clusters
    axes[1].set_title(f'Sub-clusters - {method_name}')
    unique_sub = np.unique(sub_labels)
    
    # Use a larger color palette for sub-clusters
    n_colors = len(unique_sub)
    colors_sub = plt.cm.Set3(np.linspace(0, 1, n_colors))
    
    for label, color in zip(unique_sub, colors_sub):
        mask = sub_labels == label
        if label == -1:
            axes[1].scatter(X_2d[mask, 0], X_2d[mask, 1], c='lightgray', 
                          marker='x', s=20, alpha=0.6, label='Noise')
        else:
            # Determine if this is a main cluster or sub-cluster
            if label < 100:
                axes[1].scatter(X_2d[mask, 0], X_2d[mask, 1], c=[color], 
                              s=30, alpha=0.7, label=f'Main {label}')
            else:
                main_cluster = label // 100
                sub_cluster = label % 100
                axes[1].scatter(X_2d[mask, 0], X_2d[mask, 1], c=[color], 
                              s=30, alpha=0.7, label=f'{main_cluster}.{sub_cluster}')
    
    axes[1].set_xlabel(f'{method_name} Component 1')
    axes[1].set_ylabel(f'{method_name} Component 2')
    # axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Too many labels
    
    # Cluster count comparison
    main_cluster_count = len(np.unique(main_labels[main_labels != -1]))
    sub_cluster_count = len(np.unique(sub_labels[sub_labels != -1]))
    
    axes[2].bar(['K-means\nClusters', 'Sub-clusters'], [main_cluster_count, sub_cluster_count], 
              color=['skyblue', 'lightcoral'])
    axes[2].set_ylabel('Number of Clusters')
    axes[2].set_title('Cluster Count Comparison')
    axes[2].grid(True, alpha=0.3)
    
    # Add count labels on bars
    axes[2].text(0, main_cluster_count + 0.1, str(main_cluster_count), 
               ha='center', va='bottom', fontweight='bold', fontsize=12)
    axes[2].text(1, sub_cluster_count + 0.1, str(sub_cluster_count), 
               ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    plt.tight_layout()
    plt.show()

# Plot sub-clustering results
plot_subclustering_results(dim_reductions['tsne']['data'], final_labels, sub_cluster_labels, 
                          'Sub-clustering Results', 't-SNE')

# %%
# =============================================================================
# CLUSTER ANALYSIS AND INTERPRETATION
# =============================================================================

def analyze_clusters(df, cluster_labels, feature_columns):
    """Analyze cluster characteristics and create customer profiles"""
    
    print("Analyzing cluster characteristics...")
    
    # Add cluster labels to dataframe
    df_analysis = df.copy()
    df_analysis['cluster'] = cluster_labels
    
    # Remove noise points for analysis
    df_clusters = df_analysis[df_analysis['cluster'] != -1].copy()
    
    if len(df_clusters) == 0:
        print("No valid clusters found for analysis")
        return None
    
    print(f"Analyzing {len(df_clusters)} customers across {len(df_clusters['cluster'].unique())} clusters")
    
    # Calculate cluster statistics for original (unscaled) features
    # We'll need to map back to original features
    original_features = [col.replace('_scaled', '') for col in feature_columns]
    available_original_features = [col for col in original_features if col in df.columns]
    
    cluster_stats = {}
    
    for cluster_id in sorted(df_clusters['cluster'].unique()):
        cluster_data = df_clusters[df_clusters['cluster'] == cluster_id]
        cluster_size = len(cluster_data)
        
        print(f"\n--- Cluster {cluster_id} Analysis ---")
        print(f"Size: {cluster_size:,} customers ({cluster_size/len(df_clusters)*100:.1f}%)")
        
        # Calculate means for key features
        feature_means = {}
        for feature in available_original_features:
            if feature in cluster_data.columns:
                mean_val = cluster_data[feature].mean()
                feature_means[feature] = mean_val
        
        # Store cluster profile
        cluster_stats[cluster_id] = {
            'size': cluster_size,
            'percentage': cluster_size/len(df_clusters)*100,
            'feature_means': feature_means,
            'outlier_score_mean': cluster_data['outlier_score'].mean(),
            'completeness_score_mean': cluster_data['feature_completeness_score'].mean()
        }
        
        # Print key characteristics
        key_metrics = ['return_rate', 'sales_order_no_nunique', 'customer_lifetime_days', 
                      'avg_order_size', 'return_product_variety']
        
        for metric in key_metrics:
            if metric in feature_means:
                print(f"  {metric}: {feature_means[metric]:.3f}")
    
    return cluster_stats, df_analysis

# Analyze final clusters
cluster_analysis, df_with_clusters = analyze_clusters(df_gold, sub_cluster_labels, clustering_features)

# Create cluster comparison heatmap
def create_cluster_heatmap(cluster_stats, top_features=15):
    """Create a heatmap comparing clusters across key features"""
    
    if not cluster_stats:
        print("No cluster statistics available for heatmap")
        return
    
    print(f"Creating cluster comparison heatmap...")
    
    # Prepare data for heatmap
    cluster_ids = sorted(cluster_stats.keys())
    
    # Get all features and their importance (based on variance across clusters)
    all_features = set()
    for stats in cluster_stats.values():
        all_features.update(stats['feature_means'].keys())
    
    all_features = list(all_features)
    
    # Calculate feature variance across clusters to select most discriminative features
    feature_variance = {}
    for feature in all_features:
        values = []
        for cluster_id in cluster_ids:
            if feature in cluster_stats[cluster_id]['feature_means']:
                values.append(cluster_stats[cluster_id]['feature_means'][feature])
            else:
                values.append(0)
        feature_variance[feature] = np.var(values)
    
    # Select top features by variance
    top_feature_names = sorted(feature_variance.keys(), key=lambda x: feature_variance[x], reverse=True)[:top_features]
    
    # Create matrix for heatmap
    heatmap_data = []
    for feature in top_feature_names:
        row = []
        for cluster_id in cluster_ids:
            if feature in cluster_stats[cluster_id]['feature_means']:
                row.append(cluster_stats[cluster_id]['feature_means'][feature])
            else:
                row.append(0)
        heatmap_data.append(row)
    
    heatmap_df = pd.DataFrame(heatmap_data, 
                             index=[f.replace('_', ' ').title() for f in top_feature_names],
                             columns=[f'Cluster {cid}' for cid in cluster_ids])
    
    # Create heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(heatmap_df, annot=True, fmt='.3f', cmap='RdYlBu_r', center=0, 
                cbar_kws={'label': 'Feature Value'})
    plt.title('Cluster Feature Comparison Heatmap')
    plt.xlabel('Clusters')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.show()
    
    return heatmap_df

# Create cluster heatmap
if cluster_analysis:
    cluster_heatmap = create_cluster_heatmap(cluster_analysis)

# %%
# =============================================================================
# SAVE CLUSTERING RESULTS
# =============================================================================

def save_clustering_results(df_with_clusters, cluster_stats, config, conn):
    """Save clustering results to database and files"""
    
    print("Saving clustering results...")
    
    # Add clustering results to customer data
    customer_clusters = df_with_clusters[['customer_emailid', 'cluster']].copy()
    customer_clusters.columns = ['customer_emailid', 'final_cluster_id']
    
    # Add cluster metadata
    customer_clusters['clustering_method'] = 'DBSCAN_KMEANS_SUBDBSCAN'
    customer_clusters['clustering_timestamp'] = pd.Timestamp.now()
    customer_clusters['cluster_quality_score'] = df_with_clusters['feature_completeness_score']
    
    # Create cluster summary table
    cluster_summary = []
    if cluster_stats:
        for cluster_id, stats in cluster_stats.items():
            cluster_summary.append({
                'cluster_id': cluster_id,
                'customer_count': stats['size'],
                'percentage_of_total': stats['percentage'],
                'avg_return_rate': stats['feature_means'].get('return_rate', 0),
                'avg_order_count': stats['feature_means'].get('sales_order_no_nunique', 0),
                'avg_lifetime_days': stats['feature_means'].get('customer_lifetime_days', 0),
                'avg_completeness_score': stats['completeness_score_mean']
            })
    
    cluster_summary_df = pd.DataFrame(cluster_summary)
    
    try:
        # Create clustering results table if it doesn't exist
        conn.execute("""
            CREATE TABLE IF NOT EXISTS clustering_results (
                customer_emailid VARCHAR,
                final_cluster_id INTEGER,
                clustering_method VARCHAR,
                clustering_timestamp TIMESTAMP,
                cluster_quality_score DOUBLE,
                PRIMARY KEY (customer_emailid, clustering_method)
            );
        """)
        
        # Create cluster summary table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS cluster_summary (
                cluster_id INTEGER,
                clustering_method VARCHAR,
                customer_count INTEGER,
                percentage_of_total DOUBLE,
                avg_return_rate DOUBLE,
                avg_order_count DOUBLE,
                avg_lifetime_days DOUBLE,
                avg_completeness_score DOUBLE,
                creation_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (cluster_id, clustering_method)
            );
        """)
        
        # Clear existing results for this method
        conn.execute("DELETE FROM clustering_results WHERE clustering_method = 'DBSCAN_KMEANS_SUBDBSCAN'")
        conn.execute("DELETE FROM cluster_summary WHERE clustering_method = 'DBSCAN_KMEANS_SUBDBSCAN'")
        
        # Insert new results
        conn.execute("INSERT INTO clustering_results SELECT * FROM customer_clusters")
        
        if not cluster_summary_df.empty:
            cluster_summary_df['clustering_method'] = 'DBSCAN_KMEANS_SUBDBSCAN'
            conn.execute("INSERT INTO cluster_summary SELECT * FROM cluster_summary_df")
        
        print(f"✅ Saved clustering results for {len(customer_clusters)} customers")
        print(f"✅ Saved summary for {len(cluster_summary_df)} clusters")
        
    except Exception as e:
        print(f"❌ Error saving to database: {str(e)}")
        print("Saving to CSV files as backup...")
        
        # Save to CSV as backup
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        customer_clusters.to_csv(f'clustering_results_{timestamp}.csv', index=False)
        cluster_summary_df.to_csv(f'cluster_summary_{timestamp}.csv', index=False)
        
        print(f"✅ Saved CSV backups with timestamp {timestamp}")
    
    # Save configuration and metadata
    clustering_metadata = {
        'clustering_config': config,
        'clustering_timestamp': pd.Timestamp.now().isoformat(),
        'total_customers': len(df_with_clusters),
        'customers_clustered': len(customer_clusters[customer_clusters['final_cluster_id'] != -1]),
        'noise_customers': len(customer_clusters[customer_clusters['final_cluster_id'] == -1]),
        'final_cluster_count': len(customer_clusters[customer_clusters['final_cluster_id'] != -1]['final_cluster_id'].unique()),
        'silhouette_score': kmeans_results.get('silhouette_score', 0),
        'calinski_harabasz_score': kmeans_results.get('calinski_harabasz_score', 0),
        'davies_bouldin_score': kmeans_results.get('davies_bouldin_score', 0)
    }
    
    # Save metadata to JSON
    with open('clustering_metadata.json', 'w') as f:
        json.dump(clustering_metadata, f, indent=2, default=str)
    
    print("✅ Saved clustering metadata to clustering_metadata.json")
    
    return customer_clusters, cluster_summary_df

# Save results
conn = get_connection("customer_clustering.db")
customer_results, cluster_summary = save_clustering_results(
    df_with_clusters, cluster_analysis, CLUSTERING_CONFIG, conn
)
conn.close()

# %%
# =============================================================================
# FINAL CLUSTERING REPORT
# =============================================================================

def generate_final_clustering_report(customer_results, cluster_summary, config):
    """Generate comprehensive clustering analysis report"""
    
    print("Generating final clustering report...")
    
    report = []
    report.append("="*80)
    report.append("CUSTOMER RETURN CLUSTERING - FINAL ANALYSIS REPORT")
    report.append("="*80)
    report.append(f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Clustering Method: DBSCAN → K-means → Sub-DBSCAN")
    report.append("")
    
    # Executive Summary
    total_customers = len(customer_results)
    clustered_customers = len(customer_results[customer_results['final_cluster_id'] != -1])
    noise_customers = total_customers - clustered_customers
    final_clusters = len(customer_results[customer_results['final_cluster_id'] != -1]['final_cluster_id'].unique())
    
    report.append("EXECUTIVE SUMMARY:")
    report.append("-" * 40)
    report.append(f"Total customers analyzed: {total_customers:,}")
    report.append(f"Customers successfully clustered: {clustered_customers:,} ({clustered_customers/total_customers*100:.1f}%)")
    report.append(f"Noise/outlier customers: {noise_customers:,} ({noise_customers/total_customers*100:.1f}%)")
    report.append(f"Final number of clusters: {final_clusters}")
    report.append("")
    
    # Clustering Quality Metrics
    report.append("CLUSTERING QUALITY METRICS:")
    report.append("-" * 40)
    if 'silhouette_score' in kmeans_results:
        report.append(f"Silhouette Score: {kmeans_results['silhouette_score']:.3f}")
        quality_rating = "Excellent" if kmeans_results['silhouette_score'] > 0.7 else \
                        "Good" if kmeans_results['silhouette_score'] > 0.5 else \
                        "Fair" if kmeans_results['silhouette_score'] > 0.3 else "Poor"
        report.append(f"Quality Rating: {quality_rating}")
    
    if 'calinski_harabasz_score' in kmeans_results:
        report.append(f"Calinski-Harabasz Score: {kmeans_results['calinski_harabasz_score']:.1f}")
    
    if 'davies_bouldin_score' in kmeans_results:
        report.append(f"Davies-Bouldin Score: {kmeans_results['davies_bouldin_score']:.3f}")
    
    report.append("")
    
    # Cluster Profiles
    if not cluster_summary.empty:
        report.append("CLUSTER PROFILES:")
        report.append("-" * 40)
        
        # Sort clusters by size
        cluster_summary_sorted = cluster_summary.sort_values('customer_count', ascending=False)
        
        for _, cluster in cluster_summary_sorted.iterrows():
            cluster_id = cluster['cluster_id']
            size = cluster['customer_count']
            pct = cluster['percentage_of_total']
            
            # Determine cluster archetype based on characteristics
            return_rate = cluster['avg_return_rate']
            order_count = cluster['avg_order_count']
            lifetime_days = cluster['avg_lifetime_days']
            
            if return_rate > 0.4:
                archetype = "High Returners"
            elif return_rate < 0.1 and order_count > 20:
                archetype = "Loyal Customers"
            elif lifetime_days > 730 and order_count > 15:
                archetype = "Veteran Shoppers"
            elif lifetime_days < 180:
                archetype = "New Customers"
            elif order_count > 30:
                archetype = "Frequent Buyers"
            else:
                archetype = "Regular Customers"
            
            report.append(f"Cluster {cluster_id}: {archetype}")
            report.append(f"  Size: {size:,} customers ({pct:.1f}%)")
            report.append(f"  Avg Return Rate: {return_rate:.3f}")
            report.append(f"  Avg Order Count: {order_count:.1f}")
            report.append(f"  Avg Lifetime (days): {lifetime_days:.0f}")
            report.append("")
    
    # Configuration Summary
    report.append("ALGORITHM CONFIGURATION:")
    report.append("-" * 40)
    report.append(f"Initial DBSCAN eps: {config['dbscan_initial']['eps']}")
    report.append(f"Initial DBSCAN min_samples: {config['dbscan_initial']['min_samples']}")
    report.append(f"K-means clusters: {config['kmeans']['optimal_k']}")
    report.append(f"Sub-DBSCAN eps: {config['dbscan_subclusters']['eps']}")
    report.append(f"Sub-DBSCAN min_samples: {config['dbscan_subclusters']['min_samples']}")
    report.append("")
    
    # Recommendations
    report.append("BUSINESS RECOMMENDATIONS:")
    report.append("-" * 40)
    
    if not cluster_summary.empty:
        # High return rate clusters
        high_return_clusters = cluster_summary[cluster_summary['avg_return_rate'] > 0.3]
        if not high_return_clusters.empty:
            total_high_returners = high_return_clusters['customer_count'].sum()
            report.append(f"• Focus on {total_high_returners:,} high-return customers in {len(high_return_clusters)} clusters")
            report.append("  - Implement return reason analysis")
            report.append("  - Consider product quality reviews")
            report.append("  - Offer sizing assistance or virtual try-on")
            report.append("")
        
        # Large valuable clusters
        large_clusters = cluster_summary[cluster_summary['customer_count'] > clustered_customers * 0.15]
        if not large_clusters.empty:
            report.append(f"• Develop targeted strategies for {len(large_clusters)} major customer segments")
            report.append("  - Create personalized marketing campaigns")
            report.append("  - Optimize product recommendations")
            report.append("  - Design cluster-specific retention programs")
            report.append("")
        
        # Small clusters
        small_clusters = cluster_summary[cluster_summary['customer_count'] < clustered_customers * 0.05]
        if not small_clusters.empty:
            report.append(f"• Monitor {len(small_clusters)} niche customer segments")
            report.append("  - Assess if specialized attention is warranted")
            report.append("  - Consider merging with similar larger clusters")
            report.append("")
    
    # Outlier recommendations
    if noise_customers > 0:
        report.append(f"• Investigate {noise_customers:,} outlier customers")
        report.append("  - Check for data quality issues")
        report.append("  - Identify potential fraud or unusual behavior")
        report.append("  - Consider manual review for high-value outliers")
        report.append("")
    
    report.append("NEXT STEPS:")
    report.append("-" * 40)
    report.append("1. Validate cluster profiles with business stakeholders")
    report.append("2. Implement cluster-based customer segmentation in CRM")
    report.append("3. Design A/B tests for cluster-specific interventions")
    report.append("4. Monitor cluster stability over time")
    report.append("5. Refine clustering based on business feedback")
    report.append("")
    report.append("="*80)
    
    # Print and save report
    report_text = "\n".join(report)
    print(report_text)
    
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    with open(f"clustering_final_report_{timestamp}.txt", 'w') as f:
        f.write(report_text)
    
    print(f"\n✅ Final clustering report saved to clustering_final_report_{timestamp}.txt")
    
    return report_text

# Generate final report
final_report = generate_final_clustering_report(customer_results, cluster_summary, CLUSTERING_CONFIG)

# %%
# =============================================================================
# CLUSTERING PIPELINE SUMMARY
# =============================================================================

print("\n" + "="*80)
print("CLUSTERING PIPELINE COMPLETED SUCCESSFULLY!")
print("="*80)

print(f"""
PIPELINE SUMMARY:
• Phase 1: Initial DBSCAN identified {dbscan_results['n_clusters']} dense regions and {(initial_labels == -1).sum():,} outliers
• Phase 2: K-means optimization selected K={optimal_k} clusters with silhouette score {kmeans_results.get('silhouette_score', 0):.3f}
• Phase 3: Final K-means clustering created {len(np.unique(final_labels[final_labels != -1]))} main clusters
• Phase 4: Sub-clustering refined clusters into {len(np.unique(sub_cluster_labels[sub_cluster_labels != -1]))} final segments

RESULTS SAVED:
• Database tables: clustering_results, cluster_summary
• Files: clustering_metadata.json, clustering_final_report_*.txt
• Visualizations: Generated in notebook cells above

CUSTOMER SEGMENTATION:
• {len(customer_results[customer_results['final_cluster_id'] != -1]):,} customers successfully clustered
• {len(customer_results[customer_results['final_cluster_id'] == -1]):,} customers flagged as outliers
• {len(cluster_summary) if not cluster_summary.empty else 0} distinct customer segments identified

Ready for business implementation and A/B testing!
""")

print("="*80)

# =============================================================================
# CONFIGURATION AND SETUP
# =============================================================================

# CLUSTERING PARAMETERS - Adjust these based on your analysis needs
CLUSTERING_CONFIG = {
    # DBSCAN Parameters for initial outlier detection/noise identification
    'dbscan_initial': {
        'eps': 0.5,              # Distance threshold - ADJUST based on your data scale
        'min_samples': 5,        # Minimum samples in neighborhood
        'algorithm': 'auto',     # Algorithm choice
        'n_jobs': -1            # Use all CPU cores
    },
    
    # K-means Parameters for main segmentation
    'kmeans': {
        'n_clusters_range': range(3, 15),  # Range of K to test
        'optimal_k': None,       # Will be determined automatically
        'random_state': 42,      # For reproducibility
        'n_init': 20,           # Number of random initializations
        'max_iter': 500,        # Maximum iterations
        'algorithm': 'lloyd'     # Algorithm choice
    },
    
    # Sub-clustering DBSCAN parameters for within-cluster refinement
    'dbscan_subclusters': {
        'eps': 0.3,             # Tighter epsilon for sub-clusters
        'min_samples': 3,       # Smaller minimum samples
        'algorithm': 'auto',
        'n_jobs': -1
    },
    
    # Dimensionality reduction for visualization
    'dimensionality_reduction': {
        'pca_components': 10,           # PCA components for preprocessing
        'tsne_components': 2,           # t-SNE dimensions
        'tsne_perplexity': 30,          # t-SNE perplexity
        'umap_n_neighbors': 15,         # UMAP neighbors
        'umap_min_dist': 0.1,          # UMAP minimum distance
        'random_state': 42
    },
    
    # Evaluation metrics thresholds
    'evaluation': {
        'min_silhouette_score': 0.3,   # Minimum acceptable silhouette score
        'max_davies_bouldin': 2.0,     # Maximum acceptable Davies-Bouldin index
        'min_cluster_size': 10,        # Minimum cluster size to be considered valid
        'max_noise_ratio': 0.15        # Maximum acceptable noise ratio for DBSCAN
    }
}

print("="*60)
print("CUSTOMER RETURN CLUSTERING ANALYSIS")
print("="*60)
print(f"Configuration loaded. K-means will test {len(CLUSTERING_CONFIG['kmeans']['n_clusters_range'])} different cluster counts.")
print(f"DBSCAN initial eps: {CLUSTERING_CONFIG['dbscan_initial']['eps']}")
print(f"Minimum silhouette score target: {CLUSTERING_CONFIG['evaluation']['min_silhouette_score']}")

# %%
# =============================================================================
# DATA LOADING AND PREPARATION
# =============================================================================

def load_clustering_data():
    """Load preprocessed data from gold layer"""
    
    print("Loading clustering data from gold layer...")
    
    conn = get_connection("customer_clustering.db")
    
    # Load the processed data
    query = """
    SELECT * FROM gold_cluster_processed 
    WHERE customer_emailid IS NOT NULL;
    """
    
    df = conn.execute(query).fetchdf()
    conn.close()
    
    print(f"Loaded {len(df)} customers from gold layer")
    print(f"Features available: {len(df.columns)} columns")
    
    # Separate scaled features for clustering
    feature_columns = [col for col in df.columns if col.endswith('_scaled')]
    metadata_columns = ['customer_emailid', 'outlier_score', 'feature_completeness_score', 
                       'data_quality_flags', 'processing_timestamp', 'scaling_method']
    
    print(f"Clustering features: {len(feature_columns)}")
    print(f"Metadata columns: {len(metadata_columns)}")
    
    return df, feature_columns, metadata_columns

# Load the data
df_gold, clustering_features, metadata_features = load_clustering_data()

# Basic data quality checks
print(f"\nData Quality Summary:")
print(f"Total customers: {len(df_gold):,}")
print(f"Customers with quality flags: {(df_gold['data_quality_flags'] != '').sum():,}")
print(f"Outliers detected: {df_gold['outlier_score'].lt(0).sum():,}")
print(f"Average feature completeness: {df_gold['feature_completeness_score'].mean():.3f}")

# Prepare clustering matrix
X_cluster = df_gold[clustering_features].values
print(f"\nClustering matrix shape: {X_cluster.shape}")

# %%
# =============================================================================
# EXPLORATORY DATA ANALYSIS FOR CLUSTERING
# =============================================================================

def analyze_data_distribution(df, features):
    """Analyze the distribution of clustering features"""
    
    print("Analyzing feature distributions for clustering...")
    
    # Create distribution plots for key features
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    # Select representative features for visualization
    key_features = [f for f in features if any(keyword in f for keyword in 
                   ['return_rate', 'order', 'lifetime', 'adjacency', 'seasonal', 'category'])][:6]
    
    for i, feature in enumerate(key_features):
        if i < len(axes):
            df[feature].hist(bins=50, ax=axes[i], alpha=0.7)
            axes[i].set_title(f'{feature.replace("_scaled", "")}')
            axes[i].set_xlabel('Scaled Value')
            axes[i].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.suptitle('Distribution of Key Clustering Features', y=1.02, fontsize=16)
    plt.show()
    
    # Correlation analysis
    print("\nFeature correlation analysis:")
    correlation_matrix = df[features].corr()
    
    # Find highly correlated feature pairs
    high_corr_pairs = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            corr_val = correlation_matrix.iloc[i, j]
            if abs(corr_val) > 0.8:  # High correlation threshold
                high_corr_pairs.append((correlation_matrix.columns[i], 
                                      correlation_matrix.columns[j], corr_val))
    
    if high_corr_pairs:
        print("Highly correlated feature pairs (|correlation| > 0.8):")
        for feat1, feat2, corr in high_corr_pairs:
            print(f"  {feat1} <-> {feat2}: {corr:.3f}")
    else:
        print("No highly correlated feature pairs found (good for clustering)")
    
    return correlation_matrix

# Analyze data distribution
correlation_matrix = analyze_data_distribution(df_gold, clustering_features)

# %%
# =============================================================================
# DIMENSIONALITY REDUCTION FOR VISUALIZATION
# =============================================================================

def create_dimensionality_reductions(X, config):
    """Create PCA, t-SNE, and UMAP reductions for visualization"""
    
    print("Creating dimensionality reductions for visualization...")
    
    # PCA for initial dimensionality reduction
    print(f"Running PCA (components: {config['pca_components']})...")
    pca = PCA(n_components=config['pca_components'], random_state=config['random_state'])
    X_pca = pca.fit_transform(X)
    
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio)
    
    print(f"PCA explained variance: {cumulative_variance[-1]:.3f} (first {config['pca_components']} components)")
    
    # t-SNE for non-linear visualization
    print(f"Running t-SNE (perplexity: {config['tsne_perplexity']})...")
    tsne = TSNE(n_components=config['tsne_components'], 
                perplexity=config['tsne_perplexity'],
                random_state=config['random_state'],
                n_jobs=-1)
    X_tsne = tsne.fit_transform(X_pca)  # Use PCA-reduced data for t-SNE
    
    # UMAP for another non-linear view
    print(f"Running UMAP (neighbors: {config['umap_n_neighbors']})...")
    umap_reducer = umap.UMAP(n_neighbors=config['umap_n_neighbors'],
                            min_dist=config['umap_min_dist'],
                            n_components=2,
                            random_state=config['random_state'])
    X_umap = umap_reducer.fit_transform(X_pca)
    
    return {
        'pca': {'data': X_pca, 'model': pca, 'explained_variance': explained_variance_ratio},
        'tsne': {'data': X_tsne, 'model': tsne},
        'umap': {'data': X_umap, 'model': umap_reducer}
    }

# Create dimensionality reductions
dim_reductions = create_dimensionality_reductions(X_cluster, CLUSTERING_CONFIG['dimensionality_reduction'])

# Plot PCA explained variance
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(range(1, len(dim_reductions['pca']['explained_variance']) + 1), 
         np.cumsum(dim_reductions['pca']['explained_variance']), 'bo-')
plt.xlabel('Principal Component')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.title('PCA Explained Variance')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.bar(range(1, len(dim_reductions['pca']['explained_variance']) + 1), 
        dim_reductions['pca']['explained_variance'])
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('Individual Component Variance')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %%
# =============================================================================
# PHASE 1: INITIAL DBSCAN FOR OUTLIER DETECTION
# =============================================================================

def run_initial_dbscan(X, config):
    """Run initial DBSCAN to identify outliers and core patterns"""
    
    print("Phase 1: Running initial DBSCAN for outlier detection...")
    print(f"Parameters: eps={config['eps']}, min_samples={config['min_samples']}")
    
    dbscan_initial = DBSCAN(
        eps=config['eps'],
        min_samples=config['min_samples'],
        algorithm=config['algorithm'],
        n_jobs=config['n_jobs']
    )
    
    initial_labels = dbscan_initial.fit_predict(X)
    
    # Analyze results
    n_clusters = len(set(initial_labels)) - (1 if -1 in initial_labels else 0)
    n_noise = list(initial_labels).count(-1)
    noise_ratio = n_noise / len(initial_labels)
    
    print(f"Initial DBSCAN Results:")
    print(f"  Clusters found: {n_clusters}")
    print(f"  Noise points: {n_noise:,} ({noise_ratio:.1%})")
    print(f"  Core samples: {len(dbscan_initial.core_sample_indices_):,}")
    
    # Check if noise ratio is acceptable
    max_noise_ratio = CLUSTERING_CONFIG['evaluation']['max_noise_ratio']
    if noise_ratio > max_noise_ratio:
        print(f"⚠️  WARNING: Noise ratio ({noise_ratio:.1%}) exceeds threshold ({max_noise_ratio:.1%})")
        print("   Consider adjusting eps or min_samples parameters")
    else:
        print(f"✅ Noise ratio within acceptable range")
    
    return initial_labels, dbscan_initial, {'n_clusters': n_clusters, 'noise_ratio': noise_ratio}

# Run initial DBSCAN
initial_labels, dbscan_model, dbscan_results = run_initial_dbscan(X_cluster, CLUSTERING_CONFIG['dbscan_initial'])

# Visualize initial DBSCAN results
def plot_dbscan_results(X_2d, labels, title, method_name):
    """Plot DBSCAN results in 2D space"""
    
    plt.figure(figsize=(12, 5))
    
    # Plot with cluster colors
    plt.subplot(1, 2, 1)
    unique_labels = set(labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black for noise points
            col = 'black'
            plt.scatter(X_2d[labels == k, 0], X_2d[labels == k, 1], 
                       c=[col], marker='x', s=20, alpha=0.6, label='Noise')
        else:
            plt.scatter(X_2d[labels == k, 0], X_2d[labels == k, 1], 
                       c=[col], marker='o', s=30, alpha=0.7, label=f'Cluster {k}')
    
    plt.title(f'{title} - {method_name}')
    plt.xlabel(f'{method_name} Component 1')
    plt.ylabel(f'{method_name} Component 2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Plot density
    plt.subplot(1, 2, 2)
    non_noise_mask = labels != -1
    if non_noise_mask.sum() > 0:
        plt.scatter(X_2d[non_noise_mask, 0], X_2d[non_noise_mask, 1], 
                   c=labels[non_noise_mask], cmap='viridis', s=30, alpha=0.7)
        plt.colorbar(label='Cluster')
    
    noise_mask = labels == -1
    if noise_mask.sum() > 0:
        plt.scatter(X_2d[noise_mask, 0], X_2d[noise_mask, 1], 
                   c='red', marker='x', s=20, alpha=0.6, label='Noise')
        plt.legend()
    
    plt.title(f'{title} - Density View')
    plt.xlabel(f'{method_name} Component 1')
    plt.ylabel(f'{method_name} Component 2')
    
    plt.tight_layout()
    plt.show()

# Plot initial DBSCAN results
plot_dbscan_results(dim_reductions['tsne']['data'], initial_labels, 'Initial DBSCAN Results', 't-SNE')
plot_dbscan_results(dim_reductions['umap']['data'], initial_labels, 'Initial DBSCAN Results', 'UMAP')

# %%
# =============================================================================
# PHASE 2: K-MEANS OPTIMAL CLUSTER DETERMINATION
# =============================================================================

def find_optimal_kmeans_clusters(X, k_range, exclude_noise_mask=None):
    """Find optimal number of clusters using multiple metrics"""
    
    print("Phase 2: Finding optimal K-means cluster count...")
    
    # Exclude noise points if mask provided
    if exclude_noise_mask is not None:
        X_clean = X[~exclude_noise_mask]
        print(f"Using {len(X_clean):,} non-noise points for K-means optimization")
    else:
        X_clean = X
        print(f"Using all {len(X_clean):,} points for K-means optimization")
    
    # Storage for metrics
    metrics = {
        'k_values': list(k_range),
        'inertias': [],
        'silhouette_scores': [],
        'calinski_harabasz_scores': [],
        'davies_bouldin_scores': []
    }
    
    print("Testing different numbers of clusters...")
    for k in k_range:
        print(f"  Testing K={k}...", end='')
        
        # Fit K-means
        kmeans = KMeans(
            n_clusters=k,
            random_state=CLUSTERING_CONFIG['kmeans']['random_state'],
            n_init=CLUSTERING_CONFIG['kmeans']['n_init'],
            max_iter=CLUSTERING_CONFIG['kmeans']['max_iter'],
            algorithm=CLUSTERING_CONFIG['kmeans']['algorithm']
        )
        
        cluster_labels = kmeans.fit_predict(X_clean)
        
        # Calculate metrics
        metrics['inertias'].append(kmeans.inertia_)
        
        if k > 1:  # Silhouette score requires at least 2 clusters
            sil_score = silhouette_score(X_clean, cluster_labels)
            metrics['silhouette_scores'].append(sil_score)
            
            ch_score = calinski_harabasz_score(X_clean, cluster_labels)
            metrics['calinski_harabasz_scores'].append(ch_score)
            
            db_score = davies_bouldin_score(X_clean, cluster_labels)
            metrics['davies_bouldin_scores'].append(db_score)
        else:
            metrics['silhouette_scores'].append(0)
            metrics['calinski_harabasz_scores'].append(0)
            metrics['davies_bouldin_scores'].append(float('inf'))
        
        print(f" Silhouette: {metrics['silhouette_scores'][-1]:.3f}")
    
    # Determine optimal K using multiple criteria
    optimal_k_candidates = {}
    
    # Elbow method (looking for the "elbow" in inertia)
    inertia_diffs = np.diff(metrics['inertias'])
    inertia_diffs2 = np.diff(inertia_diffs)
    if len(inertia_diffs2) > 0:
        elbow_idx = np.argmax(inertia_diffs2) + 2  # +2 because of double diff
        optimal_k_candidates['elbow'] = k_range[elbow_idx]
    
    # Best silhouette score
    best_sil_idx = np.argmax(metrics['silhouette_scores'])
    optimal_k_candidates['silhouette'] = k_range[best_sil_idx]
    
    # Best Calinski-Harabasz score (higher is better)
    best_ch_idx = np.argmax(metrics['calinski_harabasz_scores'])
    optimal_k_candidates['calinski_harabasz'] = k_range[best_ch_idx]
    
    # Best Davies-Bouldin score (lower is better)
    best_db_idx = np.argmin(metrics['davies_bouldin_scores'])
    optimal_k_candidates['davies_bouldin'] = k_range[best_db_idx]
    
    print(f"\nOptimal K candidates:")
    for method, k_opt in optimal_k_candidates.items():
        print(f"  {method:15}: K={k_opt}")
    
    # Choose final K (prioritize silhouette score with validation)
    final_k = optimal_k_candidates['silhouette']
    final_sil_score = metrics['silhouette_scores'][k_range.index(final_k)]
    
    min_sil_threshold = CLUSTERING_CONFIG['evaluation']['min_silhouette_score']
    if final_sil_score < min_sil_threshold:
        print(f"⚠️  WARNING: Best silhouette score ({final_sil_score:.3f}) below threshold ({min_sil_threshold})")
        
        # Try alternative: use Calinski-Harabasz if its silhouette is acceptable
        alt_k = optimal_k_candidates['calinski_harabasz']
        alt_sil_score = metrics['silhouette_scores'][k_range.index(alt_k)]
        
        if alt_sil_score >= min_sil_threshold:
            final_k = alt_k
            print(f"   Using Calinski-Harabasz optimum: K={final_k} (silhouette: {alt_sil_score:.3f})")
        else:
            print(f"   Proceeding with best available: K={final_k}")
    
    print(f"\n✅ Selected optimal K: {final_k}")
    
    return final_k, metrics, optimal_k_candidates

# Create noise mask from initial DBSCAN
noise_mask = initial_labels == -1

# Find optimal K
optimal_k, k_metrics, k_candidates = find_optimal_kmeans_clusters(
    X_cluster, 
    CLUSTERING_CONFIG['kmeans']['n_clusters_range'],
    exclude_noise_mask=noise_mask
)

# Update configuration with optimal K
CLUSTERING_CONFIG['kmeans']['optimal_k'] = optimal_k

# Plot clustering optimization metrics
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Elbow curve
axes[0, 0].plot(k_metrics['k_values'], k_metrics['inertias'], 'bo-')
axes[0, 0].set_xlabel('Number of Clusters (K)')
axes[0, 0].set_ylabel('Inertia')
axes[0, 0].set_title('Elbow Method')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].axvline(x=optimal_k, color='red', linestyle='--', alpha=0.7, label=f'Selected K={optimal_k}')
axes[0, 0].legend()

# Silhouette scores
axes[0, 1].plot(k_metrics['k_values'], k_metrics['silhouette_scores'], 'go-')
axes[0, 1].set_xlabel('Number of Clusters (K)')
axes[0, 1].set_ylabel('Silhouette Score')
axes[0, 1].set_title('Silhouette Analysis')
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].axhline(y=CLUSTERING_CONFIG['evaluation']['min_silhouette_score'], 
                   color='orange', linestyle='--', alpha=0.7, label='Min Threshold')
axes[0, 1].axvline(x=optimal_k, color='red', linestyle='--', alpha=0.7, label=f'Selected K={optimal_k}')
axes[0, 1].legend()

# Calinski-Harabasz scores
axes[1, 0].plot(k_metrics['k_values'], k_metrics['calinski_harabasz_scores'], 'mo-')
axes[1, 0].set_xlabel('Number of Clusters (K)')
axes[1, 0].set_ylabel('Calinski-Harabasz Score')
axes[1, 0].set_title('Calinski-Harabasz Index')
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].axvline(x=optimal_k, color='red', linestyle='--', alpha=0.7, label=f'Selected K={optimal_k}')
axes[1, 0].legend()

# Davies-Bouldin scores
axes[1, 1].plot(k_metrics['k_values'], k_metrics['davies_bouldin_scores'], 'ro-')
axes[1, 1].set_xlabel('Number of Clusters (K)')
axes[1, 1].set_ylabel('Davies-Bouldin Score')
axes[1, 1].set_title('Davies-Bouldin Index')
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].axhline(y=CLUSTERING_CONFIG['evaluation']['max_davies_bouldin'], 
                   color='orange', linestyle='--', alpha=0.7, label='Max Threshold')
axes[1, 1].axvline(x=optimal_k, color='red', linestyle='--', alpha=0.7, label=f'Selected K={optimal_k}')
axes[1, 1].legend()

plt.tight_layout()
plt.suptitle('K-means Cluster Optimization Metrics', y=1.02, fontsize=16)
plt.show()

# %%
# =============================================================================
# PHASE 3: FINAL K-MEANS CLUSTERING
# =============================================================================

def run_final_kmeans(X, k, exclude_noise_mask=None):
    """Run final K-means clustering with optimal parameters"""
    
    print(f"Phase 3: Running final K-means clustering with K={k}...")
    
    # Prepare data
    if exclude_noise_mask is not None:
        X_clean = X[~exclude_noise_mask]
        clean_indices = np.where(~exclude_noise_mask)[0]
        print(f"Using {len(X_clean):,} non-noise points for final clustering")
    else:
        X_clean = X
        clean_indices = np.arange(len(X))
        print(f"Using all {len(X_clean):,} points for final clustering")
    
    # Fit final K-means
    final_kmeans = KMeans(
        n_clusters=k,
        random_state=CLUSTERING_CONFIG['kmeans']['random_state'],
        n_init=CLUSTERING_CONFIG['kmeans']['n_init'],
        max_iter=CLUSTERING_CONFIG['kmeans']['max_iter'],
        algorithm=CLUSTERING_CONFIG['kmeans']['algorithm']
    )
    
    clean_labels = final_kmeans.fit_predict(X_clean)
    
    # Create full label array (including noise points)
    full_labels = np.full(len(X), -1)  # Initialize with noise label
    full_labels[clean_indices] = clean_labels
    
    # Calculate final metrics
    final_silhouette = silhouette_score(X_clean, clean_labels)
    final_ch_score = calinski_harabasz_score(X_clean, clean_labels)
    final_db_score = davies_bouldin_score(X_clean, clean_labels)
    
    print(f"Final K-means Results:")
    print(f"  Clusters: {k}")
    print(f"  Silhouette Score: {final_silhouette:.3f}")
    print(f"  Calinski-Harabasz Score: {final_ch_score:.1f}")
    print(f"  Davies-Bouldin Score: {final_db_score:.3f}")
    
    # Analyze cluster sizes
    unique_labels, counts = np.unique(clean_labels, return_counts=True)
    print(f"  Cluster sizes:")
    for label, count in zip(unique_labels, counts):
        percentage = count / len(clean_labels) * 100
        print(f"    Cluster {label}: {count:,} customers ({percentage:.1f}%)")
    
    # Check for small clusters
    min_cluster_size = CLUSTERING_CONFIG['evaluation']['min_cluster_size']
    small_clusters = counts < min_cluster_size
    if small_clusters.any():
        print(f"⚠️  WARNING: {small_clusters.sum()} clusters below minimum size ({min_cluster_size})")
    
    return full_labels, final_kmeans, {
        'silhouette_score': final_silhouette,
        'calinski_harabasz_score': final_ch_score,
        'davies_bouldin_score': final_db_score,
        'cluster_sizes': dict(zip(unique_labels, counts))
    }

# Run final K-means
final_labels, kmeans_model, kmeans_results = run_final_kmeans(
    X_cluster, 
    optimal_k, 
    exclude_noise_mask=noise_mask
)

# Visualize final K-means results
def plot_kmeans_results(X_2d, labels, title, method_name):
    """Plot K-means results in 2D space"""
    
    plt.figure(figsize=(15, 5))
    
    # Plot clusters
    plt.subplot(1, 3, 1)
    unique_labels = set(labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    
    for k, col in zip(unique_labels, colors):
        if k == -1:
            plt.scatter(X_2d[labels == k, 0], X_2d[labels == k, 1], 
                       c='lightgray', marker='x', s=20, alpha=0.6, label='Noise/Outliers')
        else:
            plt.scatter(X_2d[labels == k, 0], X_2d[labels == k, 1], 
                       c=[col], marker='o', s=30, alpha=0.7, label=f'Cluster {k}')
    
    plt.title(f'{title} - {method_name}')
    plt.xlabel(f'{method_name} Component 1')
    plt.ylabel(f'{method_name} Component 2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Plot with centroids (if not noise)
    plt.subplot(1, 3, 2)
    non_noise_mask = labels != -1
    if non_noise_mask.sum() > 0:
        scatter = plt.scatter(X_2d[non_noise_mask, 0], X_2d[non_noise_mask, 1], 
                            c=labels[non_noise_mask], cmap='tab10', s=30, alpha=0.7)
        plt.colorbar(scatter, label='Cluster')
        
        # Add cluster centroids in 2D space
        for cluster_id in set(labels[non_noise_mask]):
            cluster_mask = labels == cluster_id
            centroid_x = np.mean(X_2d[cluster_mask, 0])
            centroid_y = np.mean(X_2d[cluster_mask, 1])
            plt.scatter(centroid_x, centroid_y, c='red', marker='X', s=200, 
                       edgecolors='black', linewidth=2, label='Centroids' if cluster_id == 0 else '')
    
    noise_mask = labels == -1
    if noise_mask.sum() > 0:
        plt.scatter(X_2d[noise_mask, 0], X_2d[noise_mask, 1], 
                   c='lightgray', marker='x', s=20, alpha=0.6)
    
    plt.title(f'{title} - With Centroids')
    plt.xlabel(f'{method_name} Component 1')
    plt.ylabel(f'{method_name} Component 2')
    if non_noise_mask.sum() > 0:
        plt.legend()
    
    # Cluster size distribution
    plt.subplot(1, 3, 3)
    cluster_ids, cluster_sizes = np.unique(labels[labels != -1], return_counts=True)
    plt.bar(cluster_ids, cluster_sizes, color=plt.cm.tab10(np.linspace(0, 1, len(cluster_ids))))
    plt.xlabel('Cluster ID')
    plt.ylabel('Number of Customers')
    plt.title('Cluster Size Distribution')
    plt.grid(True, alpha=0.3)
    
    # Add size labels on bars
    for i, (cluster_id, size) in enumerate(zip(cluster_ids, cluster_sizes)):
        plt.text(cluster_id, size + max(cluster_sizes)*0.01, str(size), 
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.show()

# Plot final K-means results
plot_kmeans_results(dim_reductions['tsne']['data'], final_labels, 'Final K-means Results', 't-SNE')
plot_kmeans_results(dim_reductions['umap']['data'], final_labels, 'Final K-means Results', 'UMAP')

# %%