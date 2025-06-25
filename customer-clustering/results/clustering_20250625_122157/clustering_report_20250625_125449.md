# Customer Clustering Analysis Report
Generated on: 2025-06-25 12:54:49

## Summary
- Total samples: 14999
- Number of clusters: 3
- Noise points: 14733 (98.23%)
- Dimensionality reduction technique: TSNE

## Configuration
### Dimensionality Reduction
- pca_components: 10
- tsne_components: 2
- tsne_perplexity: 30
- umap_components: 2
- umap_neighbors: 15
- umap_min_dist: 0.1
- random_state: 42

### DBSCAN
- eps: 0.7
- min_samples: 10
- metric: euclidean
- algorithm: auto
- n_jobs: -1

### K-means
- n_clusters: auto
- random_state: 42
- n_init: 10
- max_clusters: 8
- exclude_dbscan_noise: True
- strict_clusters: True
- n_clusters_range: range(2, 9)
- optimal_k: 3
- max_iter: 300
- algorithm: auto

### Sub-DBSCAN
- eps: 0.6
- min_samples: 15
- metric: euclidean
- algorithm: auto
- n_jobs: -1
- max_sub_clusters: 2

## Cluster Profiles
| Cluster ID | Size | Percentage | sales_order_no_nunique_scaled | sku_nunique_scaled | items_returned_count_scaled | sales_qty_scaled | avg_order_size_scaled |
|---|---|---|---|---|---|---|---|---|
| 0 | 231 | 1.54% | 4.45 | 3.74 | 4.68 | 13.28 | 0.42 |
| 2 | 25 | 0.17% | 15.85 | 13.54 | 32.99 | 0.60 | 0.95 |
| 1 | 10 | 0.07% | 18.63 | 24.11 | 46.94 | 1.18 | 2.16 |

## Cluster Distribution
Distribution of data points across clusters:

![Cluster Distribution](results\clustering_20250625_122157\cluster_distribution_20250625_125449.png)

## Key Findings
- Largest cluster: Cluster 0 with 231 points (1.54% of data)
- Smallest cluster: Cluster 1 with 10 points (0.07% of data)
- 14733 points (98.23% of data) were classified as noise
- PCA requires 2 components to explain 80% of variance

## Next Steps
Recommended next steps for using these clustering results:

1. **Validation**: Validate the clusters with domain experts
2. **Profiling**: Develop detailed profiles for each cluster
3. **Application**: Use the clusters for targeted marketing, product recommendations, etc.
4. **Monitoring**: Set up a process to periodically re-run clustering and track changes
5. **Integration**: Integrate cluster assignments into customer data systems
