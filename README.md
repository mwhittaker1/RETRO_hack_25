# Customer Clustering Project

## Project Overview
This project implements customer segmentation using a hybrid approach combining DBSCAN and K-means clustering algorithms. The clustering is based on customer return behaviors and other purchasing metrics.

## Directory Structure

### Root Directory
- `customer_clustering.db`: Primary DuckDB database containing clustering results
- `customer_features.db`: Database containing customer feature data
- `features.md`: Documentation of features used in the analysis
- `requirements.txt`: Python package dependencies
- `feature_details_report.py`: Script for generating feature details reports
- `generate_feature_report.py`: Script for generating feature analysis reports

### customer-clustering/ Directory
- `create_clusters.ipynb`: Main notebook for cluster creation and analysis
- `create_features.py`: Script for feature engineering
- `features.py`: Core feature creation logic
- `db.py`: Database connection and utilities
- `clustering_notebook.py`: Utility functions for the clustering notebook
- `cluster_preprocessing.py`: Data preprocessing for clustering
- `update_schema.py`: Database schema update utilities
- `update_gross_values.py`: Utilities for updating gross values
- `handler.ipynb`: Interactive notebook for data handling
- `data_analysis.py`: Data analysis utilities
- `generate_excel_report.py`: Excel report generation
- `decisions_doc.md`: Documentation of key decisions
- `readme_file.md`: Original project README

### Latest Results (customer-clustering/)
- `customer_clusters_with_features_20250625_125824.csv`: Final clusters with features
- `cluster_summary_with_imputation_20250625_125824.csv`: Cluster summary with imputation stats
- `clustering_final_report_20250625_125812.txt`: Final clustering report
- `clustering_results_20250625_125449.csv`: Raw clustering results
- `cluster_summary_20250625_125449.csv`: Summary statistics for each cluster

### results/ Directory
Contains detailed clustering results organized by timestamp, with the most recent being:
- `results/clustering_20250625_122157/`: Latest clustering results with reports, visualizations, and data exports

### archived_files/ Directory
Contains older versions of files from previous runs, including:
- Previous clustering results
- Log files
- Intermediate reports
- Backup database file
- Utility scripts used for debugging

## Key Files for the Clustering Pipeline

1. **Feature Engineering**:
   - `create_features.py`: Creates and processes features for clustering

2. **Clustering**:
   - `create_clusters.ipynb`: Main clustering notebook with hybrid approach

3. **Outputs**:
   - `customer_clusters_with_features_*.csv`: Final output with cluster assignments
   - `clustering_final_report_*.txt`: Comprehensive report with recommendations

## Imputation Strategy

The project implements a business-appropriate imputation strategy:
- Core business metrics: Drop records with missing values
- Return features: Zero-fill missing values (business logic: no returns = 0 returns)
- Other features: Mean imputation

## Latest Run Statistics

- Timestamp: June 25, 2025, 12:58 PM
- Clusters Found: 3
- Clustered Customers: 266 (1.77% of data)
- Outliers/Noise: 14,733 (98.23% of data)
- Dimensionality Reduction: t-SNE
- Silhouette Score: 0.342

## Running the Pipeline

1. Run `create_features.py` to generate features for clustering
2. Open and run `create_clusters.ipynb` to perform clustering
3. Review the output in the results directory
