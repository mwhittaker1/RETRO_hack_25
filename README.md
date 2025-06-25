# Customer Clustering Project

## Project Overview
This project implements customer segmentation using a hybrid approach combining DBSCAN and K-means clustering algorithms. The clustering is based on customer return behaviors and other purchasing metrics.

## Data Layers

### Bronze Layer
Raw customer data including transaction history and return information.

### Silver Layer
Processed and engineered features used for customer segmentation, with imputation applied according to the strategy.
- Exported to `silver_customer_features_20250625_133058.xlsx` (14,999 rows, 44 columns)

### Gold Layer
Final customer clusters with business insights and recommendations.
- Exported to `gold_cluster_processed_20250625_134212.xlsx` (14,999 rows, 41 columns)

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
- `gold_cluster_processed_20250625_134212.xlsx`: Exported gold layer with processed features
- `silver_customer_features_20250625_133058.xlsx`: Exported silver layer with all engineered features
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
   - `gold_cluster_processed_*.xlsx`: Complete gold layer with processed features
   - `silver_customer_features_*.xlsx`: Complete silver layer with all engineered features
   - `customer_clusters_with_features_*.csv`: Final output with cluster assignments
   - `clustering_final_report_*.txt`: Comprehensive report with recommendations

## Imputation Strategy

The project implements a business-appropriate imputation strategy:

1. **Feature Filtering:**
   - Features with 100% null values are automatically excluded from clustering
   - Highly correlated features (|correlation| > 0.8) are reduced to avoid multicollinearity

2. **Core Business Metrics:**
   - Records with missing core metrics are dropped in the silver layer
   - Ensures high data quality for critical business metrics

3. **Return Features:**
   - Zero-filled when missing (business logic: no returns = 0 returns)
   - Preserves semantic meaning of the data

4. **Other Features:**
   - Mean imputation for remaining features
   - Maintains statistical distribution properties

5. **Imputation Tracking:**
   - Pipeline tracks imputation throughout the clustering process
   - Identifies clusters potentially influenced by imputed values
   - Exported data includes imputation flags for downstream analysis

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
3. (Optional) Run `export_silver_layer.py` to export the silver layer
4. (Optional) Run `export_gold_layer.py` to export the gold layer
5. Review the output in the results directory
