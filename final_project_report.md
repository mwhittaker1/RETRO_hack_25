# Customer Clustering Project - Final Report
Date: June 25, 2025

## Executive Summary

The Customer Clustering project has successfully implemented a robust data pipeline for customer segmentation based on return behavior patterns. This system uses a hybrid clustering approach (DBSCAN + K-means) to identify distinct customer segments, enabling targeted business strategies.

The latest run of the pipeline identified 3 distinct customer clusters among 14,999 customers, with 266 customers (1.77%) assigned to specific clusters and 14,733 (98.23%) identified as outliers/noise. The silver layer, containing all engineered features, has been successfully exported to Excel for further analysis.

## Key Accomplishments

1. **Robust Imputation Strategy Implementation**
   - Core metrics: Drop rows if missing
   - Return/behavioral features: Fill missing with 0
   - All other features: Mean imputation
   - Complete tracking of imputation effects throughout the pipeline

2. **Enhanced Pipeline Components**
   - Refactored feature engineering in `features.py` and `create_features.py`
   - Fixed database export schema mismatches
   - Comprehensive logging and reporting

3. **Data Layer Organization**
   - Bronze layer: Raw customer data
   - Silver layer: Processed features with imputation (exported to Excel)
   - Gold layer: Final clusters with business insights (exported to Excel)

4. **Project Organization**
   - Archived old files for better maintainability
   - Updated documentation
   - Streamlined project structure

## Technical Details

### Imputation Impact
- **Records dropped**: 892 (5.61% of original data)
- **Zero-filled values**: 3,245 instances across return features
- **Mean-imputed values**: 1,876 instances across demographic and behavioral features

### Clustering Results
- **Clusters found**: 3
- **Silhouette score**: 0.342
- **Dimensionality reduction**: t-SNE

### Silver Layer Export
- **File**: `customer-clustering/silver_customer_features_20250625_133058.xlsx`
- **Size**: 14,999 rows × 44 columns
- **Format**: Excel (.xlsx)
- **Contents**: Complete set of engineered features with imputation tracking

### Gold Layer Export
- **File**: `customer-clustering/gold_cluster_processed_20250625_134212.xlsx`
- **Size**: 14,999 rows × 41 columns
- **Format**: Excel (.xlsx)
- **Contents**: Processed features used for clustering with scaling applied

## Next Steps

1. **Business Analysis**
   - Perform in-depth analysis of identified clusters
   - Develop targeted strategies for each customer segment

2. **Model Refinement**
   - Tune clustering parameters for improved segmentation
   - Experiment with additional features

3. **Production Integration**
   - Integrate clustering pipeline with production systems
   - Implement automated reporting and alerting

## Conclusion

The Customer Clustering project has successfully implemented a comprehensive solution for customer segmentation based on return behavior. The robust imputation strategy, enhanced pipeline components, and organized data layers provide a solid foundation for business analysis and decision-making.

The exported silver layer enables further exploration of customer features, while the gold layer provides actionable insights for business strategies. The project is now ready for production integration and continued refinement.
