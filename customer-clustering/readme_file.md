# Customer Return Clustering Pipeline

## Overview

A comprehensive machine learning pipeline for customer segmentation based on return behavior patterns. This system analyzes customer purchase and return data to identify distinct customer segments, enabling targeted business strategies and improved customer experience.

## Features

### 🎯 **Customer Segmentation**
- **33 engineered features** across 8 categories
- **Hybrid clustering approach**: DBSCAN → K-means → sub-DBSCAN
- **Advanced return behavior analysis** with temporal patterns
- **Outlier detection** for anomalous customer identification

### 🏗️ **Production-Ready Architecture**
- **Bronze/Silver/Gold data layers** for scalable data processing
- **Comprehensive data quality validation** and automated fixes
- **Configurable parameters** for different business contexts
- **Detailed logging and error handling** throughout pipeline

### 📊 **Business Intelligence**
- **Customer archetypes** with clear business interpretations
- **Actionable insights** for retention and satisfaction strategies
- **Performance metrics** and cluster quality validation
- **Excel reporting** for stakeholder communication

## Quick Start

### Prerequisites
```bash
# Required Python packages
pip install pandas numpy scikit-learn duckdb umap-learn seaborn matplotlib openpyxl
```

### 1. Setup Your Data
```python
# Update the CSV file path in handler.ipynb
PIPELINE_CONFIG = {
    'data_source': {
        'csv_file': 'your_data.csv',  # Update this path
        # ...
    }
}
```

### 2. Run the Complete Pipeline
```bash
# Execute the full pipeline
jupyter notebook handler.ipynb
# OR run individual components:
python create_features.py
python cluster_preprocessing.py
```

### 3. Perform Clustering Analysis
```bash
# Interactive clustering analysis
jupyter notebook create_clusters.ipynb
```

### 4. Generate Business Reports
```bash
# Creates clustering_results.xlsx with all findings
python generate_excel_report.py
```

## Pipeline Architecture

```
📁 Raw Data (CSV)
    ↓
🥉 Bronze Layer (bronze_return_order_data)
    ↓ [Data Quality & Validation]
🥈 Silver Layer (silver_customer_features)  
    ↓ [Feature Engineering & Testing]
🥇 Gold Layer (gold_cluster_processed)
    ↓ [Scaling & Preprocessing]
🎯 Clustering Results (clustering_results, cluster_summary)
```

## Feature Categories

### 📊 **Volume Metrics** (5 features)
- `sales_order_no_nunique` - Number of unique orders
- `sku_nunique` - Product variety purchased
- `items_returned_count` - Total returns
- `sales_qty_mean` - Average purchase quantity
- `avg_order_size` - Items per order

### 🔄 **Return Behavior** (6 features)
- `return_rate` - Proportion of items returned
- `return_ratio` - Return quantity vs purchase quantity
- `return_product_variety` - Breadth of return behavior
- `avg_returns_per_order` - Return frequency per order
- `return_frequency_ratio` - Return transaction frequency
- `return_intensity` - Return magnitude per returned item

### ⏰ **Temporal Patterns** (3 features)
- `customer_lifetime_days` - Customer tenure
- `avg_days_to_return` - Return timing patterns
- `return_timing_spread` - Return timing variability

### 📈 **Recency Analysis** (4 features)
- `recent_orders` - Recent purchase activity (90 days)
- `recent_returns` - Recent return activity (90 days)
- `recent_vs_avg_ratio` - Trend analysis
- `behavior_stability_score` - Pattern consistency

### 🏷️ **Category Intelligence** (3 features)
- `category_diversity_score` - Purchase breadth across categories
- `category_loyalty_score` - Concentration in specific categories
- `high_return_category_affinity` - Category-specific return patterns

### 🔗 **Adjacency Patterns** (4 features)
- `sku_adjacency_orders` - Related product purchases (±14 days)
- `sku_adjacency_returns` - Related product returns
- `sku_adjacency_timing` - Timing of adjacent purchases
- `sku_adjacency_return_timing` - Timing of adjacent returns

### 🌊 **Seasonal Trends** (2 features)
- `seasonal_susceptibility_orders` - Seasonal purchase sensitivity
- `seasonal_susceptibility_returns` - Seasonal return sensitivity

### 📊 **Trend Analysis** (2 features)
- `trend_product_category_order_rate` - Following category trends
- `trend_product_category_return_rate` - Category return trend correlation

## File Structure

```
📦 customer-clustering/
├── 📄 README.md                    # This file
├── 📄 decisions.md                 # Implementation decisions & assumptions
├── 🐍 db.py                       # Database setup & data loading
├── 🐍 features.py                 # Feature engineering functions
├── 🐍 create_features.py          # Feature pipeline runner
├── 🐍 cluster_preprocessing.py    # Gold layer preprocessing
├── 🐍 generate_excel_report.py    # Excel report generation
├── 📓 da.ipynb                    # Data analysis notebook
├── 📓 create_clusters.ipynb       # Clustering analysis notebook
├── 📓 handler.ipynb               # Pipeline orchestration
├── 📊 clustering_results.xlsx     # Business report (generated)
├── 🗃️ customer_clustering.db      # DuckDB database (generated)
└── 📋 logs/                       # Execution logs (generated)
```

## Database Schema

### Bronze Layer: `bronze_return_order_data`
Raw data with minimal transformations and data quality tracking.

### Silver Layer: `silver_customer_features`
Customer-level features with business logic applied.

### Gold Layer: `gold_cluster_processed`
Scaled, cleaned features ready for machine learning.

### Results: `clustering_results`, `cluster_summary`
Final customer segments with business interpretations.

## Clustering Methodology

### Phase 1: Initial DBSCAN
- **Purpose**: Identify outliers and natural density patterns
- **Parameters**: eps=0.5, min_samples=5 (configurable)
- **Output**: Core customer patterns vs. outliers

### Phase 2: K-means Optimization
- **Purpose**: Create stable, interpretable main segments
- **Method**: Test K=3-15 using silhouette score, elbow method
- **Validation**: Calinski-Harabasz, Davies-Bouldin scores

### Phase 3: Sub-clustering DBSCAN
- **Purpose**: Refine segments for personalization
- **Parameters**: eps=0.3, min_samples=3 (configurable)
- **Output**: Detailed customer micro-segments

## Business Applications

### 🎯 **Customer Archetypes Identified**
- **High Returners**: Frequent returns, potential dissatisfaction
- **Loyal Customers**: Low returns, high engagement
- **Veteran Shoppers**: Long tenure, established patterns  
- **New Customers**: Recent acquisition, pattern forming
- **Frequent Buyers**: High order volume, various return rates

### 💼 **Actionable Strategies**
- **Retention Programs**: Target high-value, high-return customers
- **Quality Initiatives**: Address systematic return patterns
- **Personalization**: Tailor experiences by customer segment
- **Fraud Detection**: Identify unusual return behaviors
- **Inventory Management**: Optimize based on return predictions

## Configuration Options

### Clustering Parameters
```python
CLUSTERING_CONFIG = {
    'dbscan_initial': {
        'eps': 0.5,              # Distance threshold
        'min_samples': 5,        # Neighborhood size
    },
    'kmeans': {
        'n_clusters_range': range(3, 15),  # K values to test
    },
    'evaluation': {
        'min_silhouette_score': 0.3,      # Quality threshold
        'max_noise_ratio': 0.15,          # Acceptable outlier %
    }
}
```

### Data Quality Thresholds
```python
QUALITY_CHECKS = {
    'min_customers': 1000,               # Minimum dataset size
    'min_orders_per_customer': 2,        # Customer eligibility
    'max_missing_data_ratio': 0.3,       # Data completeness
    'min_feature_completeness': 0.5,     # Feature quality
}
```

## Performance Optimization

### Memory Management
- **Batch Processing**: 50K customer chunks for complex features
- **Optimized Queries**: Efficient SQL with proper indexing
- **Memory Monitoring**: Automatic scaling based on available RAM

### Scalability Features
- **Configurable Batch Sizes**: Adjust for different hardware
- **Progress Logging**: Monitor long-running operations
- **Error Recovery**: Robust handling of data quality issues

## Data Requirements

### Input Data Format (CSV)
Required columns:
```
SALES_ORDER_NO, CUSTOMER_EMAILID, ORDER_DATE, SKU, SALES_QTY,
RETURN_QTY, UNITS_RETURNED_FLAG, RETURN_DATE, RETURN_NO,
RETURN_COMMENT, Q_CLS_ID, Q_SKU_ID, CLASS_, BRAND_
```

### Data Quality Handling
- **Email Standardization**: Case-insensitive consolidation
- **Date Validation**: Automatic correction of invalid return dates
- **Missing Value Imputation**: Feature-type specific strategies
- **Outlier Detection**: Statistical and business logic validation

## Output Files

### 📊 `clustering_results.xlsx`
Comprehensive business report with:
- **Executive Summary**: Key findings and recommendations
- **Cluster Profiles**: Detailed segment characteristics
- **Customer Assignments**: Individual customer-to-cluster mapping
- **Feature Analysis**: Statistical summaries and importance
- **Quality Metrics**: Clustering validation scores
- **Visualizations**: Charts and graphs for stakeholder presentation

### 📋 Log Files
- `pipeline_execution_[timestamp].log`: Complete execution history
- `feature_creation.log`: Feature engineering details
- `cluster_preprocessing.log`: Data preparation steps
- `clustering_metadata.json`: Technical parameters and results

## Monitoring & Maintenance

### Data Quality Monitoring
- **Automated Validation**: Built-in quality checks with alerts
- **Trend Analysis**: Monitor data distribution changes over time
- **Performance Tracking**: Clustering quality metrics

### Model Maintenance
- **Cluster Stability**: Track segment consistency over time
- **Business Validation**: Regular stakeholder review cycles
- **Parameter Tuning**: A/B testing for optimization

## Troubleshooting

### Common Issues

**🔍 Low Silhouette Score (<0.3)**
- Increase feature variance threshold
- Adjust DBSCAN eps parameter
- Review feature selection criteria

**⚠️ High Noise Ratio (>15%)**
- Reduce DBSCAN eps parameter
- Increase min_samples requirement
- Review outlier detection sensitivity

**💾 Memory Issues**
- Reduce batch size in configuration
- Implement incremental processing
- Consider feature dimensionality reduction

**🏃‍♂️ Long Execution Times**
- Enable parallel processing (n_jobs=-1)
- Optimize complex feature calculations
- Consider sampling for initial development

### Support Resources
- **Log Analysis**: Check execution logs for detailed error information
- **Configuration Tuning**: Adjust parameters based on data characteristics
- **Business Validation**: Engage stakeholders for cluster interpretation

## Future Enhancements

### 💰 Order Value Integration
- Placeholder infrastructure ready for pricing data
- Revenue-based clustering refinements
- Value-tier customer segmentation

### 🔄 Real-time Scoring
- Streaming feature calculation
- Live cluster assignment API
- Dynamic segment updates

### 🤖 Advanced Analytics
- Churn prediction models
- Lifetime value estimation
- Recommendation system integration

---

## License & Support

This pipeline is designed for business analytics and customer segmentation use cases. For technical support or feature requests, please refer to the execution logs and configuration documentation.

**Last Updated**: December 2024  
**Version**: 1.0.0  
**Compatibility**: Python 3.8+, pandas 1.3+, scikit-learn 1.0+
