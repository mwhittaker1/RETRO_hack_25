"""
Feature Creation Pipeline Runner
Executes feature engineering and populates silver layer
"""

import sys
import pandas as pd
import logging
from datetime import datetime
import traceback
from typing import Dict, List

from db import get_connection
from features import create_all_features

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('feature_creation.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def merge_feature_dataframes(features: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Merge all feature dataframes into a single customer feature dataset"""
    
    logger.info("Merging feature dataframes...")
    
    # Start with basic volume metrics as the base
    merged_df = features['basic_volume'].copy()
    
    # Add temporal features (includes customer_emailid)
    temporal_features = ['customer_lifetime_days', 'avg_days_to_return', 'return_timing_spread', 
                        'customer_tenure_stage', 'first_order_date', 'last_order_date']
    merged_df = merged_df.merge(
        features['temporal'][['customer_emailid'] + temporal_features], 
        on='customer_emailid', how='left'
    )
    
    # Add return behavior features
    return_features = ['return_rate', 'return_ratio', 'return_product_variety', 
                      'avg_returns_per_order', 'return_frequency_ratio', 'return_intensity']
    merged_df = merged_df.merge(
        features['return_behavior'][['customer_emailid'] + return_features], 
        on='customer_emailid', how='left'
    )
    
    # Add consecutive returns
    consecutive_features = ['consecutive_returns', 'avg_consecutive_returns']
    merged_df = merged_df.merge(
        features['consecutive'][['customer_emailid'] + consecutive_features], 
        on='customer_emailid', how='left'
    )
    
    # Add recency analysis
    recency_features = ['recent_orders', 'recent_returns', 'recent_vs_avg_ratio', 'behavior_stability_score']
    merged_df = merged_df.merge(
        features['recency'][['customer_emailid'] + recency_features], 
        on='customer_emailid', how='left'
    )
    
    # Add category intelligence
    category_features = ['category_diversity_score', 'category_loyalty_score', 'high_return_category_affinity']
    merged_df = merged_df.merge(
        features['category'][['customer_emailid'] + category_features], 
        on='customer_emailid', how='left'
    )
    
    # Add adjacency features
    adjacency_features = ['sku_adjacency_orders', 'sku_adjacency_returns', 
                         'sku_adjacency_timing', 'sku_adjacency_return_timing']
    merged_df = merged_df.merge(
        features['adjacency'][['customer_emailid'] + adjacency_features], 
        on='customer_emailid', how='left'
    )
    
    # Add seasonal features
    seasonal_features = ['seasonal_susceptibility_orders', 'seasonal_susceptibility_returns']
    merged_df = merged_df.merge(
        features['seasonal'][['customer_emailid'] + seasonal_features], 
        on='customer_emailid', how='left'
    )
    
    # Add trend features
    trend_features = ['trend_product_category_order_rate', 'trend_product_category_return_rate']
    merged_df = merged_df.merge(
        features['trend'][['customer_emailid'] + trend_features], 
        on='customer_emailid', how='left'
    )
    
    logger.info(f"Merged features: {len(merged_df)} customers, {len(merged_df.columns)} features")
    return merged_df

def add_metadata_features(conn, merged_df: pd.DataFrame, features: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Add metadata and derived features to the merged dataset"""
    
    logger.info("Adding metadata features...")
    
    # Get basic customer stats from bronze layer
    metadata_query = """
    WITH customer_stats AS (
        SELECT 
            customer_emailid,
            lower(trim(customer_emailid)) as customer_emailid_cleaned,
            sum(sales_qty) as total_sales_qty,
            sum(return_qty) as total_return_qty,
            min(order_date) as first_order_date_bronze,
            max(order_date) as last_order_date_bronze
        FROM bronze_return_order_data
        GROUP BY customer_emailid
    )
    SELECT * FROM customer_stats;
    """
    
    metadata_df = conn.execute(metadata_query).fetchdf()
    
    # Merge metadata
    merged_df = merged_df.merge(metadata_df, on='customer_emailid', how='left')
    
    # Fill missing values with appropriate defaults
    fill_values = {
        'sales_order_no_nunique': 0,
        'sku_nunique': 0,
        'items_returned_count': 0,
        'sales_qty_mean': 0,
        'avg_order_size': 0,
        'return_rate': 0,
        'return_ratio': 0,
        'return_product_variety': 0,
        'avg_returns_per_order': 0,
        'return_frequency_ratio': 0,
        'return_intensity': 0,
        'consecutive_returns': 0,
        'avg_consecutive_returns': 0,
        'customer_lifetime_days': 0,
        'avg_days_to_return': 0,
        'return_timing_spread': 0,
        'customer_tenure_stage': 'New',
        'recent_orders': 0,
        'recent_returns': 0,
        'recent_vs_avg_ratio': 0,
        'behavior_stability_score': 1.0,
        'category_diversity_score': 0,
        'category_loyalty_score': 1.0,
        'high_return_category_affinity': 0,
        'sku_adjacency_orders': 0,
        'sku_adjacency_returns': 0,
        'sku_adjacency_timing': 0,
        'sku_adjacency_return_timing': 0,
        'seasonal_susceptibility_orders': 0,
        'seasonal_susceptibility_returns': 0,
        'trend_product_category_order_rate': 0,
        'trend_product_category_return_rate': 0
    }
    
    merged_df = merged_df.fillna(fill_values)
    
    # Add monetary value features
    monetary_features = ['avg_order_value', 'avg_return_value', 'high_value_return_affinity']
    merged_df = merged_df.merge(
        features['monetary'][['customer_emailid'] + monetary_features], 
        on='customer_emailid', how='left'
    )
    
    # Add feature_calculation_date column
    merged_df['feature_calculation_date'] = datetime.now()
    
    # Add product_category_loyalty column (this appears to be missing too)
    merged_df['product_category_loyalty'] = merged_df['category_loyalty_score'].copy()
    
    # Add data quality flags
    data_quality_flags = []
    
    # Flag customers with extreme values
    if (merged_df['return_rate'] > 0.95).any():
        data_quality_flags.append('HIGH_RETURN_RATE')
    
    if (merged_df['avg_days_to_return'] > 60).any():
        data_quality_flags.append('LATE_RETURNS')
    
    if (merged_df['consecutive_returns'] > 10).any():
        data_quality_flags.append('HIGH_CONSECUTIVE_RETURNS')
    
    merged_df['data_quality_flags'] = ';'.join(data_quality_flags) if data_quality_flags else ''
    
    logger.info(f"Added metadata features. Final dataset: {len(merged_df)} customers, {len(merged_df.columns)} features")
    return merged_df

def validate_silver_layer_data(df: pd.DataFrame) -> Dict[str, any]:
    """Validate the silver layer data before insertion"""
    
    logger.info("Validating silver layer data...")
    
    validation_results = {
        'total_customers': len(df),
        'missing_customer_emails': df['customer_emailid'].isnull().sum(),
        'duplicate_customers': df['customer_emailid'].duplicated().sum(),
        'features_with_nulls': {},
        'business_logic_violations': {},
        'validation_passed': True
    }
    
    # Check for nulls in key features
    key_features = ['sales_order_no_nunique', 'return_rate', 'customer_lifetime_days']
    for feature in key_features:
        null_count = df[feature].isnull().sum()
        if null_count > 0:
            validation_results['features_with_nulls'][feature] = null_count
            logger.warning(f"Feature '{feature}' has {null_count} null values")
    
    # Business logic validation
    if (df['return_rate'] < 0).any() or (df['return_rate'] > 1).any():
        invalid_count = ((df['return_rate'] < 0) | (df['return_rate'] > 1)).sum()
        validation_results['business_logic_violations']['return_rate_out_of_range'] = invalid_count
        validation_results['validation_passed'] = False
        logger.error(f"return_rate has {invalid_count} values outside [0,1] range")
    
    if (df['customer_lifetime_days'] < 0).any():
        invalid_count = (df['customer_lifetime_days'] < 0).sum()
        validation_results['business_logic_violations']['negative_lifetime'] = invalid_count
        validation_results['validation_passed'] = False
        logger.error(f"customer_lifetime_days has {invalid_count} negative values")
    
    # Summary statistics
    validation_results['avg_return_rate'] = df['return_rate'].mean()
    validation_results['avg_order_count'] = df['sales_order_no_nunique'].mean()
    validation_results['avg_lifetime_days'] = df['customer_lifetime_days'].mean()
    
    if validation_results['validation_passed']:
        logger.info("✅ Silver layer data validation PASSED")
    else:
        logger.error("❌ Silver layer data validation FAILED")
    
    return validation_results

def insert_silver_layer_data(conn, df: pd.DataFrame) -> bool:
    """Insert the merged feature data into silver layer"""
    
    logger.info("Inserting data into silver_customer_features table...")
    
    try:
        # Clear existing data
        conn.execute("DELETE FROM silver_customer_features;")
        logger.info("Cleared existing silver layer data")
        
        # Prepare data for insertion
        df_insert = df.copy()
        
        # Ensure datetime columns are properly formatted
        datetime_columns = ['first_order_date', 'last_order_date']
        for col in datetime_columns:
            if col in df_insert.columns:
                df_insert[col] = pd.to_datetime(df_insert[col])
        
        # Check column counts before insertion
        silver_columns = conn.execute("PRAGMA table_info(silver_customer_features);").fetchdf()
        logger.info(f"Silver table has {len(silver_columns)} columns")
        logger.info(f"DataFrame has {len(df_insert.columns)} columns")
        
        # Log column names for debugging
        silver_column_names = silver_columns['name'].tolist()
        df_column_names = df_insert.columns.tolist()
        
        # Find missing columns
        missing_columns = set(silver_column_names) - set(df_column_names)
        extra_columns = set(df_column_names) - set(silver_column_names)
        
        if missing_columns:
            logger.warning(f"Missing columns in DataFrame: {missing_columns}")
            
        if extra_columns:
            logger.warning(f"Extra columns in DataFrame not in silver table: {extra_columns}")
            
        # Log column order differences
        if silver_column_names != df_column_names:
            logger.warning("Column order mismatch between table and DataFrame")
            logger.warning(f"Silver table columns: {silver_column_names}")
            logger.warning(f"DataFrame columns: {df_column_names}")
            
            # Reorder DataFrame columns to match table order
            common_columns = [col for col in silver_column_names if col in df_column_names]
            if len(common_columns) == len(silver_column_names):
                df_insert = df_insert[common_columns]
                logger.info("Reordered DataFrame columns to match table schema")
            else:
                logger.error(f"Cannot insert: DataFrame missing columns required by table: {missing_columns}")
                return False
        
        # Insert in chunks to handle large datasets
        chunk_size = 10000
        total_inserted = 0
        
        for i in range(0, len(df_insert), chunk_size):
            chunk = df_insert[i:i+chunk_size]
            conn.execute("INSERT INTO silver_customer_features SELECT * FROM chunk")
            total_inserted += len(chunk)
            logger.info(f"Inserted chunk {i//chunk_size + 1}: {total_inserted}/{len(df_insert)} records")
        
        # Verify insertion
        final_count = conn.execute("SELECT count(*) FROM silver_customer_features").fetchone()[0]
        
        if final_count == len(df_insert):
            logger.info(f"✅ Successfully inserted {final_count} customer records into silver layer")
            return True
        else:
            logger.error(f"❌ Insertion mismatch: expected {len(df_insert)}, got {final_count}")
            return False
            
    except Exception as e:
        logger.error(f"Failed to insert silver layer data: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def generate_feature_summary_report(conn, features: Dict[str, pd.DataFrame]) -> None:
    """Generate a summary report of created features"""
    
    logger.info("Generating feature summary report...")
    
    report = []
    report.append("="*60)
    report.append("CUSTOMER FEATURE CREATION SUMMARY REPORT")
    report.append("="*60)
    report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # Feature group summary
    report.append("FEATURE GROUPS CREATED:")
    report.append("-" * 30)
    total_features = 0
    for group_name, df in features.items():
        feature_count = len(df.columns) - 1  # Subtract customer_emailid
        total_features += feature_count
        report.append(f"{group_name:20}: {len(df):6,} customers, {feature_count:2} features")
    
    report.append("")
    report.append(f"Total features created: {total_features}")
    
    # Silver layer summary
    silver_stats = conn.execute("""
        SELECT 
            count(*) as customer_count,
            avg(return_rate) as avg_return_rate,
            avg(sales_order_no_nunique) as avg_orders,
            avg(customer_lifetime_days) as avg_lifetime_days,
            count(*) FILTER (WHERE customer_tenure_stage = 'Veteran') as veteran_customers,
            count(*) FILTER (WHERE return_rate > 0.5) as high_return_customers
        FROM silver_customer_features;
    """).fetchone()
    
    report.append("")
    report.append("SILVER LAYER STATISTICS:")
    report.append("-" * 30)
    report.append(f"Total customers:        {silver_stats[0]:,}")
    report.append(f"Average return rate:    {silver_stats[1]:.3f}")
    report.append(f"Average orders:         {silver_stats[2]:.1f}")
    report.append(f"Average lifetime days:  {silver_stats[3]:.0f}")
    report.append(f"Veteran customers:      {silver_stats[4]:,}")
    report.append(f"High return customers:  {silver_stats[5]:,}")
    
    # Feature completeness
    completeness_stats = conn.execute("""
        SELECT 
            'basic_volume' as feature_group,
            100.0 * count(*) FILTER (WHERE sales_order_no_nunique > 0) / count(*) as completeness_pct
        FROM silver_customer_features
        UNION ALL
        SELECT 
            'temporal' as feature_group,
            100.0 * count(*) FILTER (WHERE customer_lifetime_days > 0) / count(*) as completeness_pct
        FROM silver_customer_features
        UNION ALL
        SELECT 
            'adjacency' as feature_group,
            100.0 * count(*) FILTER (WHERE sku_adjacency_orders > 0) / count(*) as completeness_pct
        FROM silver_customer_features;
    """).fetchdf()
    
    report.append("")
    report.append("FEATURE COMPLETENESS:")
    report.append("-" * 30)
    for _, row in completeness_stats.iterrows():
        report.append(f"{row['feature_group']:15}: {row['completeness_pct']:6.1f}%")
    
    # Data quality issues
    dq_issues = conn.execute("""
        SELECT issue_type, count(*) as issue_count
        FROM data_quality_issues
        WHERE resolved = FALSE
        GROUP BY issue_type
        ORDER BY count(*) DESC;
    """).fetchdf()
    
    if len(dq_issues) > 0:
        report.append("")
        report.append("DATA QUALITY ISSUES:")
        report.append("-" * 30)
        for _, row in dq_issues.iterrows():
            report.append(f"{row['issue_type']:25}: {row['issue_count']:,} issues")
    
    report.append("")
    report.append("="*60)
    
    # Print and save report
    report_text = "\n".join(report)
    print(report_text)
    
    with open(f"feature_summary_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt", 'w') as f:
        f.write(report_text)
    
    logger.info("Feature summary report generated and saved")

def main():
    """Main execution function"""
    
    logger.info("Starting feature creation pipeline...")
    
    try:
        # Connect to database
        conn = get_connection("customer_clustering.db")
        logger.info("Connected to customer clustering database")
        
        # Check bronze layer data availability
        bronze_count = conn.execute("SELECT count(*) FROM bronze_return_order_data").fetchone()[0]
        if bronze_count == 0:
            logger.error("No data found in bronze layer. Please run db.py first to load data.")
            return False
        
        logger.info(f"Bronze layer contains {bronze_count:,} records")
        
        # Create all features
        features = create_all_features(conn, run_tests=True)
        
        if not features:
            logger.error("Feature creation failed - no features returned")
            return False
        
        # Merge all feature dataframes
        merged_features = merge_feature_dataframes(features)
        
        # Add metadata and data quality features
        final_features = add_metadata_features(conn, merged_features, features)
        
        # Validate the final dataset
        validation_results = validate_silver_layer_data(final_features)
        
        if not validation_results['validation_passed']:
            logger.error("Data validation failed. Review validation results before proceeding.")
            return False
        
        # Insert into silver layer
        success = insert_silver_layer_data(conn, final_features)
        
        if not success:
            logger.error("Failed to insert data into silver layer")
            return False
        
        # Generate summary report
        generate_feature_summary_report(conn, features)
        
        logger.info("✅ Feature creation pipeline completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Feature creation pipeline failed: {str(e)}")
        logger.error(traceback.format_exc())
        return False
        
    finally:
        conn.close()

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
