"""
Customer Return Clustering Features Engineering
Modular functions for creating customer-level features from return/order data
"""

import duckdb
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import time

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FeatureEngineering:
    def __init__(self, conn: duckdb.DuckDBPyConnection):
        self.conn = conn
        self.feature_log = []
    
    def log_feature_creation(self, feature_name: str, start_time: float, records_processed: int, 
                           warnings: int = 0, errors: int = 0, status: str = 'SUCCESS'):
        """Log feature creation details"""
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Ensure all values are native Python types
        feature_name_py = str(feature_name)
        calculation_start_py = datetime.fromtimestamp(float(start_time))
        calculation_end_py = datetime.fromtimestamp(float(end_time))
        execution_time_py = float(execution_time)
        records_processed_py = int(records_processed)
        warnings_py = int(warnings)
        errors_py = int(errors)
        status_py = str(status)
        
        # Get next log_id (max + 1)
        next_id = 1  # Default if no records exist
        try:
            max_id = self.conn.execute("SELECT MAX(log_id) FROM feature_calculation_log").fetchone()[0]
            if max_id is not None:
                next_id = int(max_id) + 1
        except:
            # If table doesn't exist or is empty, use default value 1
            pass

        self.conn.execute("""
            INSERT INTO feature_calculation_log 
            (log_id, feature_name, calculation_start, calculation_end, execution_time_seconds, 
             records_processed, warnings_count, errors_count, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [next_id, feature_name_py, calculation_start_py, calculation_end_py,
              execution_time_py, records_processed_py, warnings_py, errors_py, status_py])
        
        logger.info(f"Feature '{feature_name}' completed in {execution_time:.2f}s, {records_processed} records processed")
        
        logger.info(f"Feature '{feature_name}' completed in {execution_time:.2f}s, {records_processed} records processed")
    
    def validate_feature_business_logic(self, feature_name: str, df: pd.DataFrame, 
                                      column: str, expected_range: Tuple[float, float] = None,
                                      warning_threshold: float = None) -> int:
        """Validate feature against business logic and return warning count"""
        warnings = 0
        
        if expected_range:
            min_val, max_val = expected_range
            out_of_range = (df[column] < min_val) | (df[column] > max_val)
            if out_of_range.sum() > 0:
                warnings += out_of_range.sum()
                logger.warning(f"{feature_name}: {out_of_range.sum()} values outside expected range [{min_val}, {max_val}]")
        
        if warning_threshold:
            above_threshold = df[column] > warning_threshold
            if above_threshold.sum() > 0:
                warnings += above_threshold.sum()
                logger.warning(f"{feature_name}: {above_threshold.sum()} values above warning threshold {warning_threshold}")
        
        # Check for nulls
        null_count = df[column].isnull().sum()
        if null_count > 0:
            logger.info(f"{feature_name}: {null_count} null values found")
        
        return warnings

# =============================================================================
# BASIC VOLUME METRICS
# =============================================================================

def create_basic_volume_metrics(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Create basic volume metrics for customers"""
    start_time = time.time()
    feature_eng = FeatureEngineering(conn)
    
    logger.info("Creating basic volume metrics...")
    
    query = """
    WITH customer_volume AS (
        SELECT 
            customer_emailid,
            count(DISTINCT sales_order_no) as sales_order_no_nunique,
            count(DISTINCT q_sku_id) as sku_nunique,
            sum(CASE WHEN return_qty > 0 THEN 1 ELSE 0 END) as items_returned_count,
            avg(CAST(sales_qty AS DOUBLE)) as sales_qty_mean,
            CAST(count(*) AS DOUBLE) / CAST(count(DISTINCT sales_order_no) AS DOUBLE) as avg_order_size
        FROM bronze_return_order_data
        GROUP BY customer_emailid
    )
    SELECT * FROM customer_volume;
    """
    
    result = conn.execute(query).fetchdf()
    
    # Validate business logic
    warnings = 0
    warnings += feature_eng.validate_feature_business_logic(
        'sales_qty_mean', result, 'sales_qty_mean', warning_threshold=15
    )
    warnings += feature_eng.validate_feature_business_logic(
        'avg_order_size', result, 'avg_order_size', warning_threshold=10
    )
    
    feature_eng.log_feature_creation('basic_volume_metrics', start_time, len(result), warnings)
    return result

def create_return_behavior_patterns(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Create return behavior pattern features"""
    start_time = time.time()
    feature_eng = FeatureEngineering(conn)
    
    logger.info("Creating return behavior patterns...")
    
    query = """
    WITH customer_returns AS (
        SELECT 
            customer_emailid,
            sum(sales_qty) as total_sales_qty,
            sum(return_qty) as total_return_qty,
            count(*) as total_items,
            count(DISTINCT sales_order_no) as total_orders,
            count(DISTINCT q_sku_id) FILTER (WHERE return_qty > 0) as return_product_variety,
            sum(CASE WHEN return_qty > 0 THEN 1 ELSE 0 END) as items_with_returns
        FROM bronze_return_order_data
        GROUP BY customer_emailid
    ),
    with_rates AS (
        SELECT *,
            CAST(total_return_qty AS DOUBLE) / NULLIF(CAST(total_sales_qty AS DOUBLE), 0) as return_rate,
            CAST(total_return_qty AS DOUBLE) / NULLIF(CAST(total_sales_qty AS DOUBLE), 0) as return_ratio,
            CAST(items_with_returns AS DOUBLE) / NULLIF(CAST(total_orders AS DOUBLE), 0) as avg_returns_per_order,
            CAST(items_with_returns AS DOUBLE) / NULLIF(CAST(total_items AS DOUBLE), 0) as return_frequency_ratio
        FROM customer_returns
    ),
    with_intensity AS (
        SELECT *,
            CASE 
                WHEN items_with_returns > 0 THEN CAST(total_return_qty AS DOUBLE) / NULLIF(CAST(items_with_returns AS DOUBLE), 0)
                ELSE 0 
            END as return_intensity
        FROM with_rates
    )
    SELECT 
        customer_emailid,
        COALESCE(return_rate, 0) as return_rate,
        COALESCE(return_ratio, 0) as return_ratio,
        COALESCE(return_product_variety, 0) as return_product_variety,
        COALESCE(avg_returns_per_order, 0) as avg_returns_per_order,
        COALESCE(return_frequency_ratio, 0) as return_frequency_ratio,
        COALESCE(return_intensity, 0) as return_intensity
    FROM with_intensity;
    """
    
    result = conn.execute(query).fetchdf()
    
    # Validate business logic
    warnings = 0
    warnings += feature_eng.validate_feature_business_logic(
        'return_rate', result, 'return_rate', expected_range=(0, 1)
    )
    warnings += feature_eng.validate_feature_business_logic(
        'return_ratio', result, 'return_ratio', expected_range=(0, 1)
    )
    warnings += feature_eng.validate_feature_business_logic(
        'return_frequency_ratio', result, 'return_frequency_ratio', expected_range=(0, 1)
    )
    warnings += feature_eng.validate_feature_business_logic(
        'return_intensity', result, 'return_intensity', expected_range=(0, 1)
    )
    warnings += feature_eng.validate_feature_business_logic(
        'return_product_variety', result, 'return_product_variety', warning_threshold=25
    )
    warnings += feature_eng.validate_feature_business_logic(
        'avg_returns_per_order', result, 'avg_returns_per_order', warning_threshold=8
    )
    
    feature_eng.log_feature_creation('return_behavior_patterns', start_time, len(result), warnings)
    return result

def create_consecutive_returns(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Create consecutive returns features"""
    start_time = time.time()
    feature_eng = FeatureEngineering(conn)
    
    print("Creating consecutive returns features (complex calculation)...")
    
    query = """
    WITH order_returns AS (
        SELECT 
            customer_emailid,
            sales_order_no,
            order_date,
            CASE WHEN sum(return_qty) > 0 THEN 1 ELSE 0 END as has_returns
        FROM bronze_return_order_data
        GROUP BY customer_emailid, sales_order_no, order_date
    ),
    ordered_returns AS (
        SELECT *,
            ROW_NUMBER() OVER (PARTITION BY customer_emailid ORDER BY order_date) as order_sequence,
            LAG(has_returns) OVER (PARTITION BY customer_emailid ORDER BY order_date) as prev_order_returned
        FROM order_returns
        ORDER BY customer_emailid, order_date
    ),
    consecutive_groups AS (
        SELECT *,
            SUM(CASE WHEN has_returns = 1 AND (prev_order_returned = 0 OR prev_order_returned IS NULL) THEN 1 ELSE 0 END) 
                OVER (PARTITION BY customer_emailid ORDER BY order_date 
                     ROWS UNBOUNDED PRECEDING) as consecutive_group
        FROM ordered_returns
        WHERE has_returns = 1
    ),
    consecutive_counts AS (
        SELECT 
            customer_emailid,
            consecutive_group,
            count(*) as consecutive_length
        FROM consecutive_groups
        GROUP BY customer_emailid, consecutive_group
    ),
    customer_consecutive_stats AS (
        SELECT 
            customer_emailid,
            count(*) as consecutive_returns,
            avg(CAST(consecutive_length AS DOUBLE)) as avg_consecutive_returns
        FROM consecutive_counts
        GROUP BY customer_emailid
    )
    SELECT 
        b.customer_emailid,
        COALESCE(c.consecutive_returns, 0) as consecutive_returns,
        COALESCE(c.avg_consecutive_returns, 0) as avg_consecutive_returns
    FROM (SELECT DISTINCT customer_emailid FROM bronze_return_order_data) b
    LEFT JOIN customer_consecutive_stats c ON b.customer_emailid = c.customer_emailid;
    """
    
    result = conn.execute(query).fetchdf()
    
    # Validate business logic
    warnings = 0
    warnings += feature_eng.validate_feature_business_logic(
        'consecutive_returns', result, 'consecutive_returns', warning_threshold=6
    )
    warnings += feature_eng.validate_feature_business_logic(
        'avg_consecutive_returns', result, 'avg_consecutive_returns', warning_threshold=3
    )
    
    feature_eng.log_feature_creation('consecutive_returns', start_time, len(result), warnings)
    return result

# =============================================================================
# TEMPORAL & TIMING PATTERNS
# =============================================================================

def create_temporal_patterns(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Create temporal and timing pattern features"""
    start_time = time.time()
    feature_eng = FeatureEngineering(conn)
    
    logger.info("Creating temporal patterns...")
    
    query = """
    WITH customer_timeline AS (
        SELECT 
            customer_emailid,
            min(order_date) as first_order_date,
            max(order_date) as last_order_date,
            DATE_DIFF('day', min(order_date), max(order_date)) as customer_lifetime_days
        FROM bronze_return_order_data
        GROUP BY customer_emailid
    ),
    return_timing AS (
        SELECT 
            customer_emailid,
            avg(DATE_DIFF('day', order_date, return_date)) as avg_days_to_return,
            stddev(DATE_DIFF('day', order_date, return_date)) as return_timing_spread
        FROM bronze_return_order_data
        WHERE return_qty > 0 AND return_date IS NOT NULL
        GROUP BY customer_emailid
    ),
    tenure_stage AS (
        SELECT *,
            CASE 
                WHEN customer_lifetime_days <= 90 THEN 'New'
                WHEN customer_lifetime_days <= 180 THEN 'Growing'
                WHEN customer_lifetime_days <= 365 THEN 'Mature'
                ELSE 'Veteran'
            END as customer_tenure_stage
        FROM customer_timeline
    )
    SELECT 
        t.customer_emailid,
        t.customer_lifetime_days,
        COALESCE(r.avg_days_to_return, 0) as avg_days_to_return,
        COALESCE(r.return_timing_spread, 0) as return_timing_spread,
        t.customer_tenure_stage,
        t.first_order_date,
        t.last_order_date
    FROM tenure_stage t
    LEFT JOIN return_timing r ON t.customer_emailid = r.customer_emailid;
    """
    
    result = conn.execute(query).fetchdf()
    
    # Validate business logic
    warnings = 0
    warnings += feature_eng.validate_feature_business_logic(
        'avg_days_to_return', result, 'avg_days_to_return', warning_threshold=60
    )
    warnings += feature_eng.validate_feature_business_logic(
        'return_timing_spread', result, 'return_timing_spread', warning_threshold=45
    )
    
    # Check for negative days to return
    negative_days = result['avg_days_to_return'] < 0
    if negative_days.sum() > 0:
        warnings += negative_days.sum()
        logger.warning(f"avg_days_to_return: {negative_days.sum()} customers with negative return timing")
    
    feature_eng.log_feature_creation('temporal_patterns', start_time, len(result), warnings)
    return result

def create_recency_analysis(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Create recency analysis features"""
    start_time = time.time()
    feature_eng = FeatureEngineering(conn)
    
    logger.info("Creating recency analysis...")
    
    # Calculate cutoff date (90 days ago from latest order)
    latest_date_result = conn.execute("SELECT max(order_date) FROM bronze_return_order_data").fetchone()
    latest_date = latest_date_result[0]
    cutoff_date = latest_date - timedelta(days=90)
    
    query = """
    WITH recent_activity AS (
        SELECT 
            customer_emailid,
            count(DISTINCT sales_order_no) FILTER (WHERE order_date >= ?) as recent_orders,
            sum(CASE WHEN return_qty > 0 THEN 1 ELSE 0 END) FILTER (WHERE order_date >= ?) as recent_returns
        FROM bronze_return_order_data
        GROUP BY customer_emailid
    ),
    historical_activity AS (
        SELECT 
            customer_emailid,
            count(DISTINCT sales_order_no) as total_orders,
            sum(CASE WHEN return_qty > 0 THEN 1 ELSE 0 END) as total_returns,
            DATE_DIFF('day', min(order_date), max(order_date)) as customer_lifetime_days
        FROM bronze_return_order_data
        GROUP BY customer_emailid
    ),
    with_ratios AS (
        SELECT 
            h.customer_emailid,
            COALESCE(r.recent_orders, 0) as recent_orders,
            COALESCE(r.recent_returns, 0) as recent_returns,
            h.total_orders,
            h.total_returns,
            h.customer_lifetime_days,
            -- Calculate recent vs average ratio
            CASE 
                WHEN h.total_returns > 0 AND h.customer_lifetime_days > 90 THEN
                    (CAST(r.recent_returns AS DOUBLE) / 90.0) / 
                    (CAST(h.total_returns AS DOUBLE) / CAST(h.customer_lifetime_days AS DOUBLE))
                ELSE 0
            END as recent_vs_avg_ratio
        FROM historical_activity h
        LEFT JOIN recent_activity r ON h.customer_emailid = r.customer_emailid
    )
    SELECT 
        customer_emailid,
        recent_orders,
        recent_returns,
        COALESCE(recent_vs_avg_ratio, 0) as recent_vs_avg_ratio,
        -- Simple behavior stability score (1 = stable, 0 = volatile)
        CASE 
            WHEN recent_vs_avg_ratio BETWEEN 0.8 AND 1.2 THEN 1.0
            WHEN recent_vs_avg_ratio BETWEEN 0.5 AND 1.5 THEN 0.7
            WHEN recent_vs_avg_ratio BETWEEN 0.2 AND 2.0 THEN 0.4
            ELSE 0.1
        END as behavior_stability_score
    FROM with_ratios;
    """
    
    result = conn.execute(query, [cutoff_date, cutoff_date]).fetchdf()
    
    # Validate business logic
    warnings = 0
    warnings += feature_eng.validate_feature_business_logic(
        'behavior_stability_score', result, 'behavior_stability_score', expected_range=(0, 1)
    )
    
    feature_eng.log_feature_creation('recency_analysis', start_time, len(result), warnings)
    return result

# =============================================================================
# PRODUCT & CATEGORY INTELLIGENCE
# =============================================================================

def create_category_intelligence(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Create product category intelligence features"""
    start_time = time.time()
    feature_eng = FeatureEngineering(conn)
    
    logger.info("Creating category intelligence features...")
    
    query = """
    WITH customer_categories AS (
        SELECT 
            customer_emailid,
            q_cls_id,
            count(*) as category_purchases,
            sum(return_qty) as category_returns,
            sum(sales_qty) as category_sales
        FROM bronze_return_order_data
        WHERE q_cls_id IS NOT NULL
        GROUP BY customer_emailid, q_cls_id
    ),
    customer_totals AS (
        SELECT 
            customer_emailid,
            sum(category_purchases) as total_purchases,
            count(DISTINCT q_cls_id) as unique_categories,
            sum(category_returns) as total_returns
        FROM customer_categories
        GROUP BY customer_emailid
    ),
    loyalty_calculations AS (
        SELECT 
            cc.customer_emailid,
            ct.unique_categories,
            ct.total_purchases,
            ct.total_returns,
            -- Category loyalty: sum of squared proportions (higher = more loyal to fewer categories)
            sum(POWER(CAST(cc.category_purchases AS DOUBLE) / CAST(ct.total_purchases AS DOUBLE), 2)) as category_loyalty_raw,
            -- Return affinity by category (Z-score style)
            avg(CASE WHEN cc.category_sales > 0 THEN CAST(cc.category_returns AS DOUBLE) / CAST(cc.category_sales AS DOUBLE) ELSE 0 END) as avg_category_return_rate
        FROM customer_categories cc
        JOIN customer_totals ct ON cc.customer_emailid = ct.customer_emailid
        GROUP BY cc.customer_emailid, ct.unique_categories, ct.total_purchases, ct.total_returns
    )
    SELECT 
        customer_emailid,
        -- Category diversity (normalized by total categories available)
        CAST(unique_categories AS DOUBLE) / 20.0 as category_diversity_score,  -- Assuming ~20 categories max
        -- Category loyalty (0-1 scale)
        category_loyalty_raw as category_loyalty_score,
        -- High return category affinity (simplified)
        CASE 
            WHEN avg_category_return_rate > 0.3 THEN 1.0
            WHEN avg_category_return_rate > 0.15 THEN 0.7
            WHEN avg_category_return_rate > 0.05 THEN 0.4
            ELSE 0.1
        END as high_return_category_affinity
    FROM loyalty_calculations;
    """
    
    result = conn.execute(query).fetchdf()
    
    # Validate business logic
    warnings = 0
    warnings += feature_eng.validate_feature_business_logic(
        'category_diversity_score', result, 'category_diversity_score', warning_threshold=1.5
    )
    warnings += feature_eng.validate_feature_business_logic(
        'category_loyalty_score', result, 'category_loyalty_score', expected_range=(0, 1)
    )
    warnings += feature_eng.validate_feature_business_logic(
        'high_return_category_affinity', result, 'high_return_category_affinity', expected_range=(0, 1)
    )
    
    feature_eng.log_feature_creation('category_intelligence', start_time, len(result), warnings)
    return result

# =============================================================================
# ADJACENCY & REPEAT BEHAVIOR
# =============================================================================

def create_adjacency_features(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Create SKU adjacency and repeat behavior features"""
    start_time = time.time()
    feature_eng = FeatureEngineering(conn)
    
    print("Creating adjacency features (complex calculation)...")
    
    query = """
    WITH customer_sku_orders AS (
        SELECT 
            customer_emailid,
            q_sku_id,
            sales_order_no,
            order_date,
            return_qty,
            ROW_NUMBER() OVER (PARTITION BY customer_emailid, q_sku_id ORDER BY order_date) as sku_order_sequence
        FROM bronze_return_order_data
        WHERE q_sku_id IS NOT NULL
    ),
    sku_adjacency_analysis AS (
        SELECT 
            a.customer_emailid,
            a.q_sku_id as sku_a,
            b.q_sku_id as sku_b,
            a.order_date as order_a_date,
            b.order_date as order_b_date,
            abs(DATE_DIFF('day', a.order_date, b.order_date)) as days_between,
            CASE WHEN a.return_qty > 0 THEN 1 ELSE 0 END as sku_a_returned,
            CASE WHEN b.return_qty > 0 THEN 1 ELSE 0 END as sku_b_returned
        FROM customer_sku_orders a
        JOIN customer_sku_orders b ON a.customer_emailid = b.customer_emailid
        WHERE a.q_sku_id != b.q_sku_id
        AND abs(DATE_DIFF('day', a.order_date, b.order_date)) <= 14  -- Within 14 days
        AND a.order_date != b.order_date  -- Different orders
    ),
    customer_adjacency_stats AS (
        SELECT 
            customer_emailid,
            count(DISTINCT CONCAT(sku_a, '_', sku_b)) as sku_adjacency_orders,
            count(*) FILTER (WHERE sku_a_returned = 1 AND sku_b_returned = 1) as sku_adjacency_returns,
            avg(CAST(days_between AS DOUBLE)) as sku_adjacency_timing,
            avg(CASE WHEN sku_a_returned = 1 AND sku_b_returned = 1 THEN CAST(days_between AS DOUBLE) ELSE NULL END) as sku_adjacency_return_timing
        FROM sku_adjacency_analysis
        GROUP BY customer_emailid
    )
    SELECT 
        b.customer_emailid,
        COALESCE(a.sku_adjacency_orders, 0) as sku_adjacency_orders,
        COALESCE(a.sku_adjacency_returns, 0) as sku_adjacency_returns,
        COALESCE(a.sku_adjacency_timing, 0) as sku_adjacency_timing,
        COALESCE(a.sku_adjacency_return_timing, 0) as sku_adjacency_return_timing
    FROM (SELECT DISTINCT customer_emailid FROM bronze_return_order_data) b
    LEFT JOIN customer_adjacency_stats a ON b.customer_emailid = a.customer_emailid;
    """
    
    result = conn.execute(query).fetchdf()
    
    # Validate business logic
    warnings = 0
    warnings += feature_eng.validate_feature_business_logic(
        'sku_adjacency_timing', result, 'sku_adjacency_timing', warning_threshold=14
    )
    warnings += feature_eng.validate_feature_business_logic(
        'sku_adjacency_return_timing', result, 'sku_adjacency_return_timing', warning_threshold=14
    )
    
    feature_eng.log_feature_creation('adjacency_features', start_time, len(result), warnings)
    return result

# =============================================================================
# SEASONAL & TREND SUSCEPTIBILITY
# =============================================================================

def create_seasonal_features(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Create seasonal susceptibility features for customers with >2 years history"""
    start_time = time.time()
    feature_eng = FeatureEngineering(conn)
    
    logger.info("Creating seasonal features...")
    
    query = """
    WITH customer_tenure AS (
        SELECT 
            customer_emailid,
            min(order_date) as first_order,
            max(order_date) as last_order,
            DATE_DIFF('day', min(order_date), max(order_date)) as lifetime_days
        FROM bronze_return_order_data
        GROUP BY customer_emailid
        HAVING DATE_DIFF('day', min(order_date), max(order_date)) > 730  -- >2 years
    ),
    seasonal_activity AS (
        SELECT 
            b.customer_emailid,
            CASE 
                WHEN EXTRACT(month FROM b.order_date) IN (10, 11, 12) THEN 'Fall'
                WHEN EXTRACT(month FROM b.order_date) IN (1, 2, 3) THEN 'Winter'
                WHEN EXTRACT(month FROM b.order_date) IN (4, 5, 6) THEN 'Spring'
                ELSE 'Summer'
            END as season,
            count(*) as seasonal_orders,
            sum(b.return_qty) as seasonal_returns
        FROM bronze_return_order_data b
        JOIN customer_tenure ct ON b.customer_emailid = ct.customer_emailid
        GROUP BY b.customer_emailid, 
            CASE 
                WHEN EXTRACT(month FROM b.order_date) IN (10, 11, 12) THEN 'Fall'
                WHEN EXTRACT(month FROM b.order_date) IN (1, 2, 3) THEN 'Winter'
                WHEN EXTRACT(month FROM b.order_date) IN (4, 5, 6) THEN 'Spring'
                ELSE 'Summer'
            END
    ),
    customer_seasonal_stats AS (
        SELECT 
            customer_emailid,
            sum(seasonal_orders) as total_orders,
            sum(seasonal_returns) as total_returns,
            -- Calculate coefficient of variation for seasonal patterns
            stddev(CAST(seasonal_orders AS DOUBLE)) / NULLIF(avg(CAST(seasonal_orders AS DOUBLE)), 0) as seasonal_susceptibility_orders,
            stddev(CAST(seasonal_returns AS DOUBLE)) / NULLIF(avg(CAST(seasonal_returns AS DOUBLE)), 0) as seasonal_susceptibility_returns
        FROM seasonal_activity
        GROUP BY customer_emailid
        HAVING count(DISTINCT season) >= 3  -- Must have activity in at least 3 seasons
    )
    SELECT 
        b.customer_emailid,
        COALESCE(s.seasonal_susceptibility_orders, 0) as seasonal_susceptibility_orders,
        COALESCE(s.seasonal_susceptibility_returns, 0) as seasonal_susceptibility_returns
    FROM (SELECT DISTINCT customer_emailid FROM bronze_return_order_data) b
    LEFT JOIN customer_seasonal_stats s ON b.customer_emailid = s.customer_emailid;
    """
    
    result = conn.execute(query).fetchdf()
    
    feature_eng.log_feature_creation('seasonal_features', start_time, len(result))
    return result

def create_trend_susceptibility(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Create trend susceptibility features based on category patterns"""
    start_time = time.time()
    feature_eng = FeatureEngineering(conn)
    
    logger.info("Creating trend susceptibility features...")
    
    query = """
    WITH category_trends AS (
        SELECT 
            q_cls_id,
            DATE_TRUNC('month', order_date) as order_month,
            count(*) as monthly_category_orders,
            sum(return_qty) as monthly_category_returns
        FROM bronze_return_order_data
        WHERE q_cls_id IS NOT NULL
        GROUP BY q_cls_id, DATE_TRUNC('month', order_date)
    ),
    customer_category_activity AS (
        SELECT 
            b.customer_emailid,
            b.q_cls_id,
            DATE_TRUNC('month', b.order_date) as order_month,
            count(*) as customer_monthly_orders,
            sum(b.return_qty) as customer_monthly_returns,
            ct.monthly_category_orders,
            ct.monthly_category_returns
        FROM bronze_return_order_data b
        JOIN category_trends ct ON b.q_cls_id = ct.q_cls_id 
            AND DATE_TRUNC('month', b.order_date) = ct.order_month
        WHERE b.q_cls_id IS NOT NULL
        GROUP BY b.customer_emailid, b.q_cls_id, DATE_TRUNC('month', b.order_date),
                 ct.monthly_category_orders, ct.monthly_category_returns
    ),
    correlation_analysis AS (
        SELECT 
            customer_emailid,
            -- Simplified trend correlation (customers following category trends)
            corr(CAST(customer_monthly_orders AS DOUBLE), CAST(monthly_category_orders AS DOUBLE)) as trend_product_category_order_rate,
            corr(CAST(customer_monthly_returns AS DOUBLE), CAST(monthly_category_returns AS DOUBLE)) as trend_product_category_return_rate
        FROM customer_category_activity
        GROUP BY customer_emailid
        HAVING count(*) >= 6  -- At least 6 months of data
    )
    SELECT 
        b.customer_emailid,
        COALESCE(c.trend_product_category_order_rate, 0) as trend_product_category_order_rate,
        COALESCE(c.trend_product_category_return_rate, 0) as trend_product_category_return_rate
    FROM (SELECT DISTINCT customer_emailid FROM bronze_return_order_data) b
    LEFT JOIN correlation_analysis c ON b.customer_emailid = c.customer_emailid;
    """
    
    result = conn.execute(query).fetchdf()
    
    # Validate business logic
    warnings = 0
    warnings += feature_eng.validate_feature_business_logic(
        'trend_product_category_order_rate', result, 'trend_product_category_order_rate', expected_range=(-1, 1)
    )
    warnings += feature_eng.validate_feature_business_logic(
        'trend_product_category_return_rate', result, 'trend_product_category_return_rate', expected_range=(-1, 1)
    )
    
    feature_eng.log_feature_creation('trend_susceptibility', start_time, len(result), warnings)
    return result

# =============================================================================
# MONETARY VALUE METRICS
# =============================================================================

def create_monetary_value_metrics(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Create monetary value metrics for customers using actual GROSS values"""
    start_time = time.time()
    feature_eng = FeatureEngineering(conn)
    
    logger.info("Creating monetary value metrics using actual GROSS data...")
    
    # Using actual GROSS (order value) data to calculate monetary features
    query = """
    WITH customer_monetary AS (
        SELECT 
            customer_emailid,
            -- Order counts and total quantities
            COUNT(DISTINCT sales_order_no) AS order_count,
            SUM(sales_qty) AS total_sales_qty,
            SUM(return_qty) AS total_return_qty,
            
            -- Monetary values using GROSS column
            SUM(gross) AS total_gross_value,
            SUM(CASE WHEN return_qty > 0 THEN gross ELSE 0 END) AS returned_gross_value,
            
            -- Per-item values
            SUM(CASE WHEN sales_qty > 0 THEN gross / sales_qty * sales_qty ELSE 0 END) AS total_item_value,
            SUM(CASE WHEN return_qty > 0 THEN gross / sales_qty * return_qty ELSE 0 END) AS returned_item_value,
            
            -- High value orders (above 75th percentile)
            COUNT(DISTINCT CASE 
                WHEN gross > (SELECT PERCENTILE_DISC(0.75) WITHIN GROUP (ORDER BY gross) FROM bronze_return_order_data)
                THEN sales_order_no 
                ELSE NULL 
            END) AS high_value_orders,
            
            -- High value returns
            COUNT(DISTINCT CASE 
                WHEN return_qty > 0 AND gross > (SELECT PERCENTILE_DISC(0.75) WITHIN GROUP (ORDER BY gross) FROM bronze_return_order_data)
                THEN sales_order_no 
                ELSE NULL 
            END) AS high_value_returns
        FROM bronze_return_order_data
        WHERE gross IS NOT NULL
        GROUP BY customer_emailid
    )
    SELECT 
        customer_emailid,
        
        -- Average order value (actual monetary amount)
        CASE 
            WHEN order_count = 0 THEN 0
            ELSE total_gross_value / NULLIF(order_count, 0)
        END AS avg_order_value,
        
        -- Average return value (per return transaction)
        CASE 
            WHEN total_return_qty = 0 THEN 0
            ELSE returned_item_value / NULLIF(total_return_qty, 0)
        END AS avg_return_value,
        
        -- High value return affinity (what percentage of high value orders are returned)
        CASE 
            WHEN high_value_orders = 0 THEN 0
            ELSE CAST(high_value_returns AS DOUBLE) / NULLIF(CAST(high_value_orders AS DOUBLE), 0)
        END * 100 AS high_value_return_affinity
        
    FROM customer_monetary;
    """
    
    df = conn.execute(query).fetchdf()
    
    # Log distribution of values before any scaling
    logger.info(f"Monetary metrics distribution before scaling:")
    logger.info(f"  avg_order_value: min={df['avg_order_value'].min():.2f}, max={df['avg_order_value'].max():.2f}, mean={df['avg_order_value'].mean():.2f}")
    logger.info(f"  avg_return_value: min={df['avg_return_value'].min():.2f}, max={df['avg_return_value'].max():.2f}, mean={df['avg_return_value'].mean():.2f}")
    logger.info(f"  high_value_return_affinity: min={df['high_value_return_affinity'].min():.2f}, max={df['high_value_return_affinity'].max():.2f}, mean={df['high_value_return_affinity'].mean():.2f}")
    
    # Handle potential outliers and scale values to 0-100 range for consistency with other features
    for col in ['avg_order_value', 'avg_return_value']:
        if df[col].max() > df[col].min():
            # Cap extremely high values at 99th percentile to handle outliers
            percentile_99 = df[col].quantile(0.99)
            df[col] = df[col].clip(upper=percentile_99)
            
            # Scale to 0-100 range
            df[col] = 100 * (df[col] - df[col].min()) / (df[col].max() - df[col].min())
    
    # high_value_return_affinity is already on a 0-100 scale
    
    # Log feature creation
    records_processed = len(df)
    warnings = feature_eng.validate_feature_business_logic('avg_order_value', df, 'avg_order_value', (0, 100))
    warnings += feature_eng.validate_feature_business_logic('avg_return_value', df, 'avg_return_value', (0, 100))
    warnings += feature_eng.validate_feature_business_logic('high_value_return_affinity', df, 'high_value_return_affinity', (0, 100))
    
    feature_eng.log_feature_creation('monetary_value_metrics', start_time, records_processed, warnings)
    
    logger.info(f"Created monetary value metrics for {len(df)} customers using actual GROSS values")
    return df

# =============================================================================
# FEATURE TESTING FUNCTIONS
# =============================================================================

def test_basic_volume_metrics(df: pd.DataFrame) -> Dict[str, bool]:
    """Unit tests for basic volume metrics"""
    tests = {}
    
    # Test data types and ranges
    tests['sales_order_no_nunique_positive'] = (df['sales_order_no_nunique'] > 0).all()
    tests['sku_nunique_positive'] = (df['sku_nunique'] > 0).all()
    tests['items_returned_count_non_negative'] = (df['items_returned_count'] >= 0).all()
    tests['sales_qty_mean_positive'] = (df['sales_qty_mean'] > 0).all()
    tests['avg_order_size_positive'] = (df['avg_order_size'] > 0).all()
    
    # Business logic tests
    tests['reasonable_order_sizes'] = (df['avg_order_size'] <= 50).all()
    tests['reasonable_qty_mean'] = (df['sales_qty_mean'] <= 100).all()
    
    return tests

def test_return_behavior_patterns(df: pd.DataFrame) -> Dict[str, bool]:
    """Unit tests for return behavior patterns"""
    tests = {}
    
    # Range tests for normalized features
    tests['return_rate_valid_range'] = ((df['return_rate'] >= 0) & (df['return_rate'] <= 1)).all()
    tests['return_ratio_valid_range'] = ((df['return_ratio'] >= 0) & (df['return_ratio'] <= 1)).all()
    tests['return_frequency_ratio_valid'] = ((df['return_frequency_ratio'] >= 0) & (df['return_frequency_ratio'] <= 1)).all()
    tests['return_intensity_valid'] = ((df['return_intensity'] >= 0) & (df['return_intensity'] <= 1)).all()
    
    # Non-negative tests
    tests['return_product_variety_non_negative'] = (df['return_product_variety'] >= 0).all()
    tests['avg_returns_per_order_non_negative'] = (df['avg_returns_per_order'] >= 0).all()
    
    return tests

def test_temporal_patterns(df: pd.DataFrame) -> Dict[str, bool]:
    """Unit tests for temporal patterns"""
    tests = {}
    
    # Non-negative tests
    tests['customer_lifetime_days_non_negative'] = (df['customer_lifetime_days'] >= 0).all()
    tests['avg_days_to_return_non_negative'] = (df['avg_days_to_return'] >= 0).all()
    tests['return_timing_spread_non_negative'] = (df['return_timing_spread'] >= 0).all()
    
    # Valid tenure stages
    valid_stages = ['New', 'Growing', 'Mature', 'Veteran']
    tests['valid_tenure_stages'] = df['customer_tenure_stage'].isin(valid_stages).all()
    
    return tests

def test_category_intelligence(df: pd.DataFrame) -> Dict[str, bool]:
    """Unit tests for category intelligence"""
    tests = {}
    
    # Range tests
    tests['category_loyalty_score_valid'] = ((df['category_loyalty_score'] >= 0) & (df['category_loyalty_score'] <= 1)).all()
    tests['high_return_category_affinity_valid'] = ((df['high_return_category_affinity'] >= 0) & (df['high_return_category_affinity'] <= 1)).all()
    tests['category_diversity_score_positive'] = (df['category_diversity_score'] >= 0).all()
    
    return tests

def test_adjacency_features(df: pd.DataFrame) -> Dict[str, bool]:
    """Unit tests for adjacency features"""
    tests = {}
    
    # Non-negative tests
    tests['sku_adjacency_orders_non_negative'] = (df['sku_adjacency_orders'] >= 0).all()
    tests['sku_adjacency_returns_non_negative'] = (df['sku_adjacency_returns'] >= 0).all()
    tests['sku_adjacency_timing_non_negative'] = (df['sku_adjacency_timing'] >= 0).all()
    tests['sku_adjacency_return_timing_non_negative'] = (df['sku_adjacency_return_timing'] >= 0).all()
    
    # Logical consistency
    tests['returns_not_exceed_orders'] = (df['sku_adjacency_returns'] <= df['sku_adjacency_orders']).all()
    
    return tests

def test_seasonal_features(df: pd.DataFrame) -> Dict[str, bool]:
    """Unit tests for seasonal features"""
    tests = {}
    
    # Non-negative tests
    tests['seasonal_susceptibility_orders_non_negative'] = (df['seasonal_susceptibility_orders'] >= 0).all()
    tests['seasonal_susceptibility_returns_non_negative'] = (df['seasonal_susceptibility_returns'] >= 0).all()
    
    return tests

def test_trend_susceptibility(df: pd.DataFrame) -> Dict[str, bool]:
    """Unit tests for trend susceptibility"""
    tests = {}
    
    # Correlation range tests
    tests['trend_order_rate_valid_range'] = ((df['trend_product_category_order_rate'] >= -1) & (df['trend_product_category_order_rate'] <= 1)).all()
    tests['trend_return_rate_valid_range'] = ((df['trend_product_category_return_rate'] >= -1) & (df['trend_product_category_return_rate'] <= 1)).all()
    
    return tests

def test_monetary_value_metrics(df: pd.DataFrame) -> Dict[str, bool]:
    """Unit tests for monetary value metrics"""
    tests = {}
    
    # Range tests for scaled metrics (0-100)
    tests['avg_order_value_range'] = ((df['avg_order_value'] >= 0) & (df['avg_order_value'] <= 100)).all()
    tests['avg_return_value_range'] = ((df['avg_return_value'] >= 0) & (df['avg_return_value'] <= 100)).all()
    tests['high_value_return_affinity_range'] = ((df['high_value_return_affinity'] >= 0) & (df['high_value_return_affinity'] <= 100)).all()
    
    # Nulls test
    tests['no_nulls_in_monetary_metrics'] = ~df[['avg_order_value', 'avg_return_value', 'high_value_return_affinity']].isnull().any().any()
    
    # Value comparison tests (business logic)
    # Some customers might have higher avg_return_value than avg_order_value if they
    # tend to return high-value items disproportionately, so this test should be removed
    # tests['avg_return_value_not_exceed_avg_order_value'] = (df['avg_return_value'] <= df['avg_order_value'] * 1.2).all()
    
    # Distribution check
    tests['reasonable_high_value_return_distribution'] = (df['high_value_return_affinity'].mean() <= 50)  # Most customers shouldn't return high-value items
    
    return tests

# =============================================================================
# BATCH PROCESSING FOR LARGE DATASETS
# =============================================================================

def process_features_in_batches(conn: duckdb.DuckDBPyConnection, batch_size: int = 50000) -> None:
    """Process feature creation in batches for large datasets"""
    
    # Get total customer count
    total_customers = conn.execute("SELECT count(DISTINCT customer_emailid) FROM bronze_return_order_data").fetchone()[0]
    logger.info(f"Processing features for {total_customers} customers in batches of {batch_size}")
    
    if total_customers <= batch_size:
        logger.info("Dataset small enough to process without batching")
        return False
    
    # Create customer batches
    conn.execute("""
        CREATE OR REPLACE TABLE customer_batches AS
        WITH customer_list AS (
            SELECT DISTINCT customer_emailid,
                   ROW_NUMBER() OVER (ORDER BY customer_emailid) as customer_row
            FROM bronze_return_order_data
        )
        SELECT customer_emailid,
               CAST(customer_row / ? AS INTEGER) as batch_id
        FROM customer_list;
    """, [batch_size])
    
    batch_count = conn.execute("SELECT max(batch_id) + 1 FROM customer_batches").fetchone()[0]
    logger.info(f"Created {batch_count} batches for processing")
    
    return True

# =============================================================================
# MAIN FEATURE CREATION ORCHESTRATOR
# =============================================================================

def create_all_features(conn: duckdb.DuckDBPyConnection, run_tests: bool = True) -> Dict[str, pd.DataFrame]:
    """Create all customer features and optionally run tests"""
    
    logger.info("Starting comprehensive feature creation...")
    start_time = time.time()
    
    features = {}
    test_results = {}
    
    # Check if batching is needed
    needs_batching = process_features_in_batches(conn)
    if needs_batching:
        logger.warning("Large dataset detected. Consider implementing batch processing for complex features.")
    
    try:
        # Create features in order of complexity (simple to complex)
        logger.info("Creating basic volume metrics...")
        features['basic_volume'] = create_basic_volume_metrics(conn)
        if run_tests:
            test_results['basic_volume'] = test_basic_volume_metrics(features['basic_volume'])
        
        logger.info("Creating return behavior patterns...")
        features['return_behavior'] = create_return_behavior_patterns(conn)
        if run_tests:
            test_results['return_behavior'] = test_return_behavior_patterns(features['return_behavior'])
        
        logger.info("Creating temporal patterns...")
        features['temporal'] = create_temporal_patterns(conn)
        if run_tests:
            test_results['temporal'] = test_temporal_patterns(features['temporal'])
        
        logger.info("Creating recency analysis...")
        features['recency'] = create_recency_analysis(conn)
        
        logger.info("Creating category intelligence...")
        features['category'] = create_category_intelligence(conn)
        if run_tests:
            test_results['category'] = test_category_intelligence(features['category'])
        
        logger.info("Creating consecutive returns...")
        features['consecutive'] = create_consecutive_returns(conn)
        
        logger.info("Creating adjacency features...")
        features['adjacency'] = create_adjacency_features(conn)
        if run_tests:
            test_results['adjacency'] = test_adjacency_features(features['adjacency'])
        
        logger.info("Creating seasonal features...")
        features['seasonal'] = create_seasonal_features(conn)
        if run_tests:
            test_results['seasonal'] = test_seasonal_features(features['seasonal'])
        
        logger.info("Creating trend susceptibility...")
        features['trend'] = create_trend_susceptibility(conn)
        if run_tests:
            test_results['trend'] = test_trend_susceptibility(features['trend'])
        
        logger.info("Creating monetary value metrics...")
        features['monetary'] = create_monetary_value_metrics(conn)
        if run_tests:
            test_results['monetary'] = test_monetary_value_metrics(features['monetary'])
        
        total_time = time.time() - start_time
        logger.info(f"All features created successfully in {total_time:.2f} seconds")
        
        # Print test results summary
        if run_tests:
            print("\n" + "="*50)
            print("FEATURE VALIDATION SUMMARY")
            print("="*50)
            
            for feature_group, tests in test_results.items():
                failed_tests = [test for test, result in tests.items() if not result]
                if failed_tests:
                    print(f"❌ {feature_group}: {len(failed_tests)} tests FAILED")
                    for test in failed_tests:
                        print(f"   - {test}")
                else:
                    print(f"✅ {feature_group}: All tests PASSED")
        
        return features
        
    except Exception as e:
        logger.error(f"Feature creation failed: {str(e)}")
        raise

if __name__ == "__main__":
    # Example usage
    from db import get_connection
    
    conn = get_connection("customer_clustering.db")
    features = create_all_features(conn, run_tests=True)
    
    print(f"\nFeature creation complete. Created {len(features)} feature groups:")
    for name, df in features.items():
        print(f"  {name}: {len(df)} customers, {len(df.columns)} features")
    
    conn.close()