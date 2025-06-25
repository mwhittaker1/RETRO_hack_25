# Customer Return Data Analysis
# Comprehensive analysis to inform feature engineering and clustering

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import duckdb
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Setup
plt.style.use('default')
sns.set_palette("husl")

from db import get_connection

# %%
# Load and connect to database
conn = get_connection("customer_clustering.db")

# Get basic data overview
print("=== BASIC DATA OVERVIEW ===")
overview = conn.execute("""
    SELECT 
        count(*) as total_records,
        count(DISTINCT customer_emailid) as unique_customers,
        count(DISTINCT sales_order_no) as unique_orders,
        count(DISTINCT q_sku_id) as unique_skus,
        count(DISTINCT q_cls_id) as unique_classes,
        min(order_date) as earliest_order,
        max(order_date) as latest_order,
        sum(sales_qty) as total_sales_qty,
        sum(return_qty) as total_return_qty,
        avg(CAST(sales_qty AS DOUBLE)) as avg_sales_qty,
        avg(CAST(return_qty AS DOUBLE)) as avg_return_qty
    FROM bronze_return_order_data;
""").fetchdf()

print(overview.to_string())

# %%
# Customer-level aggregation for analysis
print("\n=== CUSTOMER-LEVEL ANALYSIS ===")

customer_stats = conn.execute("""
    WITH customer_summary AS (
        SELECT 
            customer_emailid,
            count(DISTINCT sales_order_no) as order_count,
            count(*) as item_count,
            count(DISTINCT q_sku_id) as unique_skus,
            count(DISTINCT q_cls_id) as unique_classes,
            sum(sales_qty) as total_sales_qty,
            sum(return_qty) as total_return_qty,
            sum(CASE WHEN return_qty > 0 THEN 1 ELSE 0 END) as items_returned_count,
            min(order_date) as first_order,
            max(order_date) as last_order,
            max(order_date) - min(order_date) as customer_lifetime_days
        FROM bronze_return_order_data
        GROUP BY customer_emailid
    ),
    with_rates AS (
        SELECT *,
            CAST(total_return_qty AS DOUBLE) / CAST(total_sales_qty AS DOUBLE) as return_rate,
            CAST(items_returned_count AS DOUBLE) / CAST(item_count AS DOUBLE) as item_return_rate,
            CAST(item_count AS DOUBLE) / CAST(order_count AS DOUBLE) as avg_order_size
        FROM customer_summary
    )
    SELECT 
        count(*) as customer_count,
        avg(order_count) as avg_orders_per_customer,
        percentile_cont(0.5) WITHIN GROUP (ORDER BY order_count) as median_orders,
        avg(total_sales_qty) as avg_total_qty_purchased,
        avg(return_rate) as avg_return_rate,
        percentile_cont(0.5) WITHIN GROUP (ORDER BY return_rate) as median_return_rate,
        avg(customer_lifetime_days) as avg_customer_lifetime_days,
        avg(avg_order_size) as avg_order_size,
        max(order_count) as max_orders,
        max(total_sales_qty) as max_qty_purchased,
        max(return_rate) as max_return_rate,
        max(avg_order_size) as max_order_size
    FROM with_rates;
""").fetchdf()

print(customer_stats.to_string())

# %%
# Distribution analysis for business logic thresholds
print("\n=== DISTRIBUTION ANALYSIS FOR THRESHOLDS ===")

distributions = conn.execute("""
    WITH customer_metrics AS (
        SELECT 
            customer_emailid,
            count(DISTINCT sales_order_no) as order_count,
            count(*) as item_count,
            sum(sales_qty) as total_sales_qty,
            sum(return_qty) as total_return_qty,
            avg(CAST(sales_qty AS DOUBLE)) as sales_qty_mean,
            CAST(count(*) AS DOUBLE) / CAST(count(DISTINCT sales_order_no) AS DOUBLE) as avg_order_size,
            CAST(sum(return_qty) AS DOUBLE) / CAST(sum(sales_qty) AS DOUBLE) as return_rate,
            count(DISTINCT q_sku_id) as product_variety
        FROM bronze_return_order_data
        GROUP BY customer_emailid
    )
    SELECT 
        'sales_qty_mean' as metric,
        avg(sales_qty_mean) as mean_val,
        stddev(sales_qty_mean) as std_val,
        min(sales_qty_mean) as min_val,
        percentile_cont(0.25) WITHIN GROUP (ORDER BY sales_qty_mean) as q25,
        percentile_cont(0.5) WITHIN GROUP (ORDER BY sales_qty_mean) as median,
        percentile_cont(0.75) WITHIN GROUP (ORDER BY sales_qty_mean) as q75,
        percentile_cont(0.95) WITHIN GROUP (ORDER BY sales_qty_mean) as p95,
        percentile_cont(0.99) WITHIN GROUP (ORDER BY sales_qty_mean) as p99,
        max(sales_qty_mean) as max_val
    FROM customer_metrics
    
    UNION ALL
    
    SELECT 
        'avg_order_size' as metric,
        avg(avg_order_size) as mean_val,
        stddev(avg_order_size) as std_val,
        min(avg_order_size) as min_val,
        percentile_cont(0.25) WITHIN GROUP (ORDER BY avg_order_size) as q25,
        percentile_cont(0.5) WITHIN GROUP (ORDER BY avg_order_size) as median,
        percentile_cont(0.75) WITHIN GROUP (ORDER BY avg_order_size) as q75,
        percentile_cont(0.95) WITHIN GROUP (ORDER BY avg_order_size) as p95,
        percentile_cont(0.99) WITHIN GROUP (ORDER BY avg_order_size) as p99,
        max(avg_order_size) as max_val
    FROM customer_metrics
    
    UNION ALL
    
    SELECT 
        'return_rate' as metric,
        avg(return_rate) as mean_val,
        stddev(return_rate) as std_val,
        min(return_rate) as min_val,
        percentile_cont(0.25) WITHIN GROUP (ORDER BY return_rate) as q25,
        percentile_cont(0.5) WITHIN GROUP (ORDER BY return_rate) as median,
        percentile_cont(0.75) WITHIN GROUP (ORDER BY return_rate) as q75,
        percentile_cont(0.95) WITHIN GROUP (ORDER BY return_rate) as p95,
        percentile_cont(0.99) WITHIN GROUP (ORDER BY return_rate) as p99,
        max(return_rate) as max_val
    FROM customer_metrics
    
    UNION ALL
    
    SELECT 
        'product_variety' as metric,
        avg(CAST(product_variety AS DOUBLE)) as mean_val,
        stddev(CAST(product_variety AS DOUBLE)) as std_val,
        min(CAST(product_variety AS DOUBLE)) as min_val,
        percentile_cont(0.25) WITHIN GROUP (ORDER BY product_variety) as q25,
        percentile_cont(0.5) WITHIN GROUP (ORDER BY product_variety) as median,
        percentile_cont(0.75) WITHIN GROUP (ORDER BY product_variety) as q75,
        percentile_cont(0.95) WITHIN GROUP (ORDER BY product_variety) as p95,
        percentile_cont(0.99) WITHIN GROUP (ORDER BY product_variety) as p99,
        max(CAST(product_variety AS DOUBLE)) as max_val
    FROM customer_metrics;
""").fetchdf()

print(distributions.to_string())

# %%
# Return timing analysis
print("\n=== RETURN TIMING ANALYSIS ===")

return_timing = conn.execute("""
    WITH return_timing AS (
        SELECT 
            customer_emailid,
            sales_order_no,
            return_date - order_date as days_to_return
        FROM bronze_return_order_data
        WHERE return_qty > 0 AND return_date IS NOT NULL
    ),
    customer_timing AS (
        SELECT 
            customer_emailid,
            avg(CAST(days_to_return AS DOUBLE)) as avg_days_to_return,
            min(days_to_return) as min_days_to_return,
            max(days_to_return) as max_days_to_return,
            stddev(CAST(days_to_return AS DOUBLE)) as return_timing_spread,
            count(*) as return_count
        FROM return_timing
        GROUP BY customer_emailid
    )
    SELECT 
        count(*) as customers_with_returns,
        avg(avg_days_to_return) as overall_avg_days_to_return,
        percentile_cont(0.5) WITHIN GROUP (ORDER BY avg_days_to_return) as median_avg_days_to_return,
        percentile_cont(0.95) WITHIN GROUP (ORDER BY avg_days_to_return) as p95_avg_days_to_return,
        max(avg_days_to_return) as max_avg_days_to_return,
        avg(return_timing_spread) as avg_timing_spread,
        percentile_cont(0.95) WITHIN GROUP (ORDER BY return_timing_spread) as p95_timing_spread,
        sum(CASE WHEN avg_days_to_return > 60 THEN 1 ELSE 0 END) as customers_over_60_days,
        sum(CASE WHEN avg_days_to_return < 0 THEN 1 ELSE 0 END) as customers_negative_days
    FROM customer_timing;
""").fetchdf()

print(return_timing.to_string())

# %%
# Data quality analysis
print("\n=== DATA QUALITY ANALYSIS ===")

data_quality = conn.execute("""
    SELECT 
        'Total records' as check_type,
        count(*) as count,
        '' as percentage
    FROM bronze_return_order_data
    
    UNION ALL
    
    SELECT 
        'Records with returns' as check_type,
        count(*) as count,
        ROUND(100.0 * count(*) / (SELECT count(*) FROM bronze_return_order_data), 2) || '%' as percentage
    FROM bronze_return_order_data
    WHERE return_qty > 0
    
    UNION ALL
    
    SELECT 
        'Missing return comments' as check_type,
        count(*) as count,
        ROUND(100.0 * count(*) / (SELECT count(*) FROM bronze_return_order_data WHERE return_qty > 0), 2) || '%' as percentage
    FROM bronze_return_order_data
    WHERE return_qty > 0 AND (return_comment IS NULL OR trim(return_comment) = '')
    
    UNION ALL
    
    SELECT 
        'Return date issues fixed' as check_type,
        count(*) as count,
        ROUND(100.0 * count(*) / (SELECT count(*) FROM bronze_return_order_data), 2) || '%' as percentage
    FROM bronze_return_order_data
    WHERE data_quality_flags LIKE '%RETURN_DATE%'
    
    UNION ALL
    
    SELECT 
        'Duplicate emails (case)' as check_type,
        count(DISTINCT customer_emailid) - count(DISTINCT lower(customer_emailid)) as count,
        '' as percentage
    FROM bronze_return_order_data;
""").fetchdf()

print(data_quality.to_string())

# %%
# Customer tenure analysis for temporal features
print("\n=== CUSTOMER TENURE ANALYSIS ===")

tenure_analysis = conn.execute("""
    WITH customer_tenure AS (
        SELECT 
            customer_emailid,
            min(order_date) as first_order,
            max(order_date) as last_order,
            max(order_date) - min(order_date) as lifetime_days,
            count(DISTINCT sales_order_no) as order_count
        FROM bronze_return_order_data
        GROUP BY customer_emailid
    ),
    tenure_categories AS (
        SELECT *,
            CASE 
                WHEN lifetime_days <= 90 THEN 'New (≤90 days)'
                WHEN lifetime_days <= 180 THEN 'Growing (91-180 days)'
                WHEN lifetime_days <= 365 THEN 'Mature (181-365 days)'
                WHEN lifetime_days <= 730 THEN 'Established (1-2 years)'
                ELSE 'Veteran (>2 years)'
            END as tenure_stage
        FROM customer_tenure
    )
    SELECT 
        tenure_stage,
        count(*) as customer_count,
        ROUND(100.0 * count(*) / sum(count(*)) OVER(), 2) as percentage,
        avg(CAST(lifetime_days AS DOUBLE)) as avg_lifetime_days,
        avg(CAST(order_count AS DOUBLE)) as avg_orders
    FROM tenure_categories
    GROUP BY tenure_stage
    ORDER BY avg_lifetime_days;
""").fetchdf()

print(tenure_analysis.to_string())

# %%
# Consecutive returns analysis
print("\n=== CONSECUTIVE RETURNS ANALYSIS ===")

consecutive_analysis = conn.execute("""
    WITH order_returns AS (
        SELECT 
            customer_emailid,
            sales_order_no,
            order_date,
            sum(return_qty) as order_return_qty,
            sum(sales_qty) as order_sales_qty,
            CASE WHEN sum(return_qty) > 0 THEN 1 ELSE 0 END as has_returns
        FROM bronze_return_order_data
        GROUP BY customer_emailid, sales_order_no, order_date
    ),
    ordered_returns AS (
        SELECT *,
            ROW_NUMBER() OVER (PARTITION BY customer_emailid ORDER BY order_date) as order_sequence,
            LAG(has_returns) OVER (PARTITION BY customer_emailid ORDER BY order_date) as prev_order_returned
        FROM order_returns
    ),
    consecutive_groups AS (
        SELECT *,
            SUM(CASE WHEN has_returns = 1 AND (prev_order_returned = 0 OR prev_order_returned IS NULL) THEN 1 ELSE 0 END) 
                OVER (PARTITION BY customer_emailid ORDER BY order_date) as consecutive_group
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
            count(*) as consecutive_return_events,
            max(consecutive_length) as max_consecutive_returns,
            avg(CAST(consecutive_length AS DOUBLE)) as avg_consecutive_returns
        FROM consecutive_counts
        GROUP BY customer_emailid
    )
    SELECT 
        count(*) as customers_with_consecutive_returns,
        avg(max_consecutive_returns) as avg_max_consecutive,
        percentile_cont(0.95) WITHIN GROUP (ORDER BY max_consecutive_returns) as p95_max_consecutive,
        max(max_consecutive_returns) as overall_max_consecutive,
        avg(avg_consecutive_returns) as avg_avg_consecutive,
        sum(CASE WHEN max_consecutive_returns > 10 THEN 1 ELSE 0 END) as customers_over_10_consecutive
    FROM customer_consecutive_stats;
""").fetchdf()

print(consecutive_analysis.to_string())

# %%
# Category and product analysis
print("\n=== CATEGORY AND PRODUCT ANALYSIS ===")

category_analysis = conn.execute("""
    WITH customer_categories AS (
        SELECT 
            customer_emailid,
            count(DISTINCT q_cls_id) as unique_categories,
            count(DISTINCT q_sku_id) as unique_products,
            count(*) as total_items
        FROM bronze_return_order_data
        GROUP BY customer_emailid
    )
    SELECT 
        avg(CAST(unique_categories AS DOUBLE)) as avg_categories_per_customer,
        percentile_cont(0.95) WITHIN GROUP (ORDER BY unique_categories) as p95_categories,
        max(unique_categories) as max_categories,
        avg(CAST(unique_products AS DOUBLE)) as avg_products_per_customer,
        percentile_cont(0.95) WITHIN GROUP (ORDER BY unique_products) as p95_products,
        max(unique_products) as max_products,
        sum(CASE WHEN unique_categories > 50 THEN 1 ELSE 0 END) as customers_over_50_categories,
        sum(CASE WHEN unique_products > 50 THEN 1 ELSE 0 END) as customers_over_50_products
    FROM customer_categories;
""").fetchdf()

print(category_analysis.to_string())

# %%
# Seasonal analysis (basic)
print("\n=== SEASONAL ANALYSIS ===")

seasonal_analysis = conn.execute("""
    WITH seasonal_data AS (
        SELECT 
            customer_emailid,
            CASE 
                WHEN EXTRACT(month FROM order_date) IN (10, 11, 12) THEN 'Fall (Oct-Dec)'
                WHEN EXTRACT(month FROM order_date) IN (1, 2, 3) THEN 'Winter (Jan-Mar)'
                WHEN EXTRACT(month FROM order_date) IN (4, 5, 6) THEN 'Spring (Apr-Jun)'
                ELSE 'Summer (Jul-Sep)'
            END as season,
            count(*) as items_ordered,
            sum(return_qty) as items_returned
        FROM bronze_return_order_data
        GROUP BY customer_emailid, 
            CASE 
                WHEN EXTRACT(month FROM order_date) IN (10, 11, 12) THEN 'Fall (Oct-Dec)'
                WHEN EXTRACT(month FROM order_date) IN (1, 2, 3) THEN 'Winter (Jan-Mar)'
                WHEN EXTRACT(month FROM order_date) IN (4, 5, 6) THEN 'Spring (Apr-Jun)'
                ELSE 'Summer (Jul-Sep)'
            END
    )
    SELECT 
        season,
        count(DISTINCT customer_emailid) as active_customers,
        sum(items_ordered) as total_items_ordered,
        sum(items_returned) as total_items_returned,
        ROUND(100.0 * sum(items_returned) / sum(items_ordered), 2) as return_rate_pct
    FROM seasonal_data
    GROUP BY season
    ORDER BY 
        CASE season
            WHEN 'Fall (Oct-Dec)' THEN 1
            WHEN 'Winter (Jan-Mar)' THEN 2
            WHEN 'Spring (Apr-Jun)' THEN 3
            ELSE 4
        END;
""").fetchdf()

print(seasonal_analysis.to_string())

# %%
# RECOMMENDED BUSINESS LOGIC THRESHOLDS
print("\n" + "="*50)
print("RECOMMENDED BUSINESS LOGIC THRESHOLDS")
print("="*50)

recommendations = """
Based on the analysis above, here are the recommended thresholds for feature validation:

VOLUME METRICS:
- sales_qty_mean > 15 (warning if above P95)
- avg_order_size > 10 (warning if above P95)  
- return_product_variety > 25 (warning if above P95)

RETURN BEHAVIOR:
- return_rate > 0.95 or < 0 (error - should be 0-1)
- return_ratio > 1.0 or < 0 (error - should be 0-1)
- return_frequency_ratio > 1.0 or < 0 (error - should be 0-1)
- return_intensity > 1.0 or < 0 (error - should be 0-1)
- avg_returns_per_order > 8 (warning if above P95)

TEMPORAL PATTERNS:
- avg_days_to_return > 60 (warning - outside return window)
- avg_days_to_return < 0 (error - impossible)
- return_timing_spread > 45 (warning if above P95)
- consecutive_returns > 6 (warning if above P95)
- avg_consecutive_returns > 3 (warning if above P95)

CATEGORY PATTERNS:
- category_diversity_score > 30 (warning if above P95)
- category_loyalty_score should be 0-1 (error if outside)

Note: Actual P95 values should be calculated from your full dataset.
The analysis shows most customers have reasonable behavior patterns,
with clear outliers that should be flagged for review.
"""

print(recommendations)

# %%
# Email consolidation candidates
print("\n=== EMAIL CONSOLIDATION CANDIDATES ===")

# Simple case-based grouping since Levenshtein might not be available
email_variants = conn.execute("""
    SELECT 
        lower(customer_emailid) as standardized_email,
        list(DISTINCT customer_emailid ORDER BY customer_emailid) as email_variants,
        count(DISTINCT customer_emailid) as variant_count,
        count(*) as total_records
    FROM bronze_return_order_data
    GROUP BY lower(customer_emailid)
    HAVING count(DISTINCT customer_emailid) > 1
    ORDER BY variant_count DESC, total_records DESC
    LIMIT 20;
""").fetchdf()

print("Top email consolidation candidates:")
print(email_variants.to_string())

# %%
# Final summary for feature engineering
print("\n=== SUMMARY FOR FEATURE ENGINEERING ===")

summary_stats = conn.execute("""
    WITH customer_summary AS (
        SELECT 
            customer_emailid,
            count(DISTINCT sales_order_no) as orders,
            count(*) as items,
            sum(sales_qty) as total_qty,
            sum(return_qty) as returned_qty,
            min(order_date) as first_order,
            max(order_date) as last_order
        FROM bronze_return_order_data
        GROUP BY customer_emailid
    )
    SELECT 
        count(*) as total_customers,
        sum(CASE WHEN (last_order - first_order) > 730 THEN 1 ELSE 0 END) as customers_over_2_years,
        ROUND(100.0 * sum(CASE WHEN (last_order - first_order) > 730 THEN 1 ELSE 0 END) / count(*), 2) as pct_over_2_years,
        sum(CASE WHEN orders >= 50 THEN 1 ELSE 0 END) as customers_50plus_orders,
        ROUND(100.0 * sum(CASE WHEN orders >= 50 THEN 1 ELSE 0 END) / count(*), 2) as pct_50plus_orders
    FROM customer_summary;
""").fetchdf()

print("Customer eligibility for advanced features:")
print(summary_stats.to_string())

conn.close()
print("\nAnalysis complete! Use these insights to inform feature engineering thresholds.")
