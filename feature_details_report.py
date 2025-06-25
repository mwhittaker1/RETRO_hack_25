"""
Feature Details Report Generator
Creates a comprehensive Excel report with individual feature details and logic
"""

import pandas as pd
import numpy as np
from datetime import datetime
import logging
import sys
import os

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Define the features with detailed descriptions, logic, and business value
feature_metadata = {
    # BASIC VOLUME METRICS
    'sales_order_no_nunique': {
        'category': 'Volume Metrics',
        'subcategory': 'Purchase Activity',
        'description': 'Count of unique sales orders per customer',
        'logic': 'COUNT(DISTINCT sales_order_no) for each customer',
        'sql_implementation': 'count(DISTINCT sales_order_no) as sales_order_no_nunique',
        'business_value': 'Measures purchase frequency and customer engagement. Higher values indicate more active customers.',
        'clustering_value': 'Differentiates between high-frequency shoppers and occasional customers.',
        'expected_range': 'Integer values >= 0'
    },
    'sku_nunique': {
        'category': 'Volume Metrics',
        'subcategory': 'Product Diversity',
        'description': 'Count of unique SKUs purchased by customer',
        'logic': 'COUNT(DISTINCT q_sku_id) for each customer',
        'sql_implementation': 'count(DISTINCT q_sku_id) as sku_nunique',
        'business_value': 'Measures product diversity in customer purchases. Higher values indicate broader product interest.',
        'clustering_value': 'Helps identify customers with diverse vs. targeted purchasing patterns.',
        'expected_range': 'Integer values >= 0'
    },
    'items_returned_count': {
        'category': 'Volume Metrics',
        'subcategory': 'Return Volume',
        'description': 'Count of items returned by customer',
        'logic': 'Sum of items with returns (return_qty > 0)',
        'sql_implementation': 'sum(CASE WHEN return_qty > 0 THEN 1 ELSE 0 END) as items_returned_count',
        'business_value': 'Measures absolute return volume. Higher values may indicate fit issues or customer dissatisfaction.',
        'clustering_value': 'Raw count helps identify customers with high return volumes regardless of purchase volume.',
        'expected_range': 'Integer values >= 0'
    },
    'sales_qty_mean': {
        'category': 'Volume Metrics',
        'subcategory': 'Purchase Size',
        'description': 'Average quantity per line item',
        'logic': 'Average sales quantity across all orders',
        'sql_implementation': 'avg(CAST(sales_qty AS DOUBLE)) as sales_qty_mean',
        'business_value': 'Measures typical purchase quantity behavior. Higher values may indicate bulk purchasing.',
        'clustering_value': 'Helps distinguish between single-item purchasers and bulk buyers.',
        'expected_range': 'Float values > 0, with validation threshold at 15'
    },
    'avg_order_size': {
        'category': 'Volume Metrics',
        'subcategory': 'Basket Size',
        'description': 'Average number of items per order',
        'logic': 'Total item count divided by number of orders',
        'sql_implementation': 'CAST(count(*) AS DOUBLE) / CAST(count(DISTINCT sales_order_no) AS DOUBLE) as avg_order_size',
        'business_value': 'Measures shopping basket size. Higher values indicate customers who purchase multiple items per order.',
        'clustering_value': 'Differentiates between customers who make single-item purchases vs. multiple-item purchases.',
        'expected_range': 'Float values > 0, with validation threshold at 10'
    },
    
    # RETURN BEHAVIOR PATTERNS
    'return_rate': {
        'category': 'Return Behavior',
        'subcategory': 'Return Propensity',
        'description': 'Ratio of returned quantity to total sales quantity',
        'logic': 'Total returned quantity divided by total sales quantity',
        'sql_implementation': 'CAST(total_return_qty AS DOUBLE) / NULLIF(CAST(total_sales_qty AS DOUBLE), 0) as return_rate',
        'business_value': 'Primary indicator of customer return behavior. Higher values indicate customers who return a large portion of purchases.',
        'clustering_value': 'Essential metric for identifying high-return-risk customer segments.',
        'expected_range': 'Float values between 0 and 1'
    },
    'return_ratio': {
        'category': 'Return Behavior',
        'subcategory': 'Return Propensity',
        'description': 'Ratio of returned quantity to total sales quantity (same as return_rate)',
        'logic': 'Same as return_rate (redundant feature)',
        'sql_implementation': 'CAST(total_return_qty AS DOUBLE) / NULLIF(CAST(total_sales_qty AS DOUBLE), 0) as return_ratio',
        'business_value': 'Redundant to return_rate - measures proportion of purchases that are returned.',
        'clustering_value': 'May be removed due to redundancy with return_rate.',
        'expected_range': 'Float values between 0 and 1'
    },
    'return_product_variety': {
        'category': 'Return Behavior',
        'subcategory': 'Return Diversity',
        'description': 'Count of unique product SKUs that were returned',
        'logic': 'Count of distinct SKUs with return quantity > 0',
        'sql_implementation': 'count(DISTINCT q_sku_id) FILTER (WHERE return_qty > 0) as return_product_variety',
        'business_value': 'Measures diversity of returned products. High values indicate returns across many product types rather than issues with specific items.',
        'clustering_value': 'Helps identify if customer return behavior is product-specific or broadly distributed.',
        'expected_range': 'Integer values >= 0, with validation threshold at 25'
    },
    'avg_returns_per_order': {
        'category': 'Return Behavior',
        'subcategory': 'Return Intensity',
        'description': 'Average number of returned items per order',
        'logic': 'Count of items with returns divided by total orders',
        'sql_implementation': 'CAST(items_with_returns AS DOUBLE) / NULLIF(CAST(total_orders AS DOUBLE), 0) as avg_returns_per_order',
        'business_value': 'Measures how many items in a typical order are returned. Higher values indicate customers who return multiple items from each order.',
        'clustering_value': 'Differentiates between customers who return single items vs. those who return whole orders.',
        'expected_range': 'Float values >= 0, with validation threshold at 8'
    },
    'return_frequency_ratio': {
        'category': 'Return Behavior',
        'subcategory': 'Return Frequency',
        'description': 'Ratio of items with returns to total items purchased',
        'logic': 'Count of items with returns divided by total items',
        'sql_implementation': 'CAST(items_with_returns AS DOUBLE) / NULLIF(CAST(total_items AS DOUBLE), 0) as return_frequency_ratio',
        'business_value': 'Measures how frequently a customer returns items. Similar to return rate but on item count rather than quantity.',
        'clustering_value': 'Alternative measure of return propensity that focuses on item count rather than quantity.',
        'expected_range': 'Float values between 0 and 1'
    },
    'return_intensity': {
        'category': 'Return Behavior',
        'subcategory': 'Return Volume',
        'description': 'Average quantity returned per item with returns',
        'logic': 'Total return quantity divided by count of items with returns',
        'sql_implementation': 'CASE WHEN items_with_returns > 0 THEN CAST(total_return_qty AS DOUBLE) / NULLIF(CAST(items_with_returns AS DOUBLE), 0) ELSE 0 END as return_intensity',
        'business_value': 'Measures if customers tend to return entire quantities of an item or partial quantities. Higher values indicate full returns rather than partial.',
        'clustering_value': 'Helps identify customers who tend to return entire quantities vs. partial quantities.',
        'expected_range': 'Float values between 0 and 1'
    },
    
    # CONSECUTIVE RETURNS
    'consecutive_returns': {
        'category': 'Return Behavior',
        'subcategory': 'Return Patterns',
        'description': 'Count of instances where customer had consecutive orders with returns',
        'logic': 'Complex calculation tracking order sequence and identifying consecutive return patterns',
        'sql_implementation': '''
        WITH order_returns AS (
            SELECT customer_emailid, sales_order_no, order_date,
                CASE WHEN sum(return_qty) > 0 THEN 1 ELSE 0 END as has_returns
            FROM bronze_return_order_data
            GROUP BY customer_emailid, sales_order_no, order_date
        ),
        ordered_returns AS (
            SELECT *, ROW_NUMBER() OVER (PARTITION BY customer_emailid ORDER BY order_date) as order_sequence,
                LAG(has_returns) OVER (PARTITION BY customer_emailid ORDER BY order_date) as prev_order_returned
            FROM order_returns ORDER BY customer_emailid, order_date
        ),
        consecutive_groups AS (
            SELECT *, SUM(CASE WHEN has_returns = 1 AND (prev_order_returned = 0 OR prev_order_returned IS NULL) THEN 1 ELSE 0 END) 
                OVER (PARTITION BY customer_emailid ORDER BY order_date ROWS UNBOUNDED PRECEDING) as consecutive_group
            FROM ordered_returns WHERE has_returns = 1
        ),
        consecutive_counts AS (
            SELECT customer_emailid, consecutive_group, count(*) as consecutive_length
            FROM consecutive_groups GROUP BY customer_emailid, consecutive_group
        ),
        customer_consecutive_stats AS (
            SELECT customer_emailid, count(*) as consecutive_returns, avg(CAST(consecutive_length AS DOUBLE)) as avg_consecutive_returns
            FROM consecutive_counts GROUP BY customer_emailid
        )
        ''',
        'business_value': 'Identifies habitual returners who have patterns of returning items order after order. Higher values indicate more concerning return patterns.',
        'clustering_value': 'Strong indicator of customers with problematic return behavior that persists across multiple orders.',
        'expected_range': 'Integer values >= 0, with validation threshold at 6'
    },
    'avg_consecutive_returns': {
        'category': 'Return Behavior',
        'subcategory': 'Return Patterns',
        'description': 'Average length of consecutive return sequences',
        'logic': 'For each sequence of consecutive returns, calculates the average length',
        'sql_implementation': 'Part of the consecutive returns calculation above',
        'business_value': 'Measures the typical duration of return patterns. Higher values indicate longer streaks of orders with returns.',
        'clustering_value': 'Helps distinguish between occasional returners and those with extended return patterns.',
        'expected_range': 'Float values >= 0, with validation threshold at 3'
    },
    
    # TEMPORAL & TIMING PATTERNS
    'customer_lifetime_days': {
        'category': 'Temporal Patterns',
        'subcategory': 'Customer Tenure',
        'description': 'Number of days between first and last order',
        'logic': 'Date difference between earliest and latest order dates',
        'sql_implementation': 'DATE_DIFF(\'day\', min(order_date), max(order_date)) as customer_lifetime_days',
        'business_value': 'Measures customer relationship duration. Higher values indicate longer-term customers.',
        'clustering_value': 'Helps distinguish between new customers, growing customers, and long-term customers.',
        'expected_range': 'Integer values >= 0'
    },
    'avg_days_to_return': {
        'category': 'Temporal Patterns',
        'subcategory': 'Return Timing',
        'description': 'Average number of days between order and return',
        'logic': 'Average date difference between order date and return date for all returns',
        'sql_implementation': 'avg(DATE_DIFF(\'day\', order_date, return_date)) as avg_days_to_return',
        'business_value': 'Measures how quickly customers typically return items. Lower values may indicate fit/satisfaction issues; higher values may indicate policy abuse.',
        'clustering_value': 'Helps identify rapid returners vs. delayed returners, which may have different business implications.',
        'expected_range': 'Float values >= 0, with validation threshold at 60'
    },
    'return_timing_spread': {
        'category': 'Temporal Patterns',
        'subcategory': 'Return Timing',
        'description': 'Standard deviation of days between order and return',
        'logic': 'Standard deviation of date difference between order date and return date',
        'sql_implementation': 'stddev(DATE_DIFF(\'day\', order_date, return_date)) as return_timing_spread',
        'business_value': 'Measures consistency in return timing. Lower values indicate customers with very predictable return patterns.',
        'clustering_value': 'Helps identify customers with variable return timing vs. those with consistent patterns.',
        'expected_range': 'Float values >= 0, with validation threshold at 45'
    },
    'customer_tenure_stage': {
        'category': 'Temporal Patterns',
        'subcategory': 'Customer Tenure',
        'description': 'Categorization of customer lifetime',
        'logic': 'Categorizes customers based on lifetime days into stages',
        'sql_implementation': '''
        CASE 
            WHEN customer_lifetime_days <= 90 THEN 'New'
            WHEN customer_lifetime_days <= 180 THEN 'Growing'
            WHEN customer_lifetime_days <= 365 THEN 'Mature'
            ELSE 'Veteran'
        END as customer_tenure_stage
        ''',
        'business_value': 'Provides a simple segmentation of customer relationship duration.',
        'clustering_value': 'Helps analyze if return patterns differ across customer tenure stages.',
        'expected_range': 'Categorical: New, Growing, Mature, Veteran'
    },
    
    # RECENCY ANALYSIS
    'recent_orders': {
        'category': 'Recency Analysis',
        'subcategory': 'Recent Activity',
        'description': 'Count of orders in the last 90 days',
        'logic': 'Count of distinct orders within 90 days of latest order date',
        'sql_implementation': 'count(DISTINCT sales_order_no) FILTER (WHERE order_date >= cutoff_date) as recent_orders',
        'business_value': 'Measures recent purchase activity. Higher values indicate more active customers.',
        'clustering_value': 'Helps identify currently active customers vs. lapsed customers.',
        'expected_range': 'Integer values >= 0'
    },
    'recent_returns': {
        'category': 'Recency Analysis',
        'subcategory': 'Recent Activity',
        'description': 'Count of returns in the last 90 days',
        'logic': 'Count of items with returns within 90 days of latest order date',
        'sql_implementation': 'sum(CASE WHEN return_qty > 0 THEN 1 ELSE 0 END) FILTER (WHERE order_date >= cutoff_date) as recent_returns',
        'business_value': 'Measures recent return activity. Higher values indicate currently problematic customers.',
        'clustering_value': 'Critical for identifying customers with current return issues rather than historical patterns.',
        'expected_range': 'Integer values >= 0'
    },
    'recent_vs_avg_ratio': {
        'category': 'Recency Analysis',
        'subcategory': 'Behavior Change',
        'description': 'Ratio of recent return rate to historical return rate',
        'logic': 'Recent returns per day divided by historical returns per day',
        'sql_implementation': '''
        CASE 
            WHEN h.total_returns > 0 AND h.customer_lifetime_days > 90 THEN
                (CAST(r.recent_returns AS DOUBLE) / 90.0) / 
                (CAST(h.total_returns AS DOUBLE) / CAST(h.customer_lifetime_days AS DOUBLE))
            ELSE 0
        END as recent_vs_avg_ratio
        ''',
        'business_value': 'Measures if return behavior is improving or worsening. Values >1 indicate worsening patterns.',
        'clustering_value': 'Helps identify customers whose return behavior is changing over time.',
        'expected_range': 'Float values >= 0'
    },
    'behavior_stability_score': {
        'category': 'Recency Analysis',
        'subcategory': 'Behavior Change',
        'description': 'Score of how stable return behavior is over time',
        'logic': 'Derived from recent_vs_avg_ratio, bucketed into stability scores',
        'sql_implementation': '''
        CASE 
            WHEN recent_vs_avg_ratio BETWEEN 0.8 AND 1.2 THEN 1.0
            WHEN recent_vs_avg_ratio BETWEEN 0.5 AND 1.5 THEN 0.7
            WHEN recent_vs_avg_ratio BETWEEN 0.2 AND 2.0 THEN 0.4
            ELSE 0.1
        END as behavior_stability_score
        ''',
        'business_value': 'Measures consistency in return behavior. Higher values indicate more predictable customers.',
        'clustering_value': 'Helps identify customers with stable vs. changing return patterns.',
        'expected_range': 'Float values between 0 and 1'
    },
    
    # CATEGORY INTELLIGENCE
    'category_diversity_score': {
        'category': 'Category Intelligence',
        'subcategory': 'Product Preferences',
        'description': 'Measure of diversity in product categories purchased',
        'logic': 'Number of unique categories divided by estimated total categories',
        'sql_implementation': 'CAST(unique_categories AS DOUBLE) / 20.0 as category_diversity_score',
        'business_value': 'Measures breadth of product interest. Higher values indicate customers who shop across many categories.',
        'clustering_value': 'Helps identify category-specific shoppers vs. diverse shoppers.',
        'expected_range': 'Float values >= 0, with validation threshold at 1.5'
    },
    'category_loyalty_score': {
        'category': 'Category Intelligence',
        'subcategory': 'Product Preferences',
        'description': 'Measure of concentration in specific product categories',
        'logic': 'Sum of squared proportions of purchases by category (similar to Herfindahl index)',
        'sql_implementation': 'sum(POWER(CAST(cc.category_purchases AS DOUBLE) / CAST(ct.total_purchases AS DOUBLE), 2)) as category_loyalty_raw',
        'business_value': 'Measures category loyalty. Higher values indicate customers who concentrate purchases in fewer categories.',
        'clustering_value': 'Helps identify customers with strong category preferences vs. diverse shoppers.',
        'expected_range': 'Float values between 0 and 1'
    },
    'high_return_category_affinity': {
        'category': 'Category Intelligence',
        'subcategory': 'Return Preferences',
        'description': 'Measure of tendency to return items from certain categories',
        'logic': 'Scaled score based on average category return rate',
        'sql_implementation': '''
        CASE 
            WHEN avg_category_return_rate > 0.3 THEN 1.0
            WHEN avg_category_return_rate > 0.15 THEN 0.7
            WHEN avg_category_return_rate > 0.05 THEN 0.4
            ELSE 0.1
        END as high_return_category_affinity
        ''',
        'business_value': 'Identifies if customers have specific categories they tend to return more frequently.',
        'clustering_value': 'Helps identify if return behavior is category-specific or general.',
        'expected_range': 'Float values between 0 and 1'
    },
    
    # ADJACENCY PATTERNS
    'sku_adjacency_orders': {
        'category': 'Adjacency Patterns',
        'subcategory': 'Related Purchases',
        'description': 'Count of unique SKU pairs ordered within 14 days of each other',
        'logic': 'Count of distinct SKU pairs purchased within 14 days',
        'sql_implementation': 'count(DISTINCT CONCAT(sku_a, \'_\', sku_b)) as sku_adjacency_orders',
        'business_value': 'Measures tendency to make related purchases. Higher values may indicate customers who respond to recommendations.',
        'clustering_value': 'Helps identify customers who make related purchases vs. independent purchases.',
        'expected_range': 'Integer values >= 0'
    },
    'sku_adjacency_returns': {
        'category': 'Adjacency Patterns',
        'subcategory': 'Related Returns',
        'description': 'Count of SKU pairs where both were returned',
        'logic': 'Count of SKU pairs where both items were returned',
        'sql_implementation': 'count(*) FILTER (WHERE sku_a_returned = 1 AND sku_b_returned = 1) as sku_adjacency_returns',
        'business_value': 'Measures tendency to return related items. Higher values may indicate systematic issues with certain product combinations.',
        'clustering_value': 'Helps identify customers who return related items vs. independent returns.',
        'expected_range': 'Integer values >= 0'
    },
    'sku_adjacency_timing': {
        'category': 'Adjacency Patterns',
        'subcategory': 'Related Purchase Timing',
        'description': 'Average days between adjacent SKU purchases',
        'logic': 'Average number of days between purchases of related SKUs',
        'sql_implementation': 'avg(CAST(days_between AS DOUBLE)) as sku_adjacency_timing',
        'business_value': 'Measures typical timing between related purchases. Lower values indicate more immediate follow-up purchases.',
        'clustering_value': 'Helps identify customers who make rapid related purchases vs. delayed related purchases.',
        'expected_range': 'Float values >= 0, with validation threshold at 14'
    },
    'sku_adjacency_return_timing': {
        'category': 'Adjacency Patterns',
        'subcategory': 'Related Return Timing',
        'description': 'Average days between adjacent SKU returns',
        'logic': 'Average number of days between returns of related SKUs',
        'sql_implementation': 'avg(CASE WHEN sku_a_returned = 1 AND sku_b_returned = 1 THEN CAST(days_between AS DOUBLE) ELSE NULL END) as sku_adjacency_return_timing',
        'business_value': 'Measures if related items are returned together or separately. Lower values indicate returns made together.',
        'clustering_value': 'Helps identify if customers return related items together or separately.',
        'expected_range': 'Float values >= 0, with validation threshold at 14'
    },
    
    # SEASONAL PATTERNS
    'seasonal_susceptibility_orders': {
        'category': 'Seasonal Trends',
        'subcategory': 'Seasonal Variation',
        'description': 'Coefficient of variation in seasonal order patterns',
        'logic': 'Standard deviation of seasonal orders divided by average seasonal orders',
        'sql_implementation': 'stddev(CAST(seasonal_orders AS DOUBLE)) / NULLIF(avg(CAST(seasonal_orders AS DOUBLE)), 0) as seasonal_susceptibility_orders',
        'business_value': 'Measures how much ordering behavior varies by season. Higher values indicate more seasonal shoppers.',
        'clustering_value': 'Helps identify seasonal shoppers vs. consistent shoppers.',
        'expected_range': 'Float values >= 0'
    },
    'seasonal_susceptibility_returns': {
        'category': 'Seasonal Trends',
        'subcategory': 'Seasonal Variation',
        'description': 'Coefficient of variation in seasonal return patterns',
        'logic': 'Standard deviation of seasonal returns divided by average seasonal returns',
        'sql_implementation': 'stddev(CAST(seasonal_returns AS DOUBLE)) / NULLIF(avg(CAST(seasonal_returns AS DOUBLE)), 0) as seasonal_susceptibility_returns',
        'business_value': 'Measures how much return behavior varies by season. Higher values indicate seasonal return patterns.',
        'clustering_value': 'Helps identify if return behavior is seasonal or consistent.',
        'expected_range': 'Float values >= 0'
    },
    
    # TREND ANALYSIS
    'trend_product_category_order_rate': {
        'category': 'Trend Analysis',
        'subcategory': 'Trend Following',
        'description': 'Correlation between customer orders and overall category popularity',
        'logic': 'Correlation between customer\'s monthly orders and overall monthly category orders',
        'sql_implementation': 'corr(CAST(customer_monthly_orders AS DOUBLE), CAST(monthly_category_orders AS DOUBLE)) as trend_product_category_order_rate',
        'business_value': 'Measures if customer follows product trends. Higher values indicate trend-following customers.',
        'clustering_value': 'Helps identify trend-following customers vs. independent purchasers.',
        'expected_range': 'Float values between -1 and 1'
    },
    'trend_product_category_return_rate': {
        'category': 'Trend Analysis',
        'subcategory': 'Trend Following',
        'description': 'Correlation between customer returns and overall category return trends',
        'logic': 'Correlation between customer\'s monthly returns and overall monthly category returns',
        'sql_implementation': 'corr(CAST(customer_monthly_returns AS DOUBLE), CAST(monthly_category_returns AS DOUBLE)) as trend_product_category_return_rate',
        'business_value': 'Measures if customer return behavior follows overall return trends. Higher values may indicate product quality issues rather than customer-specific issues.',
        'clustering_value': 'Helps identify if return behavior is driven by product trends or customer behavior.',
        'expected_range': 'Float values between -1 and 1'
    },
    
    # MONETARY VALUE METRICS
    'avg_order_value': {
        'category': 'Monetary Value Metrics',
        'subcategory': 'Order Value',
        'description': 'Average monetary value per order (scaled to 0-100)',
        'logic': 'Total gross value divided by order count, then scaled to 0-100 range',
        'sql_implementation': '''
        CASE 
            WHEN order_count = 0 THEN 0
            ELSE total_gross_value / NULLIF(order_count, 0)
        END AS avg_order_value
        ''',
        'business_value': 'Measures customer spending level. Higher values indicate higher-value customers.',
        'clustering_value': 'Critical for identifying high-value vs. low-value customers.',
        'expected_range': 'Float values between 0 and 100'
    },
    'avg_return_value': {
        'category': 'Monetary Value Metrics',
        'subcategory': 'Return Value',
        'description': 'Average monetary value of returned items (scaled to 0-100)',
        'logic': 'Value of returned items divided by return quantity, then scaled to 0-100 range',
        'sql_implementation': '''
        CASE 
            WHEN total_return_qty = 0 THEN 0
            ELSE returned_item_value / NULLIF(total_return_qty, 0)
        END AS avg_return_value
        ''',
        'business_value': 'Measures typical value of returned items. Higher values indicate returns of more expensive items.',
        'clustering_value': 'Helps identify if customers return high-value or low-value items.',
        'expected_range': 'Float values between 0 and 100'
    },
    'high_value_return_affinity': {
        'category': 'Monetary Value Metrics',
        'subcategory': 'Return Value',
        'description': 'Percentage of high-value orders that are returned',
        'logic': 'Number of high-value orders with returns divided by total high-value orders',
        'sql_implementation': '''
        CASE 
            WHEN high_value_orders = 0 THEN 0
            ELSE CAST(high_value_returns AS DOUBLE) / NULLIF(CAST(high_value_orders AS DOUBLE), 0)
        END * 100 AS high_value_return_affinity
        ''',
        'business_value': 'Measures tendency to return expensive items. Higher values indicate customers who specifically return high-value purchases.',
        'clustering_value': 'Critical for identifying customers who disproportionately return expensive items.',
        'expected_range': 'Float values between 0 and 100'
    }
}

def generate_comprehensive_feature_report():
    """Generate a comprehensive Excel report with detailed feature analysis"""
    logger.info("Generating comprehensive feature details report...")
    
    # Create individual feature details DataFrame
    feature_rows = []
    for feature_name, metadata in feature_metadata.items():
        feature_rows.append({
            'Feature Name': feature_name,
            'Category': metadata['category'],
            'Subcategory': metadata['subcategory'],
            'Description': metadata['description'],
            'Logic': metadata['logic'],
            'SQL Implementation': metadata.get('sql_implementation', 'Complex implementation - see code'),
            'Business Value': metadata['business_value'],
            'Clustering Value': metadata['clustering_value'],
            'Expected Range': metadata['expected_range']
        })
    
    feature_df = pd.DataFrame(feature_rows)
    
    # Add category order for sorting
    category_order = {
        'Volume Metrics': 1,
        'Return Behavior': 2,
        'Temporal Patterns': 3,
        'Recency Analysis': 4,
        'Category Intelligence': 5,
        'Adjacency Patterns': 6,
        'Seasonal Trends': 7,
        'Trend Analysis': 8,
        'Monetary Value Metrics': 9
    }
    feature_df['Category Order'] = feature_df['Category'].map(category_order)
    feature_df = feature_df.sort_values(by=['Category Order', 'Feature Name']).drop('Category Order', axis=1)
    
    # Create a summary of categories and subcategories
    category_summary = []
    for category, cat_features in feature_df.groupby('Category'):
        subcategories = cat_features.groupby('Subcategory').size().reset_index()
        subcategories.columns = ['Subcategory', 'Feature Count']
        for _, row in subcategories.iterrows():
            category_summary.append({
                'Category': category,
                'Subcategory': row['Subcategory'],
                'Feature Count': row['Feature Count'],
                'Features': ', '.join(cat_features[cat_features['Subcategory'] == row['Subcategory']]['Feature Name'].tolist())
            })
    
    category_summary_df = pd.DataFrame(category_summary)
    category_summary_df['Category Order'] = category_summary_df['Category'].map(category_order)
    category_summary_df = category_summary_df.sort_values(by=['Category Order', 'Subcategory']).drop('Category Order', axis=1)
    
    # Create business use cases
    use_cases = pd.DataFrame([
        {
            'Use Case': 'Identifying Rental Candidates',
            'Description': 'Customers who may be using return policies to effectively "rent" items',
            'Key Features': 'high_value_return_affinity, avg_days_to_return, return_rate, consecutive_returns',
            'Business Impact': 'Reduce financial losses from rental behavior and improve inventory management'
        },
        {
            'Use Case': 'Churn Risk Assessment',
            'Description': 'Customers who may stop shopping due to bad experiences',
            'Key Features': 'recent_vs_avg_ratio, behavior_stability_score, avg_days_to_return, customer_lifetime_days',
            'Business Impact': 'Proactively address issues for valuable customers at risk of churning'
        },
        {
            'Use Case': 'Marketing Segmentation',
            'Description': 'Grouping customers for targeted marketing campaigns',
            'Key Features': 'category_loyalty_score, avg_order_value, customer_tenure_stage, seasonal_susceptibility_orders',
            'Business Impact': 'Improve marketing ROI through more relevant promotions and messaging'
        },
        {
            'Use Case': 'Return Policy Optimization',
            'Description': 'Adjusting return policies based on customer behavior patterns',
            'Key Features': 'return_rate, high_value_return_affinity, avg_days_to_return, consecutive_returns',
            'Business Impact': 'Balance customer satisfaction with business profitability through data-driven policies'
        }
    ])
    
    # Create expected customer segments
    expected_segments = pd.DataFrame([
        {
            'Cluster Name': 'Low-Risk Value Shoppers',
            'Description': 'Customers with low return rates and high average order values',
            'Key Features': 'return_rate (low), avg_order_value (high), customer_lifetime_days (high)',
            'Business Strategy': 'Ideal customers for premium promotions and loyalty rewards'
        },
        {
            'Cluster Name': 'Fit Experimenters',
            'Description': 'Customers who order multiple sizes/styles and return excess items',
            'Key Features': 'return_rate (high), avg_returns_per_order (high), avg_order_size (high)',
            'Business Strategy': 'Improve size guides, offer virtual try-on, suggest consistent sizing'
        },
        {
            'Cluster Name': 'Serial Returners',
            'Description': 'Customers with very high return rates across categories',
            'Key Features': 'return_rate (very high), consecutive_returns (high), high_value_return_affinity (high)',
            'Business Strategy': 'Flag for review, implement stricter return policies for these customers'
        },
        {
            'Cluster Name': 'Seasonal Shoppers',
            'Description': 'Customers who primarily shop (and return) during specific seasons',
            'Key Features': 'seasonal_susceptibility_orders (high), seasonal_susceptibility_returns (high)',
            'Business Strategy': 'Target seasonal promotions, improve seasonal product fit/descriptions'
        },
        {
            'Cluster Name': 'Category Specialists',
            'Description': 'Customers who shop within specific categories with low returns',
            'Key Features': 'category_loyalty_score (high), category_diversity_score (low), return_rate (low)',
            'Business Strategy': 'Recommend products within their preferred categories, create category-specific promotions'
        },
        {
            'Cluster Name': 'Trend Followers',
            'Description': 'Customers whose purchase patterns closely follow category trends',
            'Key Features': 'trend_product_category_order_rate (high), high_return_category_affinity (high)',
            'Business Strategy': 'Early access to new trends, target for new product launches'
        },
        {
            'Cluster Name': 'High-Value Returners',
            'Description': 'Customers who specifically return high-value items',
            'Key Features': 'high_value_return_affinity (high), avg_return_value (high), return_rate (medium-high)',
            'Business Strategy': 'Implement specialized return processes for high-value items, review for fraud'
        }
    ])
    
    # Create implementation considerations
    implementation_notes = pd.DataFrame([
        {
            'Aspect': 'Feature Redundancy',
            'Description': 'Some features (e.g., return_rate and return_ratio) are redundant and may be consolidated',
            'Recommendation': 'Consider removing redundant features before clustering to avoid biasing results'
        },
        {
            'Aspect': 'Feature Scaling',
            'Description': 'Features have different scales and distributions, requiring normalization',
            'Recommendation': 'Apply RobustScaler to handle outliers and normalize distributions'
        },
        {
            'Aspect': 'Outlier Handling',
            'Description': 'Some features show extreme values that may distort clustering',
            'Recommendation': 'Use Isolation Forest with contamination=0.05 to identify and handle outliers'
        },
        {
            'Aspect': 'Dimensionality Reduction',
            'Description': 'High feature count may lead to the curse of dimensionality',
            'Recommendation': 'Apply PCA or UMAP for visualization and to reduce dimensions before clustering'
        },
        {
            'Aspect': 'Feature Selection',
            'Description': 'Not all features contribute equally to meaningful clusters',
            'Recommendation': 'Use variance threshold of 0.01 to remove near-constant features'
        },
        {
            'Aspect': 'Data Quality',
            'Description': 'Some customers have limited history, affecting feature reliability',
            'Recommendation': 'Filter for customers with sufficient order history (>=2 orders)'
        }
    ])
    
    # Generate timestamp for file name
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    excel_file = f'comprehensive_feature_details_{timestamp}.xlsx'
    
    # Create Excel writer
    with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
        # Write detailed feature information
        feature_df.to_excel(writer, sheet_name='Feature Details', index=False)
        
        # Write category summary
        category_summary_df.to_excel(writer, sheet_name='Category Summary', index=False)
        
        # Write use cases
        use_cases.to_excel(writer, sheet_name='Business Use Cases', index=False)
        
        # Write expected segments
        expected_segments.to_excel(writer, sheet_name='Expected Segments', index=False)
        
        # Write implementation notes
        implementation_notes.to_excel(writer, sheet_name='Implementation Notes', index=False)
        
        # Write executive summary
        summary_df = pd.DataFrame([
            {
                'Section': 'Project Overview',
                'Content': 'This analysis provides a comprehensive breakdown of all 37 features used for customer clustering based on return behavior patterns.'
            },
            {
                'Section': 'Feature Engineering Approach',
                'Content': f'Features are organized across {len(category_order)} categories and cover multiple dimensions of customer behavior including purchase patterns, return behaviors, timing, product preferences, and monetary aspects.'
            },
            {
                'Section': 'Key Feature Categories',
                'Content': 'Volume Metrics (5), Return Behavior (8), Temporal Patterns (4), Recency Analysis (4), Category Intelligence (3), Adjacency Patterns (4), Seasonal Trends (2), Trend Analysis (2), Monetary Value Metrics (3)'
            },
            {
                'Section': 'Implementation Details',
                'Content': 'Features are calculated using SQL queries with common table expressions, window functions, and statistical calculations in DuckDB. Each feature includes business logic validation to ensure values fall within expected ranges.'
            },
            {
                'Section': 'Business Applications',
                'Content': 'Features support multiple use cases: identifying rental-like behavior, assessing churn risk, optimizing marketing segmentation, and tailoring return policies.'
            }
        ])
        summary_df.to_excel(writer, sheet_name='Executive Summary', index=False)
    
    logger.info(f"Comprehensive feature details report generated: {excel_file}")
    return excel_file

if __name__ == "__main__":
    excel_file = generate_comprehensive_feature_report()
    print(f"Report generated: {excel_file}")
