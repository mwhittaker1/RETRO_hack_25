"""
Feature Analysis Report Generator
Analyzes and documents all features in the silver layer of the customer clustering database
"""

import pandas as pd
import numpy as np
from datetime import datetime
import logging
import sys

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Define the features with detailed descriptions and business value
feature_metadata = {
    # BASIC VOLUME METRICS
    'sales_order_no_nunique': {
        'category': 'Volume Metrics',
        'description': 'Count of unique sales orders per customer',
        'logic': 'COUNT(DISTINCT sales_order_no) for each customer',
        'business_value': 'Measures purchase frequency and customer engagement. Higher values indicate more active customers.',
        'clustering_value': 'Differentiates between high-frequency shoppers and occasional customers.'
    },
    'sku_nunique': {
        'category': 'Volume Metrics',
        'description': 'Count of unique SKUs purchased by customer',
        'logic': 'COUNT(DISTINCT q_sku_id) for each customer',
        'business_value': 'Measures product diversity in customer purchases. Higher values indicate broader product interest.',
        'clustering_value': 'Helps identify customers with diverse vs. targeted purchasing patterns.'
    },
    'items_returned_count': {
        'category': 'Volume Metrics',
        'description': 'Count of items returned by customer',
        'logic': 'SUM(CASE WHEN return_qty > 0 THEN 1 ELSE 0 END) for each customer',
        'business_value': 'Measures absolute return volume. Higher values may indicate fit issues or customer dissatisfaction.',
        'clustering_value': 'Raw count helps identify customers with high return volumes regardless of purchase volume.'
    },
    'sales_qty_mean': {
        'category': 'Volume Metrics',
        'description': 'Average quantity per line item',
        'logic': 'AVG(sales_qty) for each customer',
        'business_value': 'Measures typical purchase quantity behavior. Higher values may indicate bulk purchasing.',
        'clustering_value': 'Helps distinguish between single-item purchasers and bulk buyers.'
    },
    'avg_order_size': {
        'category': 'Volume Metrics',
        'description': 'Average number of items per order',
        'logic': 'COUNT(*) / COUNT(DISTINCT sales_order_no) for each customer',
        'business_value': 'Measures shopping basket size. Higher values indicate customers who purchase multiple items per order.',
        'clustering_value': 'Differentiates between customers who make single-item purchases vs. multiple-item purchases.'
    },
    
    # RETURN BEHAVIOR PATTERNS
    'return_rate': {
        'category': 'Return Behavior',
        'description': 'Ratio of returned quantity to total sales quantity',
        'logic': 'total_return_qty / total_sales_qty for each customer',
        'business_value': 'Primary indicator of customer return behavior. Higher values indicate customers who return a large portion of purchases.',
        'clustering_value': 'Essential metric for identifying high-return-risk customer segments.'
    },
    'return_ratio': {
        'category': 'Return Behavior',
        'description': 'Ratio of returned quantity to total sales quantity (same as return_rate)',
        'logic': 'Same as return_rate (redundant feature)',
        'business_value': 'Redundant to return_rate - measures proportion of purchases that are returned.',
        'clustering_value': 'May be removed due to redundancy with return_rate.'
    },
    'return_product_variety': {
        'category': 'Return Behavior',
        'description': 'Count of unique product SKUs that were returned',
        'logic': 'COUNT(DISTINCT q_sku_id) FILTER (WHERE return_qty > 0) for each customer',
        'business_value': 'Measures diversity of returned products. High values indicate returns across many product types rather than issues with specific items.',
        'clustering_value': 'Helps identify if customer return behavior is product-specific or broadly distributed.'
    },
    'avg_returns_per_order': {
        'category': 'Return Behavior',
        'description': 'Average number of returned items per order',
        'logic': 'items_with_returns / total_orders for each customer',
        'business_value': 'Measures how many items in a typical order are returned. Higher values indicate customers who return multiple items from each order.',
        'clustering_value': 'Differentiates between customers who return single items vs. those who return whole orders.'
    },
    'return_frequency_ratio': {
        'category': 'Return Behavior',
        'description': 'Ratio of items with returns to total items purchased',
        'logic': 'items_with_returns / total_items for each customer',
        'business_value': 'Measures how frequently a customer returns items. Similar to return rate but on item count rather than quantity.',
        'clustering_value': 'Alternative measure of return propensity that focuses on item count rather than quantity.'
    },
    'return_intensity': {
        'category': 'Return Behavior',
        'description': 'Average quantity returned per item with returns',
        'logic': 'total_return_qty / items_with_returns for each customer',
        'business_value': 'Measures if customers tend to return entire quantities of an item or partial quantities. Higher values indicate full returns rather than partial.',
        'clustering_value': 'Helps identify customers who tend to return entire quantities vs. partial quantities.'
    },
    
    # CONSECUTIVE RETURNS
    'consecutive_returns': {
        'category': 'Return Behavior',
        'description': 'Count of instances where customer had consecutive orders with returns',
        'logic': 'Complex calculation tracking order sequence and identifying consecutive return patterns',
        'business_value': 'Identifies habitual returners who have patterns of returning items order after order. Higher values indicate more concerning return patterns.',
        'clustering_value': 'Strong indicator of customers with problematic return behavior that persists across multiple orders.'
    },
    'avg_consecutive_returns': {
        'category': 'Return Behavior',
        'description': 'Average length of consecutive return sequences',
        'logic': 'For each sequence of consecutive returns, calculates the average length',
        'business_value': 'Measures the typical duration of return patterns. Higher values indicate longer streaks of orders with returns.',
        'clustering_value': 'Helps distinguish between occasional returners and those with extended return patterns.'
    },
    
    # TEMPORAL & TIMING PATTERNS
    'customer_lifetime_days': {
        'category': 'Temporal Patterns',
        'description': 'Number of days between first and last order',
        'logic': 'DATE_DIFF(day, min(order_date), max(order_date)) for each customer',
        'business_value': 'Measures customer relationship duration. Higher values indicate longer-term customers.',
        'clustering_value': 'Helps distinguish between new customers, growing customers, and long-term customers.'
    },
    'avg_days_to_return': {
        'category': 'Temporal Patterns',
        'description': 'Average number of days between order and return',
        'logic': 'AVG(DATE_DIFF(day, order_date, return_date)) for orders with returns',
        'business_value': 'Measures how quickly customers typically return items. Lower values may indicate fit/satisfaction issues; higher values may indicate policy abuse.',
        'clustering_value': 'Helps identify rapid returners vs. delayed returners, which may have different business implications.'
    },
    'return_timing_spread': {
        'category': 'Temporal Patterns',
        'description': 'Standard deviation of days between order and return',
        'logic': 'STDDEV(DATE_DIFF(day, order_date, return_date)) for orders with returns',
        'business_value': 'Measures consistency in return timing. Lower values indicate customers with very predictable return patterns.',
        'clustering_value': 'Helps identify customers with variable return timing vs. those with consistent patterns.'
    },
    'customer_tenure_stage': {
        'category': 'Temporal Patterns',
        'description': 'Categorization of customer lifetime',
        'logic': 'Categorizes customers as New (<=90 days), Growing (<=180 days), Mature (<=365 days), or Veteran (>365 days)',
        'business_value': 'Provides a simple segmentation of customer relationship duration.',
        'clustering_value': 'Helps analyze if return patterns differ across customer tenure stages.'
    },
    
    # RECENCY ANALYSIS
    'recent_orders': {
        'category': 'Recency Analysis',
        'description': 'Count of orders in the last 90 days',
        'logic': 'COUNT(DISTINCT sales_order_no) for orders within 90 days of latest date',
        'business_value': 'Measures recent purchase activity. Higher values indicate more active customers.',
        'clustering_value': 'Helps identify currently active customers vs. lapsed customers.'
    },
    'recent_returns': {
        'category': 'Recency Analysis',
        'description': 'Count of returns in the last 90 days',
        'logic': 'COUNT of items with returns within 90 days of latest date',
        'business_value': 'Measures recent return activity. Higher values indicate currently problematic customers.',
        'clustering_value': 'Critical for identifying customers with current return issues rather than historical patterns.'
    },
    'recent_vs_avg_ratio': {
        'category': 'Recency Analysis',
        'description': 'Ratio of recent return rate to historical return rate',
        'logic': '(recent_returns/90 days) / (total_returns/customer_lifetime_days)',
        'business_value': 'Measures if return behavior is improving or worsening. Values >1 indicate worsening patterns.',
        'clustering_value': 'Helps identify customers whose return behavior is changing over time.'
    },
    'behavior_stability_score': {
        'category': 'Recency Analysis',
        'description': 'Score of how stable return behavior is over time',
        'logic': 'Derived from recent_vs_avg_ratio: 1.0 (stable), 0.7, 0.4, or 0.1 (volatile)',
        'business_value': 'Measures consistency in return behavior. Higher values indicate more predictable customers.',
        'clustering_value': 'Helps identify customers with stable vs. changing return patterns.'
    },
    
    # CATEGORY INTELLIGENCE
    'category_diversity_score': {
        'category': 'Category Intelligence',
        'description': 'Measure of diversity in product categories purchased',
        'logic': 'unique_categories / 20.0 (normalized by estimated total categories)',
        'business_value': 'Measures breadth of product interest. Higher values indicate customers who shop across many categories.',
        'clustering_value': 'Helps identify category-specific shoppers vs. diverse shoppers.'
    },
    'category_loyalty_score': {
        'category': 'Category Intelligence',
        'description': 'Measure of concentration in specific product categories',
        'logic': 'Sum of squared proportions of purchases by category (similar to Herfindahl index)',
        'business_value': 'Measures category loyalty. Higher values indicate customers who concentrate purchases in fewer categories.',
        'clustering_value': 'Helps identify customers with strong category preferences vs. diverse shoppers.'
    },
    'high_return_category_affinity': {
        'category': 'Category Intelligence',
        'description': 'Measure of tendency to return items from certain categories',
        'logic': 'Scaled score (0.1-1.0) based on average category return rate',
        'business_value': 'Identifies if customers have specific categories they tend to return more frequently.',
        'clustering_value': 'Helps identify if return behavior is category-specific or general.'
    },
    
    # ADJACENCY PATTERNS
    'sku_adjacency_orders': {
        'category': 'Adjacency Patterns',
        'description': 'Count of unique SKU pairs ordered within 14 days of each other',
        'logic': 'COUNT(DISTINCT CONCAT(sku_a, "_", sku_b)) for SKUs purchased within 14 days',
        'business_value': 'Measures tendency to make related purchases. Higher values may indicate customers who respond to recommendations.',
        'clustering_value': 'Helps identify customers who make related purchases vs. independent purchases.'
    },
    'sku_adjacency_returns': {
        'category': 'Adjacency Patterns',
        'description': 'Count of SKU pairs where both were returned',
        'logic': 'COUNT(*) FILTER (WHERE sku_a_returned = 1 AND sku_b_returned = 1)',
        'business_value': 'Measures tendency to return related items. Higher values may indicate systematic issues with certain product combinations.',
        'clustering_value': 'Helps identify customers who return related items vs. independent returns.'
    },
    'sku_adjacency_timing': {
        'category': 'Adjacency Patterns',
        'description': 'Average days between adjacent SKU purchases',
        'logic': 'AVG(days_between) for SKU pairs purchased within 14 days',
        'business_value': 'Measures typical timing between related purchases. Lower values indicate more immediate follow-up purchases.',
        'clustering_value': 'Helps identify customers who make rapid related purchases vs. delayed related purchases.'
    },
    'sku_adjacency_return_timing': {
        'category': 'Adjacency Patterns',
        'description': 'Average days between adjacent SKU returns',
        'logic': 'AVG(days_between) FILTER (WHERE sku_a_returned = 1 AND sku_b_returned = 1)',
        'business_value': 'Measures if related items are returned together or separately. Lower values indicate returns made together.',
        'clustering_value': 'Helps identify if customers return related items together or separately.'
    },
    
    # SEASONAL PATTERNS
    'seasonal_susceptibility_orders': {
        'category': 'Seasonal Trends',
        'description': 'Coefficient of variation in seasonal order patterns',
        'logic': 'STDDEV(seasonal_orders) / AVG(seasonal_orders) across seasons',
        'business_value': 'Measures how much ordering behavior varies by season. Higher values indicate more seasonal shoppers.',
        'clustering_value': 'Helps identify seasonal shoppers vs. consistent shoppers.'
    },
    'seasonal_susceptibility_returns': {
        'category': 'Seasonal Trends',
        'description': 'Coefficient of variation in seasonal return patterns',
        'logic': 'STDDEV(seasonal_returns) / AVG(seasonal_returns) across seasons',
        'business_value': 'Measures how much return behavior varies by season. Higher values indicate seasonal return patterns.',
        'clustering_value': 'Helps identify if return behavior is seasonal or consistent.'
    },
    
    # TREND ANALYSIS
    'trend_product_category_order_rate': {
        'category': 'Trend Analysis',
        'description': 'Correlation between customer orders and overall category popularity',
        'logic': 'CORR(customer_monthly_orders, monthly_category_orders)',
        'business_value': 'Measures if customer follows product trends. Higher values indicate trend-following customers.',
        'clustering_value': 'Helps identify trend-following customers vs. independent purchasers.'
    },
    'trend_product_category_return_rate': {
        'category': 'Trend Analysis',
        'description': 'Correlation between customer returns and overall category return trends',
        'logic': 'CORR(customer_monthly_returns, monthly_category_returns)',
        'business_value': 'Measures if customer return behavior follows overall return trends. Higher values may indicate product quality issues rather than customer-specific issues.',
        'clustering_value': 'Helps identify if return behavior is driven by product trends or customer behavior.'
    },
    
    # MONETARY VALUE METRICS
    'avg_order_value': {
        'category': 'Monetary Value Metrics',
        'description': 'Average monetary value per order (scaled to 0-100)',
        'logic': 'total_gross_value / order_count, then scaled to 0-100 range',
        'business_value': 'Measures customer spending level. Higher values indicate higher-value customers.',
        'clustering_value': 'Critical for identifying high-value vs. low-value customers.'
    },
    'avg_return_value': {
        'category': 'Monetary Value Metrics',
        'description': 'Average monetary value of returned items (scaled to 0-100)',
        'logic': 'returned_item_value / total_return_qty, then scaled to 0-100 range',
        'business_value': 'Measures typical value of returned items. Higher values indicate returns of more expensive items.',
        'clustering_value': 'Helps identify if customers return high-value or low-value items.'
    },
    'high_value_return_affinity': {
        'category': 'Monetary Value Metrics',
        'description': 'Percentage of high-value orders that are returned',
        'logic': 'high_value_returns / high_value_orders * 100',
        'business_value': 'Measures tendency to return expensive items. Higher values indicate customers who specifically return high-value purchases.',
        'clustering_value': 'Critical for identifying customers who disproportionately return expensive items.'
    }
}

def generate_excel_report():
    """Generate a comprehensive Excel report analyzing all features"""
    logger.info("Generating feature analysis report...")
    
    # Create a DataFrame from the feature metadata
    feature_rows = []
    for feature_name, metadata in feature_metadata.items():
        feature_rows.append({
            'Feature Name': feature_name,
            'Category': metadata['category'],
            'Description': metadata['description'],
            'Implementation Logic': metadata['logic'],
            'Business Value': metadata['business_value'],
            'Clustering Value': metadata['clustering_value']
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
    
    # Create a pivot table of feature counts by category
    category_counts = feature_df['Category'].value_counts().reset_index()
    category_counts.columns = ['Category', 'Feature Count']
    category_counts['Category Order'] = category_counts['Category'].map(category_order)
    category_counts = category_counts.sort_values('Category Order').drop('Category Order', axis=1)
    
    # Create a summary of key clusters expected
    expected_clusters = pd.DataFrame([
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
    
    # Create a feature importance summary for expected use cases
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
    
    # Create Excel writer
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    excel_file = f'feature_analysis_report_{timestamp}.xlsx'
    
    with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
        # Write feature details
        feature_df.to_excel(writer, sheet_name='Feature Details', index=False)
        
        # Write category summary
        category_counts.to_excel(writer, sheet_name='Feature Categories', index=False)
        
        # Write expected clusters
        expected_clusters.to_excel(writer, sheet_name='Expected Clusters', index=False)
        
        # Write use cases
        use_cases.to_excel(writer, sheet_name='Business Use Cases', index=False)
        
        # Write executive summary
        summary_df = pd.DataFrame([
            {
                'Section': 'Project Overview',
                'Content': 'This analysis examines features used for customer clustering based on return behavior and sales trends.'
            },
            {
                'Section': 'Feature Summary',
                'Content': f'The project utilizes {len(feature_df)} features across {len(category_counts)} categories to identify meaningful customer segments.'
            },
            {
                'Section': 'Key Insights',
                'Content': 'Features are designed to capture multiple dimensions of customer behavior including return patterns, purchase value, product preferences, and temporal behaviors.'
            },
            {
                'Section': 'Clustering Approach',
                'Content': 'The feature set is well-designed to identify multiple customer segments with distinct return behaviors and business implications.'
            },
            {
                'Section': 'Business Applications',
                'Content': 'Results can be applied to return policy optimization, marketing segmentation, inventory management, and customer retention strategies.'
            }
        ])
        summary_df.to_excel(writer, sheet_name='Executive Summary', index=False)
    
    logger.info(f"Feature analysis report generated: {excel_file}")
    return excel_file

if __name__ == "__main__":
    excel_file = generate_excel_report()
    print(f"Report generated: {excel_file}")
