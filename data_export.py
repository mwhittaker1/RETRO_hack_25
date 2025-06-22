"""
data_export.py - Comprehensive Customer Clustering Analysis Export
Exports detailed clustering results to Excel with multiple sheets
"""

import pandas as pd
import numpy as np
from datetime import datetime
from functions import ReturnsClusteringAnalysis
import warnings
warnings.filterwarnings('ignore')

def create_feature_inventory():
    """Define comprehensive feature inventory"""
    return {
        # Core aggregated features
        'SALES_ORDER_NO_nunique': {
            'description': 'Number of unique orders placed',
            'calculation': 'nunique() on SALES_ORDER_NO',
            'business_meaning': 'Customer purchase frequency',
            'expected_range': '1 to 1000+',
            'used_in_thresholds': ['orders_75 (VIP identification)'],
            'used_in_clustering': True
        },
        
        'SKU_nunique': {
            'description': 'Number of unique products purchased',
            'calculation': 'nunique() on SKU',
            'business_meaning': 'Product variety/exploration behavior',
            'expected_range': '1 to 3000+',
            'used_in_thresholds': ['75th percentile for EXPLORERS'],
            'used_in_clustering': True
        },
        
        'RETURN_RATE': {
            'description': 'Items returned / total items purchased',
            'calculation': 'items_with_returns / total_items_purchased',
            'business_meaning': 'Frequency of return behavior',
            'expected_range': '0.0 to 1.0',
            'used_in_thresholds': ['return_rate_75', 'return_rate_25'],
            'used_in_clustering': True
        },
        
        'RETURN_RATIO': {
            'description': 'Quantity returned / quantity purchased',
            'calculation': 'sum(RETURN_QTY) / sum(SALES_QTY)',
            'business_meaning': 'Intensity of returns (partial vs full)',
            'expected_range': '0.0 to 1.0+',
            'used_in_thresholds': ['return_ratio_75'],
            'used_in_clustering': True
        },
        
        'ITEMS_RETURNED_COUNT': {
            'description': 'Total number of items returned',
            'calculation': 'count of records with RETURN_QTY > 0',
            'business_meaning': 'Absolute return volume',
            'expected_range': '1 to 1000+',
            'used_in_thresholds': [],
            'used_in_clustering': True
        },
        
        'RETURN_PRODUCT_VARIETY': {
            'description': 'Number of different SKUs returned',
            'calculation': 'nunique() on SKU where RETURN_QTY > 0',
            'business_meaning': 'Breadth of return behavior',
            'expected_range': '1 to product variety',
            'used_in_thresholds': [],
            'used_in_clustering': True
        },
        
        'CUSTOMER_LIFETIME_DAYS': {
            'description': 'Days between first and last order',
            'calculation': '(ORDER_DATE_max - ORDER_DATE_min).days',
            'business_meaning': 'Customer tenure/relationship length',
            'expected_range': '0 to 1000+ days',
            'used_in_thresholds': [],
            'used_in_clustering': True
        },
        
        'RECENT_ORDERS': {
            'description': 'Unique orders in last 90 days',
            'calculation': 'nunique() on SALES_ORDER_NO where ORDER_DATE >= recent_date',
            'business_meaning': 'Current engagement level',
            'expected_range': '0 to 50+',
            'used_in_thresholds': ['recent_orders_25'],
            'used_in_clustering': True
        },
        
        'RECENT_RETURNS': {
            'description': 'Items returned in last 90 days',
            'calculation': 'count where ORDER_DATE >= recent_date AND RETURN_QTY > 0',
            'business_meaning': 'Recent return activity',
            'expected_range': '0 to 100+',
            'used_in_thresholds': [],
            'used_in_clustering': True
        },
        
        'SALES_QTY_mean': {
            'description': 'Average quantity per purchase item',
            'calculation': 'mean() on SALES_QTY',
            'business_meaning': 'Purchase size behavior',
            'expected_range': '1.0 to 10+',
            'used_in_thresholds': [],
            'used_in_clustering': True
        },
        
        'AVG_DAYS_TO_RETURN': {
            'description': 'Average days between order and return',
            'calculation': 'mean() on (RETURN_DATE - ORDER_DATE).days',
            'business_meaning': 'Return timing pattern',
            'expected_range': '0 to 365+ days',
            'used_in_thresholds': ['Used in FAST RETURNERS logic'],
            'used_in_clustering': True
        },
        
        'RETURN_TIMING_SPREAD': {
            'description': 'Variability in return timing',
            'calculation': 'max - min of days_to_return per customer',
            'business_meaning': 'Consistency of return behavior',
            'expected_range': '0 to 365+ days',
            'used_in_thresholds': [],
            'used_in_clustering': True
        },
        
        'AVG_RETURNS_PER_ORDER': {
            'description': 'Average items returned per order',
            'calculation': 'mean() of items returned grouped by order',
            'business_meaning': 'Batch return behavior',
            'expected_range': '1.0 to 20+',
            'used_in_thresholds': ['returns_per_order_75'],
            'used_in_clustering': True
        },
        
        'RETURN_FREQUENCY_RATIO': {
            'description': 'Returns per order ratio',
            'calculation': 'items_returned_count / unique_orders',
            'business_meaning': 'Return frequency relative to purchase frequency',
            'expected_range': '0.0 to 10+',
            'used_in_thresholds': [],
            'used_in_clustering': True
        },
        
        'AVG_RETURN_INTENSITY': {
            'description': 'Average return quantity / sales quantity per returned item',
            'calculation': 'mean(RETURN_QTY / SALES_QTY) for returned items',
            'business_meaning': 'Partial vs full return tendency',
            'expected_range': '0.0 to 1.0+',
            'used_in_thresholds': ['return_intensity_75'],
            'used_in_clustering': True
        },
        
        # New features (if implemented)
        'RECENT_VS_AVG_RATIO': {
            'description': 'Recent return rate / historical return rate',
            'calculation': 'recent_return_rate / overall_return_rate',
            'business_meaning': 'Trend in return behavior (increasing/decreasing)',
            'expected_range': '0.0 to 5.0+',
            'used_in_thresholds': ['recent_vs_avg_75 for CHURN ALERT'],
            'used_in_clustering': True
        },
        
        'RETURN_TREND_INCREASING': {
            'description': 'Binary flag for increasing return trend',
            'calculation': '1 if RECENT_VS_AVG_RATIO > 1.5 else 0',
            'business_meaning': 'Early churn warning signal',
            'expected_range': '0 or 1',
            'used_in_thresholds': ['Used in CHURN ALERT logic'],
            'used_in_clustering': True
        }
    }

def create_cluster_profiles(analyzer, cluster_summary, interpretations):
    """Create comprehensive cluster profiles"""
    cluster_profiles = []
    
    for cluster_id in cluster_summary.index:
        profile = cluster_summary.loc[cluster_id]
        cluster_info = interpretations[cluster_id]
        
        # Identify key distinguishing features for this cluster
        feature_analysis = {}
        
        # Compare each feature to overall population
        for feature in analyzer.customer_features.columns:
            if feature != 'CLUSTER' and feature in profile.index:
                cluster_value = profile[feature]
                population_75 = analyzer.customer_features[feature].quantile(0.75)
                population_25 = analyzer.customer_features[feature].quantile(0.25)
                
                if cluster_value >= population_75:
                    feature_analysis[feature] = 'HIGH'
                elif cluster_value <= population_25:
                    feature_analysis[feature] = 'LOW'
                else:
                    feature_analysis[feature] = 'MODERATE'
        
        cluster_profiles.append({
            'cluster_id': cluster_id,
            'name': cluster_info['type'],
            'size': cluster_info['customers'],
            'percentage': (cluster_info['customers'] / len(analyzer.customer_features)) * 100,
            'action': cluster_info['action'],
            'key_features': feature_analysis,
            'distinguishing_features': [k for k, v in feature_analysis.items() 
                        if v in ['HIGH', 'LOW']]
        })
    
    return cluster_profiles

def create_expected_clusters_reference():
    """Create comprehensive reference of all possible cluster types"""
    return {
        "VIP Champions": {
            "description": "High-value, low-return customers with long tenure",
            "key_characteristics": [
                "Orders: 50+ (Top 10%)",
                "Return Rate: <0.25 (Bottom 25%)", 
                "Customer Lifetime: 800+ days",
                "Recent Activity: High (3+ recent orders)",
                "Product Variety: High exploration"
            ],
            "business_value": "High",
            "risk_level": "Low",
            "marketing_action": "Premium loyalty program, exclusive access, VIP treatment",
            "expected_size": "10-15% of returning customers",
            "priority": "High - Retention",
            "kpis_to_track": ["Lifetime Value", "Purchase Frequency", "Satisfaction Scores"]
        },
        
        "Heavy Returners": {
            "description": "Customers with very high return rates - potential policy abusers",
            "key_characteristics": [
                "Return Rate: >0.6 (Top 25%)",
                "Return Ratio: >0.5 (High quantity returns)",
                "Return Intensity: >0.8 (Full returns)",
                "Days to Return: <10 (Fast returns)",
                "Returns per Order: 5+"
            ],
            "business_value": "Low-Medium",
            "risk_level": "High",
            "marketing_action": "Return policy enforcement, education, account monitoring",
            "expected_size": "5-10% of returning customers",
            "priority": "High - Risk Management",
            "kpis_to_track": ["Return Rate", "Return Value", "Policy Violations"]
        },
        
        "Churn Risk - Disengaged": {
            "description": "Previously active customers now showing low engagement",
            "key_characteristics": [
                "Recent Orders: <2 (Bottom 25%)",
                "Historical Orders: 15+ (Was active)",
                "Customer Lifetime: 500+ days (Established)",
                "Return Rate: Moderate-High (Had issues)",
                "Recent Returns: Low (Not currently active)"
            ],
            "business_value": "Medium-High",
            "risk_level": "High",
            "marketing_action": "Win-back campaigns, personalized offers, satisfaction surveys",
            "expected_size": "15-25% of returning customers",
            "priority": "High - Retention",
            "kpis_to_track": ["Reactivation Rate", "Time Since Last Order", "Win-back Success"]
        },
        
        "Product Explorers": {
            "description": "High variety shoppers who experiment with many products",
            "key_characteristics": [
                "Product Variety: High (Top 25%)",
                "Return Product Variety: High",
                "Return Rate: Moderate (Experimentation fails)",
                "Customer Lifetime: Medium-High",
                "Return Timing: Variable"
            ],
            "business_value": "High",
            "risk_level": "Medium",
            "marketing_action": "Product discovery, early access, personalized recommendations",
            "expected_size": "20-30% of returning customers",
            "priority": "Medium - Growth",
            "kpis_to_track": ["Product Discovery Rate", "Category Expansion", "Recommendation CTR"]
        },
        
        "Bulk Returners": {
            "description": "Customers who return many items per order in batches",
            "key_characteristics": [
                "Returns per Order: High (5+)",
                "Return Intensity: >0.8 (Full returns)",
                "Return Timing: Clustered dates",
                "Order Size: Large",
                "Return Behavior: Systematic"
            ],
            "business_value": "Medium",
            "risk_level": "Medium-High",
            "marketing_action": "Order size limits, return policy review, education",
            "expected_size": "5-15% of returning customers",
            "priority": "Medium - Policy Review",
            "kpis_to_track": ["Return Volume", "Order Size", "Return Patterns"]
        },
        
        "Fast Full Returners": {
            "description": "Quick decision makers who return complete items rapidly",
            "key_characteristics": [
                "Days to Return: <7 days",
                "Return Intensity: >0.8 (Full returns)",
                "Return Rate: High",
                "Decision Speed: Fast",
                "Return Reason: Likely sizing/expectations"
            ],
            "business_value": "Medium",
            "risk_level": "Medium",
            "marketing_action": "Product education, sizing guides, expectation setting",
            "expected_size": "10-20% of returning customers",
            "priority": "Medium - Product Experience",
            "kpis_to_track": ["Return Speed", "Sizing Accuracy", "Product Satisfaction"]
        },
        
        "Complex High Volume": {
            "description": "High-value customers with high return rates - need special handling",
            "key_characteristics": [
                "Orders: High (Top 25%)",
                "Return Rate: High (Top 25%)",
                "Customer Value: High",
                "Behavior: Complex/Unpredictable",
                "Tenure: Usually long"
            ],
            "business_value": "High",
            "risk_level": "Medium",
            "marketing_action": "Personalized service, quality review, account management",
            "expected_size": "5-10% of returning customers",
            "priority": "High - Special Handling",
            "kpis_to_track": ["Customer Satisfaction", "Service Quality", "Retention"]
        },
        
        "New Heavy Returners": {
            "description": "Recently acquired customers with high early return rates",
            "key_characteristics": [
                "Customer Lifetime: <180 days",
                "Return Rate: >0.4",
                "Order Count: Low-Medium",
                "Return Timing: Early in relationship",
                "Risk: High churn potential"
            ],
            "business_value": "Medium",
            "risk_level": "High",
            "marketing_action": "Onboarding improvement, early intervention, expectation setting",
            "expected_size": "5-15% of returning customers",
            "priority": "High - Early Intervention",
            "kpis_to_track": ["Early Return Rate", "Onboarding Success", "90-day Retention"]
        },
        
        "Active Loyalists": {
            "description": "Recently engaged customers with controlled return behavior",
            "key_characteristics": [
                "Recent Orders: High (5+)",
                "Return Rate: <0.4 (Controlled)",
                "Engagement: High current activity",
                "Behavior: Balanced",
                "Trend: Positive"
            ],
            "business_value": "High",
            "risk_level": "Low",
            "marketing_action": "Loyalty rewards, referral programs, community building",
            "expected_size": "15-25% of returning customers",
            "priority": "Medium - Loyalty Building",
            "kpis_to_track": ["Engagement Score", "Referral Rate", "Loyalty Program Usage"]
        },
        
        "Low Volume High Returns": {
            "description": "Infrequent shoppers with high return rates - quality issues",
            "key_characteristics": [
                "Orders: <15 (Low volume)",
                "Return Rate: >0.5 (High)",
                "Purchase Frequency: Low",
                "Return Reason: Likely quality/fit",
                "Value: Low current, potential future"
            ],
            "business_value": "Low-Medium",
            "risk_level": "Medium",
            "marketing_action": "Quality investigation, product recommendations, education",
            "expected_size": "10-20% of returning customers",
            "priority": "Low-Medium - Quality Focus",
            "kpis_to_track": ["Product Quality Scores", "Return Reasons", "Purchase Frequency"]
        },
        
        "Churn Alert - Worsening Behavior": {
            "description": "Customers whose return behavior is getting worse over time",
            "key_characteristics": [
                "Recent vs Avg Ratio: >1.5 (Worsening)",
                "Return Trend: Increasing",
                "Recent Activity: May be declining",
                "Behavior Change: Negative shift",
                "Risk: Imminent churn"
            ],
            "business_value": "Medium-High",
            "risk_level": "Very High",
            "marketing_action": "Immediate intervention, root cause analysis, personal outreach",
            "expected_size": "5-10% of returning customers",
            "priority": "Critical - Immediate Action",
            "kpis_to_track": ["Behavior Trend", "Intervention Success", "Retention Rate"]
        },
        
        "Standard Balanced": {
            "description": "Customers with typical, balanced return behavior",
            "key_characteristics": [
                "All metrics: Around median values",
                "Return Rate: 0.3-0.5 (Moderate)",
                "Return Ratio: 0.2-0.4 (Moderate)",
                "Behavior: Predictable",
                "Risk: Low"
            ],
            "business_value": "Medium",
            "risk_level": "Low",
            "marketing_action": "Standard engagement, cross-sell, category expansion",
            "expected_size": "30-50% of returning customers",
            "priority": "Medium - Standard Programs",
            "kpis_to_track": ["Cross-sell Success", "Category Expansion", "Purchase Frequency"]
        }
    }

def export_comprehensive_analysis(data_file_path, output_filename=None):
    """
    Main function to run complete analysis and export to Excel
    
    Args:
        data_file_path (str): Path to the CSV data file
        output_filename (str, optional): Custom output filename
    """
    
    print("="*80)
    print("COMPREHENSIVE CUSTOMER CLUSTERING ANALYSIS EXPORT")
    print("="*80)
    
    # Load data
    print("📊 Loading data...")
    df = pd.read_csv(data_file_path)
    print(f"   Loaded {len(df):,} records")
    
    # Initialize analyzer
    print("🔧 Initializing clustering analysis...")
    analyzer = ReturnsClusteringAnalysis(df)
    
    # Prepare features
    print("🛠️  Preparing customer features...")
    customer_features = analyzer.prepare_customer_features()
    print(f"   Created features for {len(customer_features):,} customers")
    
    # Find optimal clusters
    print("🎯 Finding optimal number of clusters...")
    optimal_k, scores = analyzer.find_optimal_clusters(max_clusters=10)
    
    # Perform clustering
    print(f"🔍 Performing clustering with {optimal_k} clusters...")
    cluster_labels, cluster_centers = analyzer.perform_clustering(n_clusters=8)
    
    # Analyze clusters
    print("📈 Analyzing cluster characteristics...")
    cluster_summary, interpretations = analyzer.analyze_clusters()
    
    # Create comprehensive profiles
    print("📋 Creating comprehensive profiles...")
    feature_inventory = create_feature_inventory()
    cluster_profiles = create_cluster_profiles(analyzer, cluster_summary, interpretations)
    
    print("💾 Exporting to Excel...")
    
    # Sheet 1: Cluster Summary Table
    cluster_data = []
    for profile in cluster_profiles:
        row = {
            'Cluster_ID': profile['cluster_id'],
            'Cluster_Name': profile['name'],
            'Customer_Count': profile['size'],
            'Percentage': round(profile['percentage'], 1),
            'Why_It_Matters': 'Customer retention and value optimization',
            'Target_Actions': profile['action'],
            'Distinguishing_Features': ', '.join(profile['distinguishing_features'][:3])  # Top 3
        }
        
        # Add key feature values
        key_features = ['SALES_ORDER_NO_nunique', 'RETURN_RATE', 'RETURN_RATIO', 
                'RECENT_ORDERS', 'AVG_DAYS_TO_RETURN', 'SKU_nunique']
        
        if 'RECENT_VS_AVG_RATIO' in analyzer.customer_features.columns:
            key_features.append('RECENT_VS_AVG_RATIO')
        
        for feature in key_features:
            if feature in cluster_summary.columns:
                cluster_value = cluster_summary.loc[profile['cluster_id'], feature]
                
                # Create descriptive value
                if cluster_value >= analyzer.customer_features[feature].quantile(0.75):
                    row[f'{feature}_Level'] = 'HIGH'
                    row[f'{feature}_Value'] = round(cluster_value, 3)
                elif cluster_value <= analyzer.customer_features[feature].quantile(0.25):
                    row[f'{feature}_Level'] = 'LOW'
                    row[f'{feature}_Value'] = round(cluster_value, 3)
                else:
                    row[f'{feature}_Level'] = 'MODERATE'
                    row[f'{feature}_Value'] = round(cluster_value, 3)
        
        cluster_data.append(row)

    cluster_df = pd.DataFrame(cluster_data)

    # Sheet 2: Feature Dictionary
    feature_dict_data = []
    for feature_name, details in feature_inventory.items():
        feature_dict_data.append({
            'Feature_Name': feature_name,
            'Description': details['description'],
            'Calculation_Method': details['calculation'],
            'Business_Meaning': details['business_meaning'],
            'Expected_Range': details['expected_range'],
            'Used_In_Clustering': 'Yes' if details['used_in_clustering'] else 'No',
            'Threshold_Usage': ', '.join(details['used_in_thresholds']) if details['used_in_thresholds'] else 'None',
            'Data_Type': 'Continuous' if 'ratio' in feature_name.lower() or 'avg' in feature_name.lower() else 'Count'
        })

    feature_dict_df = pd.DataFrame(feature_dict_data)

    # Sheet 3: Detailed Cluster Metrics
    detailed_metrics = cluster_summary.copy()
    detailed_metrics.index.name = 'Cluster_ID'
    detailed_metrics = detailed_metrics.reset_index()

    # Add interpretation info
    detailed_metrics['Cluster_Type'] = detailed_metrics['Cluster_ID'].map(
        lambda x: interpretations[x]['type']
    )
    detailed_metrics['Recommended_Action'] = detailed_metrics['Cluster_ID'].map(
        lambda x: interpretations[x]['action']
    )

    # Sheet 4: Feature Statistics by Cluster
    feature_stats = analyzer.customer_features.groupby('CLUSTER').agg(['mean', 'std', 'min', 'max']).round(3)
    feature_stats.columns = [f'{col[0]}_{col[1]}' for col in feature_stats.columns]
    feature_stats.index.name = 'Cluster_ID'
    feature_stats = feature_stats.reset_index()

    # Sheet 5: Business Priority Matrix
    priority_data = []
    for profile in cluster_profiles:
        cluster_id = profile['cluster_id']
        cluster_data = cluster_summary.loc[cluster_id]
        
        # Calculate priority scores
        business_value_score = 0
        risk_score = 0
        opportunity_score = 0
        
        # Business Value (0-10)
        if cluster_data['AVG_ORDERS'] >= analyzer.customer_features['SALES_ORDER_NO_nunique'].quantile(0.9):
            business_value_score += 4
        elif cluster_data['AVG_ORDERS'] >= analyzer.customer_features['SALES_ORDER_NO_nunique'].quantile(0.75):
            business_value_score += 2
        
        if cluster_data['AVG_LIFETIME_DAYS'] >= 500:
            business_value_score += 2
        
        if profile['percentage'] >= 15:  # Large segment
            business_value_score += 2
        
        if cluster_data['AVG_RECENT_ORDERS'] >= 3:
            business_value_score += 2
        
        # Risk Score (0-10)
        if cluster_data['AVG_RETURN_RATE'] >= analyzer.customer_features['RETURN_RATE'].quantile(0.75):
            risk_score += 3
        
        if cluster_data['AVG_RECENT_ORDERS'] <= analyzer.customer_features['RECENT_ORDERS'].quantile(0.25):
            risk_score += 3
        
        if 'CHURN' in profile['name'] or 'HIGH RISK' in profile['name']:
            risk_score += 4
        
        # Opportunity Score (0-10)
        if 'VIP' in profile['name'] or 'CHAMPION' in profile['name']:
            opportunity_score += 4
        elif 'EXPLORER' in profile['name']:
            opportunity_score += 3
        elif 'STANDARD' in profile['name']:
            opportunity_score += 2
        
        if cluster_data['AVG_RECENT_ORDERS'] >= 2:
            opportunity_score += 2
        
        if business_value_score >= 6:
            opportunity_score += 2
        
        priority_data.append({
            'Cluster_ID': cluster_id,
            'Cluster_Name': profile['name'],
            'Customer_Count': profile['size'],
            'Percentage': profile['percentage'],
            'Business_Value_Score': min(business_value_score, 10),
            'Risk_Score': min(risk_score, 10),
            'Opportunity_Score': min(opportunity_score, 10),
            'Overall_Priority': round((business_value_score + risk_score + opportunity_score) / 3, 1),
            'Primary_Focus': 'Retention' if risk_score >= 6 else 'Growth' if opportunity_score >= 6 else 'Maintenance'
        })

    priority_df = pd.DataFrame(priority_data).sort_values('Overall_Priority', ascending=False)

    # Sheet 6: Customer Sample Lists
    sample_lists = {}
    for cluster_id in cluster_summary.index:
        cluster_customers = analyzer.get_cluster_customers(cluster_id)
        sample_size = min(100, len(cluster_customers))  # Max 100 samples per cluster
        sample_lists[f'Cluster_{cluster_id}_Samples'] = cluster_customers[:sample_size]

    # Pad lists to same length for DataFrame
    max_samples = max(len(samples) for samples in sample_lists.values())
    for key in sample_lists:
        while len(sample_lists[key]) < max_samples:
            sample_lists[key].append('')

    samples_df = pd.DataFrame(sample_lists)

    # Create Excel file with multiple sheets
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_filename is None:
        filename = f'customer_clustering_analysis_{timestamp}.xlsx'
    else:
        filename = f'{output_filename}_{timestamp}.xlsx'

    # Sheet 7: Expected Clusters Reference
    expected_clusters = create_expected_clusters_reference()
    expected_clusters_data = []
    
    for cluster_name, details in expected_clusters.items():
        expected_clusters_data.append({
            'Cluster_Name': cluster_name,
            'Description': details['description'],
            'Key_Characteristics': ' | '.join(details['key_characteristics']),
            'Business_Value': details['business_value'],
            'Risk_Level': details['risk_level'],
            'Marketing_Action': details['marketing_action'],
            'Expected_Size': details['expected_size'],
            'Priority': details['priority'],
            'KPIs_to_Track': ' | '.join(details['kpis_to_track']),
            'Found_in_Current_Analysis': 'Yes' if any(cluster_name.upper() in interp['type'].upper() 
                        for interp in interpretations.values()) else 'No'
        })
    
    expected_clusters_df = pd.DataFrame(expected_clusters_data)

    try:
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # Write each sheet
            cluster_df.to_excel(writer, sheet_name='Cluster_Summary', index=False)
            feature_dict_df.to_excel(writer, sheet_name='Feature_Dictionary', index=False)
            detailed_metrics.to_excel(writer, sheet_name='Detailed_Metrics', index=False)
            feature_stats.to_excel(writer, sheet_name='Feature_Statistics', index=False)
            priority_df.to_excel(writer, sheet_name='Business_Priority', index=False)
            samples_df.to_excel(writer, sheet_name='Customer_Samples', index=False)
            expected_clusters_df.to_excel(writer, sheet_name='Expected_Clusters_Reference', index=False)
            
            # Add a summary sheet
            summary_data = {
                'Analysis_Summary': [
                    f'Analysis Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
                    f'Total Customers Analyzed: {len(analyzer.customer_features):,}',
                    f'Number of Clusters Found: {len(cluster_profiles)}',
                    f'Number of Possible Clusters: {len(expected_clusters)}',
                    f'Number of Features: {len(feature_inventory)}',
                    f'Optimal K (Silhouette): {optimal_k}',
                    f'Best Silhouette Score: {max(scores):.3f}',
                    '',
                    'Sheet Descriptions:',
                    '- Cluster_Summary: Main business overview of each cluster',
                    '- Feature_Dictionary: Complete definition of all features used',
                    '- Detailed_Metrics: Statistical breakdown by cluster',
                    '- Feature_Statistics: Mean, std, min, max for each feature by cluster',
                    '- Business_Priority: Priority scoring for marketing focus',
                    '- Customer_Samples: Sample customer emails for each cluster',
                    '- Expected_Clusters_Reference: All possible cluster types and characteristics',
                    '- Analysis_Summary: Overview and metadata'
                ]
            }
            
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Analysis_Summary', index=False)

        print(f"✅ Excel file created: {filename}")
        print(f"\nFile contains {len(cluster_profiles)} clusters found out of {len(expected_clusters)} possible types")
        print(f"Sheets included:")
        print("  📊 Cluster_Summary - Main business overview")
        print("  📚 Feature_Dictionary - Complete feature definitions") 
        print("  📈 Detailed_Metrics - Statistical breakdown")
        print("  📊 Feature_Statistics - Feature stats by cluster")
        print("  🎯 Business_Priority - Priority scoring")
        print("  👥 Customer_Samples - Sample emails per cluster")
        print("  📋 Expected_Clusters_Reference - All possible cluster types")
        print("  📋 Analysis_Summary - Overview and metadata")

        # Also create a simple CSV for quick reference
        csv_filename = f'cluster_summary_{timestamp}.csv'
        cluster_df.to_csv(csv_filename, index=False)
        print(f"\n📄 Quick reference CSV also created: {csv_filename}")
        
        return filename, csv_filename
        
    except Exception as e:
        print(f"❌ Error creating Excel file: {e}")
        # Fallback to CSV only
        csv_filename = f'cluster_summary_{timestamp}.csv'
        cluster_df.to_csv(csv_filename, index=False)
        print(f"📄 Fallback CSV created: {csv_filename}")
        return None, csv_filename
        
    except Exception as e:
        print(f"❌ Error creating Excel file: {e}")
        # Fallback to CSV only
        csv_filename = f'cluster_summary_{timestamp}.csv'
        cluster_df.to_csv(csv_filename, index=False)
        print(f"📄 Fallback CSV created: {csv_filename}")
        return None, csv_filename

def run_simple_export(data_file_path, output_filename=None):
    """
    Simplified export function for basic cluster summary
    
    Args:
        data_file_path (str): Path to the CSV data file
        output_filename (str, optional): Custom output filename
    """
    
    print("🚀 Running simplified clustering analysis...")
    
    # Load and analyze
    df = pd.read_csv(data_file_path)
    analyzer = ReturnsClusteringAnalysis(df)
    customer_features = analyzer.prepare_customer_features()
    
    # Quick clustering with 5 clusters
    cluster_labels, cluster_centers = analyzer.perform_clustering(n_clusters=5)
    cluster_summary, interpretations = analyzer.analyze_clusters()

    # Create simple export
    export_data = []
    for cluster_id in cluster_summary.index:
        profile = cluster_summary.loc[cluster_id]
        
        export_data.append({
            'Cluster_ID': cluster_id,
            'Cluster_Name': interpretations[cluster_id]['type'],
            'Customer_Count': int(profile['CUSTOMER_COUNT']),
            'Percentage': round((int(profile['CUSTOMER_COUNT']) / len(analyzer.customer_features)) * 100, 1),
            'Target_Actions': interpretations[cluster_id]['action'],
            'Avg_Orders': round(profile['AVG_ORDERS'], 1),
            'Return_Rate': round(profile['AVG_RETURN_RATE'], 3),
            'Return_Ratio': round(profile['AVG_RETURN_RATIO'], 3),
            'Recent_Orders': round(profile['AVG_RECENT_ORDERS'], 1),
            'Product_Variety': round(profile['AVG_PRODUCT_VARIETY'], 0),
            'Days_to_Return': round(profile['AVG_DAYS_TO_RETURN'], 1)
        })

    # Export
    df_export = pd.DataFrame(export_data)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if output_filename is None:
        filename = f'customer_clustering_simple_{timestamp}.xlsx'
    else:
        filename = f'{output_filename}_{timestamp}.xlsx'

    try:
        df_export.to_excel(filename, index=False)
        print(f"✅ Simple Excel file created: {filename}")
    except Exception as e:
        print(f"❌ Excel error: {e}")
        csv_filename = filename.replace('.xlsx', '.csv')
        df_export.to_csv(csv_filename, index=False)
        print(f"📄 CSV fallback created: {csv_filename}")
        filename = csv_filename
    
    print("\nCluster Summary:")
    print(df_export)
    
    return filename

if __name__ == "__main__":
    export_comprehensive_analysis('returns_data.csv', output_filename='comprehensive_data.csv')