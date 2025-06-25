"""
Excel Business Report Generator for Customer Clustering Results
Creates a comprehensive Excel report with multiple sheets containing clustering analysis
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime
import logging
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

from db import get_connection

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ClusteringExcelReport:
    def __init__(self, db_path="customer_clustering.db"):
        self.db_path = db_path
        self.conn = get_connection(db_path)
        self.report_data = {}
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
    def load_clustering_data(self):
        """Load all necessary data for the Excel report"""
        logger.info("Loading clustering data from database...")
        
        try:
            # Load customer clustering results
            self.report_data['customer_results'] = self.conn.execute("""
                SELECT 
                    cr.customer_emailid,
                    cr.final_cluster_id,
                    cr.cluster_quality_score,
                    cr.clustering_timestamp,
                    sf.sales_order_no_nunique,
                    sf.sku_nunique,
                    sf.items_returned_count,
                    sf.return_rate,
                    sf.customer_lifetime_days,
                    sf.avg_days_to_return,
                    sf.avg_order_size,
                    sf.return_product_variety,
                    sf.category_diversity_score,
                    sf.seasonal_susceptibility_orders,
                    sf.recent_vs_avg_ratio,
                    sf.behavior_stability_score,
                    gp.outlier_score,
                    gp.feature_completeness_score
                FROM clustering_results cr
                LEFT JOIN silver_customer_features sf ON cr.customer_emailid = sf.customer_emailid
                LEFT JOIN gold_cluster_processed gp ON cr.customer_emailid = gp.customer_emailid
                WHERE cr.clustering_method = 'DBSCAN_KMEANS_SUBDBSCAN';
            """).fetchdf()
            
            # Load cluster summary
            self.report_data['cluster_summary'] = self.conn.execute("""
                SELECT * FROM cluster_summary 
                WHERE clustering_method = 'DBSCAN_KMEANS_SUBDBSCAN'
                ORDER BY customer_count DESC;
            """).fetchdf()
            
            # Load feature statistics
            self.report_data['feature_stats'] = self.get_feature_statistics()
            
            # Load data quality metrics
            self.report_data['data_quality'] = self.get_data_quality_metrics()
            
            # Load execution metadata
            self.report_data['execution_metadata'] = self.get_execution_metadata()
            
            logger.info(f"Loaded data for {len(self.report_data['customer_results'])} customers across {len(self.report_data['cluster_summary'])} clusters")
            
        except Exception as e:
            logger.error(f"Failed to load clustering data: {e}")
            raise
    
    def get_feature_statistics(self):
        """Calculate comprehensive feature statistics"""
        logger.info("Calculating feature statistics...")
        
        # Get all scaled features from gold layer
        scaled_features_query = """
        SELECT * FROM gold_cluster_processed 
        WHERE customer_emailid IN (
            SELECT customer_emailid FROM clustering_results 
            WHERE clustering_method = 'DBSCAN_KMEANS_SUBDBSCAN'
        );
        """
        
        gold_data = self.conn.execute(scaled_features_query).fetchdf()
        
        # Get original features from silver layer
        silver_features_query = """
        SELECT * FROM silver_customer_features 
        WHERE customer_emailid IN (
            SELECT customer_emailid FROM clustering_results 
            WHERE clustering_method = 'DBSCAN_KMEANS_SUBDBSCAN'
        );
        """
        
        silver_data = self.conn.execute(silver_features_query).fetchdf()
        
        # Define feature categories
        feature_categories = {
            'Volume Metrics': [
                'sales_order_no_nunique', 'sku_nunique', 'items_returned_count',
                'sales_qty_mean', 'avg_order_size'
            ],
            'Return Behavior': [
                'return_rate', 'return_ratio', 'return_product_variety',
                'avg_returns_per_order', 'return_frequency_ratio', 'return_intensity'
            ],
            'Temporal Patterns': [
                'customer_lifetime_days', 'avg_days_to_return', 'return_timing_spread'
            ],
            'Recency Analysis': [
                'recent_orders', 'recent_returns', 'recent_vs_avg_ratio',
                'behavior_stability_score'
            ],
            'Category Intelligence': [
                'category_diversity_score', 'category_loyalty_score',
                'high_return_category_affinity'
            ],
            'Adjacency Patterns': [
                'sku_adjacency_orders', 'sku_adjacency_returns',
                'sku_adjacency_timing', 'sku_adjacency_return_timing'
            ],
            'Seasonal Trends': [
                'seasonal_susceptibility_orders', 'seasonal_susceptibility_returns'
            ],
            'Trend Analysis': [
                'trend_product_category_order_rate', 'trend_product_category_return_rate'
            ]
        }
        
        # Calculate statistics for each feature
        feature_stats = []
        
        for category, features in feature_categories.items():
            for feature in features:
                if feature in silver_data.columns:
                    stats = {
                        'Feature': feature,
                        'Category': category,
                        'Count': silver_data[feature].count(),
                        'Mean': silver_data[feature].mean(),
                        'Std': silver_data[feature].std(),
                        'Min': silver_data[feature].min(),
                        'Q25': silver_data[feature].quantile(0.25),
                        'Median': silver_data[feature].median(),
                        'Q75': silver_data[feature].quantile(0.75),
                        'Max': silver_data[feature].max(),
                        'Missing_Count': silver_data[feature].isnull().sum(),
                        'Missing_Percent': silver_data[feature].isnull().mean() * 100,
                        'Outliers_Count': ((silver_data[feature] < silver_data[feature].quantile(0.01)) | 
                                         (silver_data[feature] > silver_data[feature].quantile(0.99))).sum()
                    }
                    feature_stats.append(stats)
        
        return pd.DataFrame(feature_stats)
    
    def get_data_quality_metrics(self):
        """Get comprehensive data quality metrics"""
        logger.info("Calculating data quality metrics...")
        
        quality_metrics = []
        
        # Bronze layer quality
        bronze_quality = self.conn.execute("""
            SELECT 
                'Bronze Layer' as layer,
                count(*) as total_records,
                count(DISTINCT customer_emailid) as unique_customers,
                count(DISTINCT sales_order_no) as unique_orders,
                sum(CASE WHEN data_quality_flags != '' THEN 1 ELSE 0 END) as records_with_flags,
                sum(CASE WHEN return_qty > 0 THEN 1 ELSE 0 END) as records_with_returns
            FROM bronze_return_order_data;
        """).fetchone()
        
        quality_metrics.append({
            'Layer': bronze_quality[0],
            'Total_Records': bronze_quality[1],
            'Unique_Customers': bronze_quality[2],
            'Unique_Orders': bronze_quality[3],
            'Records_With_Flags': bronze_quality[4],
            'Records_With_Returns': bronze_quality[5],
            'Data_Quality_Score': (1 - bronze_quality[4] / bronze_quality[1]) * 100 if bronze_quality[1] > 0 else 0
        })
        
        # Silver layer quality
        silver_quality = self.conn.execute("""
            SELECT 
                'Silver Layer' as layer,
                count(*) as customer_count,
                avg(feature_completeness_score) as avg_completeness,
                count(*) FILTER (WHERE return_rate > 1.0 OR return_rate < 0) as invalid_return_rates,
                count(*) FILTER (WHERE customer_lifetime_days < 0) as invalid_lifetime_days
            FROM silver_customer_features;
        """).fetchone()
        
        quality_metrics.append({
            'Layer': silver_quality[0],
            'Customer_Count': silver_quality[1],
            'Avg_Completeness': silver_quality[2],
            'Invalid_Return_Rates': silver_quality[3],
            'Invalid_Lifetime_Days': silver_quality[4],
            'Data_Quality_Score': silver_quality[2] * 100 if silver_quality[2] else 0
        })
        
        # Gold layer quality
        gold_quality = self.conn.execute("""
            SELECT 
                'Gold Layer' as layer,
                count(*) as processed_customers,
                avg(feature_completeness_score) as avg_completeness,
                count(*) FILTER (WHERE data_quality_flags = '' OR data_quality_flags IS NULL) as clean_customers,
                count(*) FILTER (WHERE outlier_score < -0.5) as extreme_outliers
            FROM gold_cluster_processed;
        """).fetchone()
        
        quality_metrics.append({
            'Layer': gold_quality[0],
            'Processed_Customers': gold_quality[1],
            'Avg_Completeness': gold_quality[2],
            'Clean_Customers': gold_quality[3],
            'Extreme_Outliers': gold_quality[4],
            'Data_Quality_Score': (gold_quality[3] / gold_quality[1]) * 100 if gold_quality[1] > 0 else 0
        })
        
        return pd.DataFrame(quality_metrics)
    
    def get_execution_metadata(self):
        """Load execution metadata from various sources"""
        logger.info("Loading execution metadata...")
        
        metadata = {
            'pipeline_info': {},
            'clustering_config': {},
            'performance_metrics': {}
        }
        
        # Try to load from JSON files
        json_files = ['clustering_metadata.json', 'preprocessing_metadata.json']
        
        for json_file in json_files:
            try:
                if Path(json_file).exists():
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                        metadata[json_file.replace('.json', '')] = data
                        logger.info(f"Loaded metadata from {json_file}")
            except Exception as e:
                logger.warning(f"Could not load {json_file}: {e}")
        
        # Get basic execution info from database
        try:
            execution_info = self.conn.execute("""
                SELECT 
                    count(*) as total_customers_clustered,
                    count(DISTINCT final_cluster_id) as clusters_created,
                    max(clustering_timestamp) as last_clustering_run,
                    avg(cluster_quality_score) as avg_quality_score
                FROM clustering_results 
                WHERE clustering_method = 'DBSCAN_KMEANS_SUBDBSCAN';
            """).fetchone()
            
            metadata['execution_summary'] = {
                'total_customers_clustered': execution_info[0],
                'clusters_created': execution_info[1],
                'last_clustering_run': execution_info[2],
                'avg_quality_score': execution_info[3],
                'report_generation_time': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.warning(f"Could not load execution info: {e}")
            metadata['execution_summary'] = {'error': str(e)}
        
        return metadata
    
    def create_executive_summary_sheet(self):
        """Create executive summary sheet data"""
        logger.info("Creating executive summary...")
        
        customer_results = self.report_data['customer_results']
        cluster_summary = self.report_data['cluster_summary']
        
        # Calculate key metrics
        total_customers = len(customer_results)
        clustered_customers = len(customer_results[customer_results['final_cluster_id'] != -1])
        noise_customers = total_customers - clustered_customers
        num_clusters = len(cluster_summary)
        
        # Cluster size distribution
        cluster_sizes = customer_results[customer_results['final_cluster_id'] != -1]['final_cluster_id'].value_counts().sort_index()
        
        # Key business metrics
        avg_return_rate = customer_results['return_rate'].mean()
        avg_lifetime_days = customer_results['customer_lifetime_days'].mean()
        avg_order_count = customer_results['sales_order_no_nunique'].mean()
        
        # Create summary data
        summary_data = [
            ['Metric', 'Value', 'Description'],
            ['Total Customers Analyzed', f"{total_customers:,}", 'All customers in the analysis'],
            ['Customers Successfully Clustered', f"{clustered_customers:,}", f'{clustered_customers/total_customers:.1%} of total'],
            ['Outlier/Noise Customers', f"{noise_customers:,}", f'{noise_customers/total_customers:.1%} of total'],
            ['Number of Clusters Created', num_clusters, 'Distinct customer segments identified'],
            ['', '', ''],
            ['BUSINESS METRICS', '', ''],
            ['Average Return Rate', f"{avg_return_rate:.1%}", 'Proportion of items returned'],
            ['Average Customer Lifetime (days)', f"{avg_lifetime_days:.0f}", 'Days from first to last order'],
            ['Average Orders per Customer', f"{avg_order_count:.1f}", 'Order frequency measure'],
            ['', '', ''],
            ['CLUSTER SIZE DISTRIBUTION', '', ''],
        ]
        
        # Add cluster sizes
        for cluster_id in sorted(cluster_sizes.index):
            size = cluster_sizes[cluster_id]
            percentage = size / clustered_customers * 100
            summary_data.append([f'Cluster {cluster_id}', f"{size:,}", f'{percentage:.1f}% of clustered customers'])
        
        return pd.DataFrame(summary_data)
    
    def create_cluster_profiles_sheet(self):
        """Create detailed cluster profiles"""
        logger.info("Creating cluster profiles...")
        
        customer_results = self.report_data['customer_results']
        cluster_summary = self.report_data['cluster_summary']
        
        # Enhanced cluster profiles with business interpretation
        profiles = []
        
        for _, cluster in cluster_summary.iterrows():
            cluster_id = cluster['cluster_id']
            cluster_customers = customer_results[customer_results['final_cluster_id'] == cluster_id]
            
            if len(cluster_customers) == 0:
                continue
            
            # Calculate detailed statistics
            profile = {
                'Cluster_ID': cluster_id,
                'Customer_Count': len(cluster_customers),
                'Percentage_of_Total': len(cluster_customers) / len(customer_results[customer_results['final_cluster_id'] != -1]) * 100,
                
                # Core metrics
                'Avg_Return_Rate': cluster_customers['return_rate'].mean(),
                'Median_Return_Rate': cluster_customers['return_rate'].median(),
                'Avg_Order_Count': cluster_customers['sales_order_no_nunique'].mean(),
                'Avg_Lifetime_Days': cluster_customers['customer_lifetime_days'].mean(),
                'Avg_Order_Size': cluster_customers['avg_order_size'].mean(),
                
                # Advanced metrics
                'Avg_Product_Variety': cluster_customers['sku_nunique'].mean(),
                'Avg_Return_Variety': cluster_customers['return_product_variety'].mean(),
                'Avg_Days_to_Return': cluster_customers['avg_days_to_return'].mean(),
                'Avg_Category_Diversity': cluster_customers['category_diversity_score'].mean(),
                'Avg_Behavior_Stability': cluster_customers['behavior_stability_score'].mean(),
                
                # Quality metrics
                'Avg_Quality_Score': cluster_customers['cluster_quality_score'].mean(),
                'Avg_Completeness_Score': cluster_customers['feature_completeness_score'].mean(),
                'Outlier_Customers': (cluster_customers['outlier_score'] < -0.5).sum(),
            }
            
            # Business archetype determination
            return_rate = profile['Avg_Return_Rate']
            order_count = profile['Avg_Order_Count']
            lifetime_days = profile['Avg_Lifetime_Days']
            
            if return_rate > 0.4:
                archetype = "High Returners"
                strategy = "Focus on return reason analysis and quality improvement"
            elif return_rate < 0.1 and order_count > 20:
                archetype = "Loyal Customers"
                strategy = "Retention programs and premium service offerings"
            elif lifetime_days > 730 and order_count > 15:
                archetype = "Veteran Shoppers"
                strategy = "Loyalty rewards and exclusive experiences"
            elif lifetime_days < 180:
                archetype = "New Customers"
                strategy = "Onboarding optimization and early engagement"
            elif order_count > 30:
                archetype = "Frequent Buyers"
                strategy = "Volume incentives and personalized recommendations"
            else:
                archetype = "Regular Customers"
                strategy = "Balanced engagement and satisfaction monitoring"
            
            profile['Business_Archetype'] = archetype
            profile['Recommended_Strategy'] = strategy
            
            profiles.append(profile)
        
        return pd.DataFrame(profiles)
    
    def create_feature_importance_sheet(self):
        """Create feature importance and usage analysis"""
        logger.info("Creating feature importance analysis...")
        
        feature_stats = self.report_data['feature_stats']
        customer_results = self.report_data['customer_results']
        
        # Calculate feature importance based on cluster discrimination
        feature_importance = []
        
        # For features available in customer_results, calculate cluster discrimination
        numeric_columns = customer_results.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            if col in ['final_cluster_id', 'outlier_score', 'feature_completeness_score']:
                continue
                
            try:
                # Calculate variance across clusters
                cluster_means = customer_results.groupby('final_cluster_id')[col].mean()
                overall_mean = customer_results[col].mean()
                
                # Calculate between-cluster variance (simplified F-statistic concept)
                between_cluster_var = ((cluster_means - overall_mean) ** 2).mean()
                within_cluster_var = customer_results[col].var()
                
                importance_score = between_cluster_var / within_cluster_var if within_cluster_var > 0 else 0
                
                feature_importance.append({
                    'Feature': col,
                    'Importance_Score': importance_score,
                    'Between_Cluster_Variance': between_cluster_var,
                    'Within_Cluster_Variance': within_cluster_var,
                    'Overall_Mean': overall_mean,
                    'Overall_Std': customer_results[col].std(),
                    'Used_in_Clustering': 'Yes' if col in feature_stats['Feature'].values else 'No'
                })
                
            except Exception as e:
                logger.warning(f"Could not calculate importance for {col}: {e}")
        
        importance_df = pd.DataFrame(feature_importance)
        if len(importance_df) > 0:
            importance_df = importance_df.sort_values('Importance_Score', ascending=False)
        
        return importance_df
    
    def create_customer_assignments_sheet(self):
        """Create customer-to-cluster assignments"""
        logger.info("Creating customer assignments...")
        
        customer_results = self.report_data['customer_results']
        
        # Add cluster archetype
        cluster_profiles = self.create_cluster_profiles_sheet()
        archetype_map = dict(zip(cluster_profiles['Cluster_ID'], cluster_profiles['Business_Archetype']))
        
        customer_assignments = customer_results.copy()
        customer_assignments['Business_Archetype'] = customer_assignments['final_cluster_id'].map(archetype_map)
        customer_assignments['Is_Outlier'] = customer_assignments['final_cluster_id'] == -1
        customer_assignments['Cluster_Assignment_Date'] = customer_assignments['clustering_timestamp']
        
        # Reorder columns for better readability
        column_order = [
            'customer_emailid', 'final_cluster_id', 'Business_Archetype', 'Is_Outlier',
            'cluster_quality_score', 'return_rate', 'sales_order_no_nunique', 
            'customer_lifetime_days', 'avg_order_size', 'items_returned_count',
            'outlier_score', 'feature_completeness_score', 'Cluster_Assignment_Date'
        ]
        
        available_columns = [col for col in column_order if col in customer_assignments.columns]
        return customer_assignments[available_columns]
    
    def create_quality_metrics_sheet(self):
        """Create clustering quality and validation metrics"""
        logger.info("Creating quality metrics...")
        
        # Combine different quality metrics
        quality_data = []
        
        # Data quality metrics
        data_quality = self.report_data['data_quality']
        for _, row in data_quality.iterrows():
            quality_data.append({
                'Metric_Category': 'Data Quality',
                'Metric_Name': f"{row['Layer']} Quality Score",
                'Value': f"{row['Data_Quality_Score']:.1f}%",
                'Description': f"Overall data quality for {row['Layer']}"
            })
        
        # Clustering performance metrics
        customer_results = self.report_data['customer_results']
        
        # Silhouette score (if available in metadata)
        metadata = self.report_data['execution_metadata']
        if 'clustering_metadata' in metadata:
            clustering_meta = metadata['clustering_metadata']
            if 'silhouette_score' in clustering_meta:
                quality_data.append({
                    'Metric_Category': 'Clustering Quality',
                    'Metric_Name': 'Silhouette Score',
                    'Value': f"{clustering_meta['silhouette_score']:.3f}",
                    'Description': 'Measure of cluster separation quality (higher is better)'
                })
        
        # Basic clustering metrics
        total_customers = len(customer_results)
        clustered_customers = len(customer_results[customer_results['final_cluster_id'] != -1])
        
        quality_data.extend([
            {
                'Metric_Category': 'Coverage',
                'Metric_Name': 'Customer Coverage Rate',
                'Value': f"{clustered_customers/total_customers:.1%}",
                'Description': 'Percentage of customers successfully clustered'
            },
            {
                'Metric_Category': 'Quality',
                'Metric_Name': 'Average Cluster Quality Score',
                'Value': f"{customer_results['cluster_quality_score'].mean():.3f}",
                'Description': 'Average quality score across all customers'
            },
            {
                'Metric_Category': 'Completeness',
                'Metric_Name': 'Average Feature Completeness',
                'Value': f"{customer_results['feature_completeness_score'].mean():.3f}",
                'Description': 'Average completeness of customer features'
            }
        ])
        
        return pd.DataFrame(quality_data)
    
    def create_technical_details_sheet(self):
        """Create technical implementation details"""
        logger.info("Creating technical details...")
        
        technical_details = []
        
        # Feature categories and counts
        feature_stats = self.report_data['feature_stats']
        category_counts = feature_stats['Category'].value_counts()
        
        technical_details.append(['FEATURE ENGINEERING', '', ''])
        technical_details.append(['Category', 'Feature Count', 'Description'])
        
        for category, count in category_counts.items():
            technical_details.append([category, count, f'{count} features in {category} category'])
        
        technical_details.append(['', '', ''])
        technical_details.append(['CLUSTERING ALGORITHM', '', ''])
        technical_details.append(['Method', 'Hybrid DBSCAN → K-means → Sub-DBSCAN', 'Three-phase clustering approach'])
        technical_details.append(['Phase 1', 'Initial DBSCAN', 'Outlier detection and density analysis'])
        technical_details.append(['Phase 2', 'K-means Optimization', 'Main cluster identification'])
        technical_details.append(['Phase 3', 'Sub-clustering DBSCAN', 'Cluster refinement and micro-segmentation'])
        
        # Add metadata if available
        metadata = self.report_data['execution_metadata']
        if 'clustering_metadata' in metadata:
            config = metadata['clustering_metadata'].get('clustering_config', {})
            
            technical_details.append(['', '', ''])
            technical_details.append(['CONFIGURATION PARAMETERS', '', ''])
            
            if 'kmeans' in config:
                technical_details.append(['K-means Clusters', config['kmeans'].get('optimal_k', 'N/A'), 'Optimal number of clusters identified'])
            
            if 'dbscan_initial' in config:
                technical_details.append(['Initial DBSCAN eps', config['dbscan_initial'].get('eps', 'N/A'), 'Distance threshold for initial clustering'])
                technical_details.append(['Initial DBSCAN min_samples', config['dbscan_initial'].get('min_samples', 'N/A'), 'Minimum samples for core points'])
        
        # Add execution summary
        if 'execution_summary' in metadata:
            exec_summary = metadata['execution_summary']
            technical_details.append(['', '', ''])
            technical_details.append(['EXECUTION SUMMARY', '', ''])
            technical_details.append(['Report Generation Time', exec_summary.get('report_generation_time', 'N/A'), 'When this report was generated'])
            technical_details.append(['Last Clustering Run', exec_summary.get('last_clustering_run', 'N/A'), 'Most recent clustering execution'])
        
        return pd.DataFrame(technical_details, columns=['Parameter', 'Value', 'Description'])
    
    def generate_excel_report(self, output_file=None):
        """Generate the complete Excel report"""
        if output_file is None:
            output_file = f'clustering_results_{self.timestamp}.xlsx'
        
        logger.info(f"Generating Excel report: {output_file}")
        
        try:
            # Load all data
            self.load_clustering_data()
            
            # Create all sheets
            sheets = {
                'Executive Summary': self.create_executive_summary_sheet(),
                'Cluster Profiles': self.create_cluster_profiles_sheet(),
                'Customer Assignments': self.create_customer_assignments_sheet(),
                'Feature Statistics': self.report_data['feature_stats'],
                'Feature Importance': self.create_feature_importance_sheet(),
                'Quality Metrics': self.create_quality_metrics_sheet(),
                'Data Quality': self.report_data['data_quality'],
                'Technical Details': self.create_technical_details_sheet()
            }
            
            # Write to Excel
            with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                for sheet_name, sheet_data in sheets.items():
                    logger.info(f"Writing sheet: {sheet_name}")
                    sheet_data.to_excel(writer, sheet_name=sheet_name, index=False)
                    
                    # Auto-adjust column widths
                    worksheet = writer.sheets[sheet_name]
                    for column in worksheet.columns:
                        max_length = 0
                        column_letter = column[0].column_letter
                        for cell in column:
                            try:
                                if len(str(cell.value)) > max_length:
                                    max_length = len(str(cell.value))
                            except:
                                pass
                        adjusted_width = min(max_length + 2, 50)  # Cap at 50 characters
                        worksheet.column_dimensions[column_letter].width = adjusted_width
            
            logger.info(f"✅ Excel report generated successfully: {output_file}")
            
            # Generate summary report
            self.print_report_summary(output_file, sheets)
            
            return output_file
            
        except Exception as e:
            logger.error(f"Failed to generate Excel report: {e}")
            raise
        
        finally:
            self.conn.close()
    
    def print_report_summary(self, output_file, sheets):
        """Print a summary of the generated report"""
        print("\n" + "="*80)
        print("EXCEL REPORT GENERATION SUMMARY")
        print("="*80)
        
        print(f"📊 Report File: {output_file}")
        print(f"📅 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📋 Total Sheets: {len(sheets)}")
        
        print(f"\n📈 CLUSTERING RESULTS:")
        if 'Customer Assignments' in sheets:
            customer_data = sheets['Customer Assignments']
            total_customers = len(customer_data)
            clustered_customers = len(customer_data[customer_data['final_cluster_id'] != -1])
            num_clusters = len(customer_data[customer_data['final_cluster_id'] != -1]['final_cluster_id'].unique())
            
            print(f"  Total Customers: {total_customers:,}")
            print(f"  Clustered Customers: {clustered_customers:,} ({clustered_customers/total_customers:.1%})")
            print(f"  Clusters Created: {num_clusters}")
        
        print(f"\n📊 SHEET CONTENTS:")
        for sheet_name, sheet_data in sheets.items():
            print(f"  {sheet_name:20}: {len(sheet_data):,} rows × {len(sheet_data.columns)} columns")
        
        print(f"\n🎯 BUSINESS VALUE:")
        print(f"  ✅ Customer segments identified for targeted strategies")
        print(f"  ✅ Return behavior patterns analyzed for optimization")
        print(f"  ✅ Data quality metrics provided for ongoing monitoring")
        print(f"  ✅ Technical details documented for reproducibility")
        
        print(f"\n📋 NEXT STEPS:")
        print(f"  1. Review cluster profiles with business stakeholders")
        print(f"  2. Validate customer archetypes against business knowledge")
        print(f"  3. Implement cluster-specific strategies and campaigns")
        print(f"  4. Monitor cluster stability and performance over time")
        
        print("="*80)

def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate Excel report for customer clustering results")
    parser.add_argument("--db-path", default="customer_clustering.db", help="Path to DuckDB database")
    parser.add_argument("--output", help="Output Excel file name (default: clustering_results_TIMESTAMP.xlsx)")
    
    args = parser.parse_args()
    
    try:
        # Check if database exists
        if not Path(args.db_path).exists():
            logger.error(f"Database not found: {args.db_path}")
            logger.error("Please run the clustering pipeline first to generate results")
            sys.exit(1)
        
        # Generate report
        report_generator = ClusteringExcelReport(args.db_path)
        output_file = report_generator.generate_excel_report(args.output)
        
        print(f"\n🎉 Excel report generated successfully!")
        print(f"📊 File: {output_file}")
        print(f"💼 Ready for business stakeholder review and implementation")
        
        return True
        
    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
