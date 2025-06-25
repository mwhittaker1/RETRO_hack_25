"""
Cluster Preprocessing Pipeline
Prepares silver layer data for clustering by handling scaling, outliers, and feature selection
"""

import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import IsolationForest
import warnings
from typing import Dict, List, Tuple, Optional
import sys
from datetime import datetime

from db import get_connection

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('cluster_preprocessing.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class ClusterPreprocessor:
    def __init__(self, scaling_method: str = 'robust'):
        """
        Initialize the cluster preprocessor
        
        Args:
            scaling_method: 'robust', 'standard', or 'minmax'
        """
        self.scaling_method = scaling_method
        self.scaler = None
        self.feature_stats = {}
        self.outlier_detector = None
        self.selected_features = []
        
        logger.info(f"Initialized ClusterPreprocessor with {scaling_method} scaling")
    
    def load_silver_data(self, conn) -> pd.DataFrame:
        """Load and prepare data from silver layer"""
        
        logger.info("Loading data from silver layer...")
        
        query = """
        SELECT * FROM silver_customer_features
        WHERE customer_emailid IS NOT NULL
        AND sales_order_no_nunique > 0;  -- Ensure valid customers
        """
        
        df = conn.execute(query).fetchdf()
        logger.info(f"Loaded {len(df)} customers with {len(df.columns)} features")
        
        return df
    
    def select_clustering_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Select appropriate features for clustering"""
        
        logger.info("Selecting features for clustering...")
        
        # Define feature categories and selection criteria
        feature_groups = {
            'volume_metrics': [
                'sales_order_no_nunique', 'sku_nunique', 'items_returned_count',
                'sales_qty_mean', 'avg_order_size'
            ],
            'return_behavior': [
                'return_rate', 'return_ratio', 'return_product_variety',
                'avg_returns_per_order', 'return_frequency_ratio', 'return_intensity'
            ],
            'temporal_patterns': [
                'customer_lifetime_days', 'avg_days_to_return', 'return_timing_spread'
            ],
            'recency_trends': [
                'recent_orders', 'recent_returns', 'recent_vs_avg_ratio',
                'behavior_stability_score'
            ],
            'category_intelligence': [
                'category_diversity_score', 'category_loyalty_score',
                'high_return_category_affinity'
            ],
            'adjacency_patterns': [
                'sku_adjacency_orders', 'sku_adjacency_returns',
                'sku_adjacency_timing', 'sku_adjacency_return_timing'
            ],
            'seasonal_trends': [
                'seasonal_susceptibility_orders', 'seasonal_susceptibility_returns'
            ],
            'trend_susceptibility': [
                'trend_product_category_order_rate', 'trend_product_category_return_rate'
            ]
        }
        
        # Start with core features that are most reliable
        core_features = (
            feature_groups['volume_metrics'] + 
            feature_groups['return_behavior'] + 
            feature_groups['temporal_patterns'] +
            feature_groups['recency_trends']
        )
        
        # Add advanced features that have sufficient data
        advanced_features = []
        
        # Check category features
        for feature in feature_groups['category_intelligence']:
            if feature in df.columns:
                non_zero_pct = (df[feature] != 0).mean()
                if non_zero_pct > 0.1:  # At least 10% of customers have non-zero values
                    advanced_features.append(feature)
                    logger.info(f"Including {feature}: {non_zero_pct:.1%} customers have data")
                else:
                    logger.info(f"Excluding {feature}: only {non_zero_pct:.1%} customers have data")
        
        # Check adjacency features
        for feature in feature_groups['adjacency_patterns']:
            if feature in df.columns:
                non_zero_pct = (df[feature] != 0).mean()
                if non_zero_pct > 0.05:  # At least 5% for adjacency (more sparse)
                    advanced_features.append(feature)
                    logger.info(f"Including {feature}: {non_zero_pct:.1%} customers have data")
        
        # Check seasonal features (only for customers with >2 years history)
        for feature in feature_groups['seasonal_trends']:
            if feature in df.columns:
                non_zero_pct = (df[feature] != 0).mean()
                if non_zero_pct > 0.05:
                    advanced_features.append(feature)
                    logger.info(f"Including {feature}: {non_zero_pct:.1%} customers have data")
        
        # Check trend features
        for feature in feature_groups['trend_susceptibility']:
            if feature in df.columns:
                non_zero_pct = (df[feature] != 0).mean()
                if non_zero_pct > 0.1:
                    advanced_features.append(feature)
                    logger.info(f"Including {feature}: {non_zero_pct:.1%} customers have data")
        
        # Combine all selected features
        selected_features = []
        for feature in core_features + advanced_features:
            if feature in df.columns:
                selected_features.append(feature)
            else:
                logger.warning(f"Feature {feature} not found in dataset")
        
        # Create clustering dataset
        clustering_df = df[['customer_emailid'] + selected_features].copy()
        
        logger.info(f"Selected {len(selected_features)} features for clustering:")
        for group, features in feature_groups.items():
            group_features = [f for f in features if f in selected_features]
            logger.info(f"  {group}: {len(group_features)} features")
        
        return clustering_df, selected_features
    
    def handle_missing_values(self, df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
        """Handle missing values in selected features"""
        
        logger.info("Handling missing values...")
        
        df_clean = df.copy()
        
        for feature in features:
            missing_count = df_clean[feature].isnull().sum()
            if missing_count > 0:
                missing_pct = missing_count / len(df_clean) * 100
                logger.info(f"Feature {feature}: {missing_count} missing values ({missing_pct:.1f}%)")
                
                # Fill with appropriate defaults based on feature type
                if 'rate' in feature.lower() or 'ratio' in feature.lower():
                    fill_value = 0.0
                elif 'count' in feature.lower() or 'nunique' in feature.lower():
                    fill_value = 0
                elif 'days' in feature.lower():
                    fill_value = df_clean[feature].median()
                elif 'score' in feature.lower():
                    fill_value = df_clean[feature].median()
                else:
                    fill_value = 0.0
                
                df_clean[feature] = df_clean[feature].fillna(fill_value)
                logger.info(f"  Filled with {fill_value}")
        
        return df_clean
    
    def remove_low_variance_features(self, df: pd.DataFrame, features: List[str], 
                                   variance_threshold: float = 0.01) -> Tuple[pd.DataFrame, List[str]]:
        """Remove features with very low variance"""
        
        logger.info(f"Removing low variance features (threshold: {variance_threshold})...")
        
        # Calculate variance for each feature
        feature_variance = {}
        for feature in features:
            variance = df[feature].var()
            feature_variance[feature] = variance
        
        # Filter features
        high_variance_features = [f for f in features if feature_variance[f] >= variance_threshold]
        removed_features = [f for f in features if f not in high_variance_features]
        
        if removed_features:
            logger.info(f"Removed {len(removed_features)} low variance features:")
            for feature in removed_features:
                logger.info(f"  {feature}: variance = {feature_variance[feature]:.6f}")
        
        clustering_features = ['customer_emailid'] + high_variance_features
        return df[clustering_features], high_variance_features
    
    def detect_outliers(self, df: pd.DataFrame, features: List[str], 
                       contamination: float = 0.05) -> Tuple[pd.DataFrame, np.ndarray]:
        """Detect outliers using Isolation Forest"""
        
        logger.info(f"Detecting outliers using Isolation Forest (contamination: {contamination})...")
        
        # Prepare data for outlier detection
        X = df[features].values
        
        # Fit Isolation Forest
        self.outlier_detector = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_jobs=-1
        )
        
        outlier_predictions = self.outlier_detector.fit_predict(X)
        outlier_scores = self.outlier_detector.score_samples(X)
        
        # Add outlier information to dataframe
        df_with_outliers = df.copy()
        df_with_outliers['outlier_flag'] = outlier_predictions == -1
        df_with_outliers['outlier_score'] = outlier_scores
        
        outlier_count = (outlier_predictions == -1).sum()
        logger.info(f"Detected {outlier_count} outliers ({outlier_count/len(df)*100:.1f}%)")
        
        # Log outlier characteristics
        if outlier_count > 0:
            outliers = df_with_outliers[df_with_outliers['outlier_flag']]
            logger.info("Outlier characteristics:")
            logger.info(f"  Avg return rate: {outliers['return_rate'].mean():.3f}")
            logger.info(f"  Avg order count: {outliers['sales_order_no_nunique'].mean():.1f}")
            logger.info(f"  Avg lifetime days: {outliers['customer_lifetime_days'].mean():.0f}")
        
        return df_with_outliers, outlier_scores
    
    def scale_features(self, df: pd.DataFrame, features: List[str]) -> Tuple[pd.DataFrame, Dict]:
        """Scale features using the specified scaling method"""
        
        logger.info(f"Scaling features using {self.scaling_method} scaler...")
        
        # Initialize scaler
        if self.scaling_method == 'robust':
            self.scaler = RobustScaler()
        elif self.scaling_method == 'standard':
            self.scaler = StandardScaler()
        else:
            raise ValueError(f"Unsupported scaling method: {self.scaling_method}")
        
        # Fit and transform features
        X = df[features].values
        X_scaled = self.scaler.fit_transform(X)
        
        # Create scaled dataframe
        df_scaled = df[['customer_emailid', 'outlier_flag', 'outlier_score']].copy()
        
        # Add scaled features
        for i, feature in enumerate(features):
            scaled_feature_name = f"{feature}_scaled"
            df_scaled[scaled_feature_name] = X_scaled[:, i]
        
        # Store scaling statistics
        scaling_stats = {}
        for i, feature in enumerate(features):
            if self.scaling_method == 'robust':
                scaling_stats[feature] = {
                    'median': self.scaler.center_[i],
                    'scale': self.scaler.scale_[i]
                }
            elif self.scaling_method == 'standard':
                scaling_stats[feature] = {
                    'mean': self.scaler.mean_[i],
                    'std': self.scaler.scale_[i]
                }
        
        logger.info(f"Scaled {len(features)} features")
        return df_scaled, scaling_stats
    
    def calculate_feature_completeness(self, df: pd.DataFrame, original_features: List[str]) -> float:
        """Calculate feature completeness score for each customer"""
        
        logger.info("Calculating feature completeness scores...")
        
        # Count non-zero/non-default values for each customer
        completeness_scores = []
        
        for _, row in df.iterrows():
            non_default_count = 0
            total_features = len(original_features)
            
            for feature in original_features:
                if feature in df.columns:
                    value = row[feature]
                    # Consider non-zero values as "complete" for most features
                    if pd.notna(value) and value != 0:
                        non_default_count += 1
            
            completeness_score = non_default_count / total_features if total_features > 0 else 0
            completeness_scores.append(completeness_score)
        
        avg_completeness = np.mean(completeness_scores)
        logger.info(f"Average feature completeness: {avg_completeness:.3f}")
        
        return completeness_scores
    
    def create_gold_layer_dataset(self, df_scaled: pd.DataFrame, features: List[str], 
                                 scaling_stats: Dict, original_features: List[str]) -> pd.DataFrame:
        """Create the final gold layer dataset"""
        
        logger.info("Creating gold layer dataset...")
        
        # Calculate feature completeness
        completeness_scores = self.calculate_feature_completeness(df_scaled, original_features)
        
        # Create final dataset
        gold_df = df_scaled.copy()
        gold_df['feature_completeness_score'] = completeness_scores
        gold_df['processing_timestamp'] = datetime.now()
        gold_df['scaling_method'] = self.scaling_method
        
        # Add data quality flags
        data_quality_flags = []
        
        # Flag low completeness customers
        low_completeness = np.array(completeness_scores) < 0.3
        if low_completeness.sum() > 0:
            data_quality_flags.extend(['LOW_COMPLETENESS'] * low_completeness.sum())
        
        # Flag extreme outliers
        extreme_outliers = gold_df['outlier_score'] < gold_df['outlier_score'].quantile(0.01)
        if extreme_outliers.sum() > 0:
            data_quality_flags.extend(['EXTREME_OUTLIER'] * extreme_outliers.sum())
        
        gold_df['data_quality_flags'] = ''
        if low_completeness.sum() > 0:
            gold_df.loc[low_completeness, 'data_quality_flags'] += 'LOW_COMPLETENESS;'
        if extreme_outliers.sum() > 0:
            gold_df.loc[extreme_outliers, 'data_quality_flags'] += 'EXTREME_OUTLIER;'
        
        # Store feature selection and scaling metadata
        self.feature_stats = {
            'selected_features': features,
            'scaling_stats': scaling_stats,
            'total_customers': len(gold_df),
            'outlier_count': gold_df['outlier_flag'].sum(),
            'avg_completeness': np.mean(completeness_scores),
            'processing_timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Gold layer dataset created: {len(gold_df)} customers, {len(gold_df.columns)} columns")
        return gold_df
    
    def insert_gold_layer_data(self, conn, gold_df: pd.DataFrame) -> bool:
        """Insert processed data into gold layer"""
        
        logger.info("Inserting data into gold_cluster_processed table...")
        
        try:
            # Clear existing data
            conn.execute("DELETE FROM gold_cluster_processed;")
            logger.info("Cleared existing gold layer data")
            
            # Get scaled feature columns
            scaled_feature_columns = [col for col in gold_df.columns if col.endswith('_scaled')]
            
            # Prepare data for insertion - only include columns that exist in the gold table
            gold_table_columns = [
                'customer_emailid', 'outlier_score', 'feature_completeness_score',
                'processing_timestamp', 'scaling_method', 'data_quality_flags'
            ] + scaled_feature_columns
            
            # Filter to only include columns that exist in both the dataframe and table
            available_columns = [col for col in gold_table_columns if col in gold_df.columns]
            
            # Hey Claude! Look here for the updated data that includes order values!
            # Add any new monetary feature columns here when they become available
            
            df_insert = gold_df[available_columns].copy()
            
            # Insert in chunks
            chunk_size = 5000
            total_inserted = 0
            
            for i in range(0, len(df_insert), chunk_size):
                chunk = df_insert[i:i+chunk_size]
                conn.execute("INSERT INTO gold_cluster_processed SELECT * FROM chunk")
                total_inserted += len(chunk)
                logger.info(f"Inserted chunk {i//chunk_size + 1}: {total_inserted}/{len(df_insert)} records")
            
            # Verify insertion
            final_count = conn.execute("SELECT count(*) FROM gold_cluster_processed").fetchone()[0]
            
            if final_count == len(df_insert):
                logger.info(f"✅ Successfully inserted {final_count} customer records into gold layer")
                return True
            else:
                logger.error(f"❌ Insertion mismatch: expected {len(df_insert)}, got {final_count}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to insert gold layer data: {str(e)}")
            return False
    
    def generate_preprocessing_report(self, conn) -> None:
        """Generate a comprehensive preprocessing report"""
        
        logger.info("Generating preprocessing report...")
        
        report = []
        report.append("="*60)
        report.append("CLUSTER PREPROCESSING SUMMARY REPORT")
        report.append("="*60)
        report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Scaling method: {self.scaling_method}")
        report.append("")
        
        # Feature selection summary
        if self.feature_stats:
            report.append("FEATURE SELECTION:")
            report.append("-" * 30)
            report.append(f"Selected features: {len(self.feature_stats['selected_features'])}")
            report.append(f"Total customers: {self.feature_stats['total_customers']:,}")
            report.append(f"Outliers detected: {self.feature_stats['outlier_count']:,}")
            report.append(f"Average completeness: {self.feature_stats['avg_completeness']:.3f}")
            report.append("")
        
        # Gold layer statistics
        gold_stats = conn.execute("""
            SELECT 
                count(*) as total_customers,
                avg(feature_completeness_score) as avg_completeness,
                count(*) FILTER (WHERE outlier_score < -0.5) as extreme_outliers,
                count(*) FILTER (WHERE data_quality_flags LIKE '%LOW_COMPLETENESS%') as low_completeness_customers,
                count(*) FILTER (WHERE data_quality_flags = '') as clean_customers
            FROM gold_cluster_processed;
        """).fetchone()
        
        report.append("GOLD LAYER STATISTICS:")
        report.append("-" * 30)
        report.append(f"Total customers: {gold_stats[0]:,}")
        report.append(f"Average completeness: {gold_stats[1]:.3f}")
        report.append(f"Extreme outliers: {gold_stats[2]:,}")
        report.append(f"Low completeness: {gold_stats[3]:,}")
        report.append(f"Clean customers: {gold_stats[4]:,}")
        report.append("")
        
        # Feature scaling summary
        if self.feature_stats and 'scaling_stats' in self.feature_stats:
            report.append("SCALING STATISTICS (sample):")
            report.append("-" * 30)
            sample_features = list(self.feature_stats['scaling_stats'].keys())[:5]
            for feature in sample_features:
                stats = self.feature_stats['scaling_stats'][feature]
                if self.scaling_method == 'robust':
                    report.append(f"{feature[:25]:25}: median={stats['median']:.3f}, scale={stats['scale']:.3f}")
                else:
                    report.append(f"{feature[:25]:25}: mean={stats['mean']:.3f}, std={stats['std']:.3f}")
        
        # Ready for clustering indicators
        ready_customers = conn.execute("""
            SELECT count(*) FROM gold_cluster_processed 
            WHERE data_quality_flags = '' OR data_quality_flags IS NULL;
        """).fetchone()[0]
        
        report.append("")
        report.append("CLUSTERING READINESS:")
        report.append("-" * 30)
        report.append(f"Customers ready for clustering: {ready_customers:,}")
        report.append(f"Percentage ready: {ready_customers/gold_stats[0]*100:.1f}%")
        report.append("")
        report.append("RECOMMENDED NEXT STEPS:")
        report.append("- Review outliers for potential exclusion from clustering")
        report.append("- Consider feature selection refinement based on variance")
        report.append("- Proceed with DBSCAN -> K-means -> sub-DBSCAN pipeline")
        report.append("")
        report.append("="*60)
        
        # Print and save report
        report_text = "\n".join(report)
        print(report_text)
        
        with open(f"preprocessing_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt", 'w') as f:
            f.write(report_text)
        
        logger.info("Preprocessing report generated and saved")

def main(scaling_method: str = 'robust', 
         variance_threshold: float = 0.01,
         outlier_contamination: float = 0.05):
    """
    Main preprocessing pipeline execution
    
    Args:
        scaling_method: 'robust' or 'standard'
        variance_threshold: Minimum variance for feature selection
        outlier_contamination: Expected proportion of outliers
    """
    
    logger.info("Starting cluster preprocessing pipeline...")
    logger.info(f"Parameters: scaling={scaling_method}, variance_threshold={variance_threshold}, contamination={outlier_contamination}")
    
    try:
        # Connect to database
        conn = get_connection("customer_clustering.db")
        logger.info("Connected to customer clustering database")
        
        # Check silver layer data availability
        silver_count = conn.execute("SELECT count(*) FROM silver_customer_features").fetchone()[0]
        if silver_count == 0:
            logger.error("No data found in silver layer. Please run create_features.py first.")
            return False
        
        logger.info(f"Silver layer contains {silver_count:,} customer records")
        
        # Initialize preprocessor
        preprocessor = ClusterPreprocessor(scaling_method)
        
        # Load silver layer data
        silver_df = preprocessor.load_silver_data(conn)
        
        # Select features for clustering
        clustering_df, selected_features = preprocessor.select_clustering_features(silver_df)
        
        # Handle missing values
        clustering_df = preprocessor.handle_missing_values(clustering_df, selected_features)
        
        # Remove low variance features
        clustering_df, high_variance_features = preprocessor.remove_low_variance_features(
            clustering_df, selected_features, variance_threshold
        )
        
        # Detect outliers
        outlier_df, outlier_scores = preprocessor.detect_outliers(
            clustering_df, high_variance_features, outlier_contamination
        )
        
        # Scale features
        scaled_df, scaling_stats = preprocessor.scale_features(outlier_df, high_variance_features)
        
        # Create gold layer dataset
        gold_df = preprocessor.create_gold_layer_dataset(
            scaled_df, high_variance_features, scaling_stats, selected_features
        )
        
        # Insert into gold layer
        success = preprocessor.insert_gold_layer_data(conn, gold_df)
        
        if not success:
            logger.error("Failed to insert data into gold layer")
            return False
        
        # Generate preprocessing report
        preprocessor.generate_preprocessing_report(conn)
        
        logger.info("✅ Cluster preprocessing pipeline completed successfully!")
        
        # Save preprocessing metadata for clustering pipeline
        metadata = {
            'feature_stats': preprocessor.feature_stats,
            'scaling_method': scaling_method,
            'selected_features': high_variance_features,
            'preprocessing_timestamp': datetime.now().isoformat()
        }
        
        import json
        with open('preprocessing_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info("Saved preprocessing metadata to preprocessing_metadata.json")
        return True
        
    except Exception as e:
        logger.error(f"Cluster preprocessing pipeline failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False
        
    finally:
        conn.close()

if __name__ == "__main__":
    # Parse command line arguments for customization
    import argparse
    
    parser = argparse.ArgumentParser(description="Cluster Preprocessing Pipeline")
    parser.add_argument("--scaling", choices=['robust', 'standard'], default='robust',
                       help="Scaling method (default: robust)")
    parser.add_argument("--variance-threshold", type=float, default=0.01,
                       help="Minimum variance threshold for feature selection (default: 0.01)")
    parser.add_argument("--outlier-contamination", type=float, default=0.05,
                       help="Expected outlier contamination rate (default: 0.05)")
    
    args = parser.parse_args()
    
    success = main(
        scaling_method=args.scaling,
        variance_threshold=args.variance_threshold,
        outlier_contamination=args.outlier_contamination
    )
    
    sys.exit(0 if success else 1)
