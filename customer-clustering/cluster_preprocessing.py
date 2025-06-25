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
    def __init__(self, scaling_method: str = 'robust', correlation_threshold: float = 0.8):
        """
        Initialize the cluster preprocessor
        
        Args:
            scaling_method: 'robust', 'standard', or 'minmax'
            correlation_threshold: Threshold for identifying highly correlated features (0.0-1.0)
        """
        self.scaling_method = scaling_method
        self.correlation_threshold = correlation_threshold
        self.scaler = None
        self.feature_stats = {}
        self.outlier_detector = None
        self.selected_features = []
        self.removed_null_features = []
        self.removed_correlated_features = []
        
        logger.info(f"Initialized ClusterPreprocessor with {scaling_method} scaling and correlation threshold {correlation_threshold}")
    
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
                
                # Use explicit columns for insert if specified
                if hasattr(self, 'explicit_column_insert') and self.explicit_column_insert:
                    # Get column names as a comma-separated string
                    column_names = ', '.join(df_insert.columns)
                    logger.info(f"Using explicit column insert with {len(df_insert.columns)} columns")
                    conn.execute(f"INSERT INTO gold_cluster_processed ({column_names}) SELECT {column_names} FROM chunk")
                else:
                    # Traditional insert (requires exact column match)
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
            report.append(f"Original features: {len(self.feature_stats.get('original_features', []))}")
            report.append(f"Selected features: {len(self.feature_stats['selected_features'])}")
            
            # Report on removed features
            if hasattr(self, 'removed_null_features') and self.removed_null_features:
                report.append(f"Features with 100% nulls removed: {len(self.removed_null_features)}")
                
            if hasattr(self, 'removed_correlated_features') and self.removed_correlated_features:
                report.append(f"Highly correlated features removed: {len(self.removed_correlated_features)}")
                
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
    
    def remove_null_features(self, df: pd.DataFrame, features: List[str]) -> Tuple[pd.DataFrame, List[str]]:
        """Remove features with 100% null values"""
        
        logger.info("Checking for features with 100% null values...")
        
        # Find features that are completely null
        completely_null_features = []
        for col in features:
            if df[col].isnull().sum() == len(df):
                completely_null_features.append(col)
        
        self.removed_null_features = completely_null_features
        
        if completely_null_features:
            logger.info(f"Removing {len(completely_null_features)} features with 100% null values:")
            for feature in completely_null_features:
                logger.info(f"  - {feature}")
            
            # Remove completely null features
            valid_features = [f for f in features if f not in completely_null_features]
            
            logger.info(f"Features reduced from {len(features)} to {len(valid_features)}")
        else:
            logger.info("No features with 100% null values found.")
            valid_features = features
            
        return df, valid_features
    
    def remove_correlated_features(self, df: pd.DataFrame, features: List[str]) -> Tuple[pd.DataFrame, List[str]]:
        """Remove highly correlated features to reduce multicollinearity"""
        
        logger.info(f"Analyzing feature correlations (threshold: {self.correlation_threshold})...")
        
        # Calculate correlation matrix
        correlation_matrix = df[features].corr().abs()
        
        # Create a set to keep track of features to remove
        features_to_remove = set()
        
        # List of highly correlated feature pairs
        high_correlation_pairs = []
        
        # Identify pairs of highly correlated features
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                if correlation_matrix.iloc[i, j] > self.correlation_threshold:
                    feat_i = correlation_matrix.columns[i]
                    feat_j = correlation_matrix.columns[j]
                    correlation = correlation_matrix.iloc[i, j]
                    high_correlation_pairs.append((feat_i, feat_j, correlation))
        
        if high_correlation_pairs:
            logger.info(f"Found {len(high_correlation_pairs)} highly correlated feature pairs (|correlation| > {self.correlation_threshold}):")
            
            # Print highly correlated pairs
            for feat_i, feat_j, corr in high_correlation_pairs:
                logger.info(f"  {feat_i} <-> {feat_j}: {corr:.3f}")
            
            # Count occurrence of each feature in high correlation pairs
            feature_counts = {}
            for feat_i, feat_j, _ in high_correlation_pairs:
                feature_counts[feat_i] = feature_counts.get(feat_i, 0) + 1
                feature_counts[feat_j] = feature_counts.get(feat_j, 0) + 1
            
            # For each pair, remove the feature that appears more frequently
            logger.info("Features selected for removal:")
            for feat_i, feat_j, _ in high_correlation_pairs:
                # Skip if both features are already marked for removal
                if feat_i in features_to_remove and feat_j in features_to_remove:
                    continue
                    
                # If one is already marked, skip
                if feat_i in features_to_remove:
                    continue
                if feat_j in features_to_remove:
                    continue
                    
                # Otherwise, remove the one with higher count
                if feature_counts[feat_i] > feature_counts[feat_j]:
                    features_to_remove.add(feat_i)
                    logger.info(f"  - {feat_i} (appears in {feature_counts[feat_i]} pairs)")
                else:
                    features_to_remove.add(feat_j)
                    logger.info(f"  - {feat_j} (appears in {feature_counts[feat_j]} pairs)")
            
            self.removed_correlated_features = list(features_to_remove)
            
            # Remove the identified features
            valid_features = [f for f in features if f not in features_to_remove]
            
            logger.info(f"Features reduced from {len(features)} to {len(valid_features)}")
        else:
            logger.info("No highly correlated feature pairs found.")
            valid_features = features
            self.removed_correlated_features = []
            
        return df, valid_features

    def process_data(self, conn) -> pd.DataFrame:
        """Run the full preprocessing pipeline"""
        
        logger.info("Starting cluster preprocessing pipeline...")
        
        # Load silver layer data
        silver_df = self.load_silver_data(conn)
        original_features = self.select_features(silver_df)
        
        # Store original feature list
        self.feature_stats = {'original_features': original_features}
        
        # First, check for and remove 100% null features
        silver_df, features_no_nulls = self.remove_null_features(silver_df, original_features)
        
        # Next, check for and remove highly correlated features
        silver_df, uncorrelated_features = self.remove_correlated_features(silver_df, features_no_nulls)
        
        # Continue with the rest of the preprocessing pipeline
        silver_df, selected_features = self.remove_low_variance_features(silver_df, uncorrelated_features)
        silver_df, outlier_scores = self.detect_outliers(silver_df, selected_features)
        gold_df, scaling_stats = self.scale_features(silver_df, selected_features)
        gold_df = self.create_gold_layer_dataset(gold_df, selected_features, scaling_stats, original_features)
        
        # Insert into gold layer table
        success = self.insert_gold_layer_data(conn, gold_df)
        
        # Generate preprocessing report
        report = self.generate_preprocessing_report(conn)
        
        # Write report to file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = f'preprocessing_report_{timestamp}.txt'
        with open(report_path, 'w') as f:
            f.write('\n'.join(report))
        
        logger.info(f"Preprocessing report written to {report_path}")
        
        if success:
            logger.info("Cluster preprocessing pipeline completed successfully")
        else:
            logger.error("Cluster preprocessing pipeline failed")
        
        return gold_df

    # Advanced Feature Selection Methods
    def advanced_feature_selection(self, df: pd.DataFrame, features: List[str], 
                            min_importance: float = 0.01,
                            min_variance: float = 0.01,
                            max_vif: float = 10.0,
                            max_stability: float = 0.5,
                            min_iv: float = 0.02) -> Tuple[pd.DataFrame, List[str], Dict]:
        """
        Apply advanced feature selection methods to further optimize the feature set
        
        Args:
            df: DataFrame with features
            features: List of feature names
            min_importance: Minimum feature importance to retain (Random Forest)
            min_variance: Minimum variance to retain
            max_vif: Maximum Variance Inflation Factor allowed
            max_stability: Maximum stability metric allowed (lower is better)
            min_iv: Minimum Information Value required
            
        Returns:
            Tuple of (DataFrame, selected_features, selection_metadata)
        """
        
        logger.info("Applying advanced feature selection methods...")
        
        # Create output directory for visualizations
        import os
        output_dir = "feature_selection_outputs"
        os.makedirs(output_dir, exist_ok=True)
        
        # Set up tracking for features flagged by each method
        flagged_features = {
            'high_vif': [],
            'low_variance': [],
            'low_importance': [],
            'unstable': [],
            'low_iv': []
        }
        
        # Create feature matrix
        feature_matrix = df[features]
        X = feature_matrix.values
        
        # 1. Variance Inflation Factor (VIF) Analysis
        try:
            from statsmodels.stats.outliers_influence import variance_inflation_factor
            
            logger.info(f"Running VIF analysis (max_vif={max_vif})...")
            
            # Calculate VIF for each feature
            vif_data = pd.DataFrame()
            vif_data["Feature"] = features
            vif_data["VIF"] = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
            
            # Sort by highest VIF
            vif_data = vif_data.sort_values("VIF", ascending=False)
            
            # Identify features with high VIF
            high_vif_features = vif_data[vif_data["VIF"] > max_vif]["Feature"].tolist()
            if high_vif_features:
                logger.info(f"Found {len(high_vif_features)} features with VIF > {max_vif}:")
                for feature in high_vif_features:
                    logger.info(f"  - {feature} (VIF: {vif_data.loc[vif_data['Feature'] == feature, 'VIF'].values[0]:.3f})")
                
                flagged_features['high_vif'] = high_vif_features
            else:
                logger.info(f"No features with VIF > {max_vif} detected")
                
            # Save VIF results
            vif_data.to_csv(f"{output_dir}/vif_analysis.csv", index=False)
            
        except ImportError:
            logger.warning("statsmodels not installed. Skipping VIF analysis...")
        
        # 2. Low-Variance Feature Filtering
        logger.info(f"Running variance analysis (min_variance={min_variance})...")
        
        # Calculate variance for each feature
        variance_data = pd.DataFrame({
            "Feature": features,
            "Variance": [np.var(feature_matrix[col]) for col in features]
        })
        
        # Sort by variance
        variance_data = variance_data.sort_values("Variance")
        
        # Identify low variance features
        low_variance_features = variance_data[variance_data["Variance"] < min_variance]["Feature"].tolist()
        
        if low_variance_features:
            logger.info(f"Found {len(low_variance_features)} features with low variance (< {min_variance}):")
            for feature in low_variance_features:
                logger.info(f"  - {feature} (variance: {variance_data.loc[variance_data['Feature'] == feature, 'Variance'].values[0]:.6f})")
            
            flagged_features['low_variance'] = low_variance_features
        else:
            logger.info(f"No features with variance below {min_variance} detected")
        
        # 3. Feature Importance using Random Forest
        logger.info(f"Running feature importance analysis (min_importance={min_importance})...")
        
        # We'll use feature completeness score as a proxy target
        if 'feature_completeness_score' in df.columns:
            y = df['feature_completeness_score'].values
        else:
            # Create a simple target based on data completeness
            non_zero_counts = (X != 0).sum(axis=1)
            y = non_zero_counts / X.shape[1]
        
        # Train a random forest model
        from sklearn.ensemble import RandomForestRegressor
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X, y)
        
        # Get feature importances
        importances = rf.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        # Create DataFrame for importances
        importance_df = pd.DataFrame({
            'Feature': [features[i] for i in indices],
            'Importance': importances[indices]
        })
        
        # Identify low importance features
        low_importance_features = importance_df[importance_df['Importance'] < min_importance]['Feature'].tolist()
        
        if low_importance_features:
            logger.info(f"Found {len(low_importance_features)} features with low importance (< {min_importance}):")
            for feature in low_importance_features:
                logger.info(f"  - {feature} (importance: {importance_df.loc[importance_df['Feature'] == feature, 'Importance'].values[0]:.6f})")
            
            flagged_features['low_importance'] = low_importance_features
        else:
            logger.info(f"No features with importance below {min_importance} detected")
        
        # 4. Feature Stability Analysis
        logger.info(f"Running feature stability analysis (max_stability={max_stability})...")
        
        from sklearn.model_selection import KFold
        
        # Set up K-fold cross-validation
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        feature_importance_cv = np.zeros((len(features), 5))
        
        # Run Random Forest on each fold and collect feature importances
        for i, (train_idx, test_idx) in enumerate(kf.split(X)):
            X_train, y_train = X[train_idx], y[train_idx]
            
            # Train Random Forest
            rf = RandomForestRegressor(n_estimators=50, random_state=42+i)
            rf.fit(X_train, y_train)
            
            # Store importances
            feature_importance_cv[:, i] = rf.feature_importances_
        
        # Calculate stability metric (coefficient of variation = std/mean)
        mean_importance = np.mean(feature_importance_cv, axis=1)
        std_importance = np.std(feature_importance_cv, axis=1)
        stability = std_importance / (mean_importance + 1e-10)  # Add small constant to avoid division by zero
        
        # Create DataFrame for stability analysis
        stability_df = pd.DataFrame({
            'Feature': features,
            'Mean_Importance': mean_importance,
            'Std_Importance': std_importance,
            'Stability': stability
        })
        
        # Identify unstable features
        unstable_features = stability_df[stability_df['Stability'] > max_stability]['Feature'].tolist()
        
        if unstable_features:
            logger.info(f"Found {len(unstable_features)} unstable features (stability > {max_stability}):")
            for feature in unstable_features:
                feature_data = stability_df.loc[stability_df['Feature'] == feature]
                logger.info(f"  - {feature} (stability: {feature_data['Stability'].values[0]:.4f})")
            
            flagged_features['unstable'] = unstable_features
        else:
            logger.info(f"No unstable features detected (stability > {max_stability})")
        
        # 5. Information Value (IV) Analysis
        logger.info(f"Running Information Value analysis (min_iv={min_iv})...")
        
        # Create a simple binary target for IV demonstration
        if 'outlier_flag' in df.columns:
            binary_target = df['outlier_flag'].astype(int)
        else:
            # Use feature completeness as a proxy
            binary_target = (df['feature_completeness_score'] > df['feature_completeness_score'].median()).astype(int)
        
        # Helper function for IV calculation
        def calculate_woe_iv(feature, target):
            """Calculate Weight of Evidence (WoE) and Information Value (IV) for a feature"""
            df = pd.DataFrame({'feature': feature, 'target': target})
            
            # Handle numeric features by binning them
            if np.issubdtype(feature.dtype, np.number):
                df['feature_bin'] = pd.qcut(feature, q=5, duplicates='drop')
            else:
                df['feature_bin'] = feature
            
            # Calculate counts and rates
            grouped = df.groupby('feature_bin')['target'].agg(['count', 'sum'])
            grouped.columns = ['total', 'event']
            grouped['non_event'] = grouped['total'] - grouped['event']
            
            # Calculate percentages
            grouped['event_pct'] = grouped['event'] / grouped['event'].sum()
            grouped['non_event_pct'] = grouped['non_event'] / grouped['non_event'].sum()
            
            # Calculate WoE and IV with handling for edge cases
            grouped['woe'] = np.log(np.maximum(grouped['event_pct'], 1e-10) / np.maximum(grouped['non_event_pct'], 1e-10))
            grouped['iv'] = (grouped['event_pct'] - grouped['non_event_pct']) * grouped['woe']
            
            # Return the total IV
            return grouped['iv'].sum()
        
        # Calculate IV for each feature
        iv_values = {}
        for col in features:
            iv_values[col] = calculate_woe_iv(feature_matrix[col], binary_target)
        
        # Create IV DataFrame
        iv_df = pd.DataFrame({
            'Feature': list(iv_values.keys()),
            'IV': list(iv_values.values())
        }).sort_values('IV', ascending=False)
        
        # Identify low IV features
        low_iv_features = iv_df[iv_df['IV'] < min_iv]['Feature'].tolist()
        
        if low_iv_features:
            logger.info(f"Found {len(low_iv_features)} features with low Information Value (< {min_iv}):")
            for feature in low_iv_features:
                logger.info(f"  - {feature} (IV: {iv_df.loc[iv_df['Feature'] == feature, 'IV'].values[0]:.4f})")
            
            flagged_features['low_iv'] = low_iv_features
        else:
            logger.info(f"No features with IV below {min_iv} detected")
        
        # 6. Consolidate results and create optimized feature recommendations
        logger.info("Consolidating feature selection recommendations...")
        
        # Count how many methods flagged each feature
        feature_flags = {}
        for feature in features:
            feature_flags[feature] = sum(1 for method, flagged in flagged_features.items() 
                                      if feature in flagged)
        
        # Create summary DataFrame
        summary_df = pd.DataFrame({
            'Feature': list(feature_flags.keys()),
            'Removal_Flags': list(feature_flags.values())
        }).sort_values('Removal_Flags', ascending=False)
        
        # Select features for removal (flagged by 2+ methods)
        removal_candidates = summary_df[summary_df['Removal_Flags'] >= 2]['Feature'].tolist()
        
        if removal_candidates:
            logger.info(f"Found {len(removal_candidates)} features flagged by multiple selection methods:")
            for feature in removal_candidates:
                flags = []
                for method, flagged in flagged_features.items():
                    if feature in flagged:
                        flags.append(method)
                logger.info(f"  - {feature}: {feature_flags[feature]} flags ({', '.join(flags)})")
            
            # Remove the identified features
            optimized_features = [f for f in features if f not in removal_candidates]
            
            logger.info(f"Advanced feature selection reduced features from {len(features)} to {len(optimized_features)}")
        else:
            logger.info("No features were flagged by multiple selection methods - current feature set appears optimal")
            optimized_features = features
        
        # Create metadata about the selection process
        selection_metadata = {
            'original_features': features,
            'optimized_features': optimized_features,
            'removed_features': removal_candidates,
            'flagged_features': flagged_features,
            'timestamp': datetime.now().isoformat(),
            'parameters': {
                'min_importance': min_importance,
                'min_variance': min_variance,
                'max_vif': max_vif,
                'max_stability': max_stability,
                'min_iv': min_iv
            }
        }
        
        # Save metadata
        import json
        with open(f"{output_dir}/advanced_feature_selection.json", 'w') as f:
            json.dump(selection_metadata, f, indent=2, default=str)
        
        logger.info(f"Advanced feature selection results saved to {output_dir}/advanced_feature_selection.json")
        
        return df, optimized_features, selection_metadata
    
def main(scaling_method: str = 'robust', 
         variance_threshold: float = 0.01,
         outlier_contamination: float = 0.05,
         explicit_column_insert: bool = False,
         advanced_selection: bool = False):
    """
    Main preprocessing pipeline execution
    
    Args:
        scaling_method: 'robust' or 'standard'
        variance_threshold: Minimum variance for feature selection
        outlier_contamination: Expected proportion of outliers
        explicit_column_insert: Use explicit column names when inserting data
        advanced_selection: Apply advanced feature selection methods
    """
    
    logger.info("Starting cluster preprocessing pipeline...")
    logger.info(f"Parameters: scaling={scaling_method}, variance_threshold={variance_threshold}, contamination={outlier_contamination}")
    if explicit_column_insert:
        logger.info("Using explicit column insert mode to handle schema mismatches")
    
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
        
        # Set explicit column insert flag if specified
        if explicit_column_insert:
            preprocessor.explicit_column_insert = explicit_column_insert
        
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
        
        # Apply advanced feature selection if enabled
        if advanced_selection:
            logger.info("Applying advanced feature selection...")
            
            # Check if we have the required packages
            try:
                import statsmodels
                has_statsmodels = True
            except ImportError:
                logger.warning("statsmodels not installed. Some advanced selection methods will be limited.")
                has_statsmodels = False
            
            # Apply advanced selection with parameters from arguments
            clustering_df, advanced_features, selection_metadata = preprocessor.advanced_feature_selection(
                clustering_df, 
                high_variance_features,
                min_importance=0.01,
                min_variance=0.01,
                max_vif=10.0,
                max_stability=0.5,
                min_iv=0.02
            )
            
            # Use the advanced optimized feature set
            high_variance_features = advanced_features
            logger.info(f"Advanced feature selection completed. Using {len(advanced_features)} features.")
            
            # Store advanced selection metadata
            preprocessor.advanced_selection_metadata = selection_metadata
        
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
    parser.add_argument("--explicit-column-insert", choices=['true', 'false'], default='false',
                       help="Use explicit column names when inserting data (default: false)")
    
    # Add advanced feature selection options
    parser.add_argument("--advanced-selection", choices=['true', 'false'], default='false',
                       help="Enable advanced feature selection techniques (default: false)")
    parser.add_argument("--min-importance", type=float, default=0.01,
                       help="Minimum feature importance to retain in Random Forest (default: 0.01)")
    parser.add_argument("--min-variance", type=float, default=0.01,
                       help="Minimum variance to retain features (default: 0.01)")
    parser.add_argument("--max-vif", type=float, default=10.0,
                       help="Maximum Variance Inflation Factor allowed (default: 10.0)")
    parser.add_argument("--max-stability", type=float, default=0.5,
                       help="Maximum stability metric allowed, lower is better (default: 0.5)")
    parser.add_argument("--min-iv", type=float, default=0.02,
                       help="Minimum Information Value required (default: 0.02)")
    parser.add_argument("--advanced-selection", choices=['true', 'false'], default='false',
                       help="Apply advanced feature selection methods (default: false)")
    parser.add_argument("--min-importance", type=float, default=0.01,
                       help="Minimum feature importance to retain (Random Forest, default: 0.01)")
    parser.add_argument("--min-variance", type=float, default=0.01,
                       help="Minimum variance to retain (default: 0.01)")
    parser.add_argument("--max-vif", type=float, default=10.0,
                       help="Maximum Variance Inflation Factor allowed (default: 10.0)")
    parser.add_argument("--max-stability", type=float, default=0.5,
                       help="Maximum stability metric allowed (default: 0.5)")
    parser.add_argument("--min-iv", type=float, default=0.02,
                       help="Minimum Information Value required (default: 0.02)")
    
    args = parser.parse_args()
    
    success = main(
        scaling_method=args.scaling,
        variance_threshold=args.variance_threshold,
        outlier_contamination=args.outlier_contamination,
        explicit_column_insert=args.explicit_column_insert == 'true',
        advanced_selection=args.advanced_selection == 'true'
    )
    
    sys.exit(0 if success else 1)
