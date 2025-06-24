import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class ReturnsClusteringAnalysis:
    def __init__(self, df):
        """
        Initialize with your returns dataframe
        Expected columns: CUSTOMER_EMAILID, SALES_ORDER_NO, SKU, SALES_QTY, 
            RETURN_QTY, ORDER_DATE, RETURN_DATE, UNITS_RETURNED_FLAG
        Note: Each row represents a combined order-return record, not separate transactions
        """
        self.df = df.copy()
        self.customer_features = None
        self.clusters = None
        self.scaler = StandardScaler()
        
    def prepare_customer_features(self):
        """Create customer-level features for clustering from combined order-return data"""
        
        # Convert dates - handle Excel serial numbers for RETURN_DATE
        self.df['ORDER_DATE'] = pd.to_datetime(self.df['ORDER_DATE'])
                
        # Convert Excel serial numbers to dates for RETURN_DATE
        def convert_excel_date(date_val):
            if pd.isna(date_val) or date_val == '-':
                return pd.NaT
            try:
                # Convert Excel serial number to datetime
                # Excel serial date starts from 1900-01-01 (but Excel incorrectly treats 1900 as leap year)
                return pd.to_datetime('1899-12-30') + pd.Timedelta(days=float(date_val))
            except:
                return pd.NaT

        self.df['RETURN_DATE'] = self.df['RETURN_DATE'].apply(convert_excel_date)        
        
        print("Analyzing combined order-return data structure...")
        print(f"Total records: {len(self.df):,}")
        print(f"Records with returns (RETURN_QTY > 0): {(self.df['RETURN_QTY'] > 0).sum():,}")
        print(f"Records with no returns (RETURN_QTY = 0): {(self.df['RETURN_QTY'] == 0).sum():,}")
        print(f"Valid return dates: {self.df['RETURN_DATE'].notna().sum():,}")
        
        # Focus on customers who have made at least one return
        customers_with_returns = self.df[self.df['RETURN_QTY'] > 0]['CUSTOMER_EMAILID'].unique()
        analysis_df = self.df[self.df['CUSTOMER_EMAILID'].isin(customers_with_returns)].copy()
        
        print(f"Analyzing {len(customers_with_returns):,} customers who have made returns")
        print(f"Total records for these customers: {len(analysis_df):,}")
        
        # Calculate days to return for items that were returned
        returned_items = analysis_df[
            (analysis_df['RETURN_QTY'] > 0) & (analysis_df['RETURN_DATE'].notna())
        ].copy()
        
        if len(returned_items) > 0:
            returned_items['DAYS_TO_RETURN'] = (
                returned_items['RETURN_DATE'] - returned_items['ORDER_DATE']
            ).dt.days
            print(f"Items with valid return timing: {len(returned_items):,}")
        else:
            print("No valid return dates found - proceeding without return timing analysis")
        
        # Main customer aggregation
        customer_agg = analysis_df.groupby('CUSTOMER_EMAILID').agg({
            'SALES_ORDER_NO': 'nunique',              # Unique orders
            'SKU': 'nunique',                         # Product variety purchased
            'SALES_QTY': ['sum', 'mean'],             # Purchase behavior
            'RETURN_QTY': ['sum', 'mean'],            # Return quantities
            'ORDER_DATE': ['min', 'max'],             # Customer lifetime
        }).round(2)
        
        # Flatten column names
        customer_agg.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col 
            for col in customer_agg.columns]
        
        # Calculate return metrics
        customer_agg['RETURN_RATIO'] = (
            customer_agg['RETURN_QTY_sum'] / customer_agg['SALES_QTY_sum']
        )
        
        # Count of items that had returns (not return transactions)
        items_with_returns = analysis_df[analysis_df['RETURN_QTY'] > 0].groupby('CUSTOMER_EMAILID').size()
        customer_agg['ITEMS_RETURNED_COUNT'] = items_with_returns.fillna(0)
        
        # Return rate = items returned / total items purchased
        total_items_purchased = analysis_df.groupby('CUSTOMER_EMAILID').size()
        customer_agg['RETURN_RATE'] = customer_agg['ITEMS_RETURNED_COUNT'] / total_items_purchased
        
        # Product return diversity
        return_sku_variety = analysis_df[analysis_df['RETURN_QTY'] > 0].groupby('CUSTOMER_EMAILID')['SKU'].nunique()
        customer_agg['RETURN_PRODUCT_VARIETY'] = return_sku_variety.fillna(0)
        
        # Customer lifetime
        customer_agg['CUSTOMER_LIFETIME_DAYS'] = (
            customer_agg['ORDER_DATE_max'] - customer_agg['ORDER_DATE_min']
        ).dt.days
        
        # Recent activity (last 90 days)
        recent_date = analysis_df['ORDER_DATE'].max() - timedelta(days=90)
        
        # Recent orders (unique order numbers)
        recent_orders = analysis_df[
            analysis_df['ORDER_DATE'] >= recent_date
        ].groupby('CUSTOMER_EMAILID')['SALES_ORDER_NO'].nunique()
        customer_agg['RECENT_ORDERS'] = recent_orders.fillna(0)
        
        # Recent returns (items returned recently)
        recent_returns = analysis_df[
            (analysis_df['ORDER_DATE'] >= recent_date) & (analysis_df['RETURN_QTY'] > 0)
        ].groupby('CUSTOMER_EMAILID').size()
        customer_agg['RECENT_RETURNS'] = recent_returns.fillna(0)
        
        # RECENT vs AVERAGE RETURN RATIO - Churn Predictor
        # Filter to customers with sufficient order history (10+ orders)
        customers_with_history = customer_agg[customer_agg['SALES_ORDER_NO_nunique'] >= 10].index
        
        recent_vs_avg_ratios = []
        for customer_id in customers_with_history:
            customer_data = analysis_df[analysis_df['CUSTOMER_EMAILID'] == customer_id].copy()
            customer_data = customer_data.sort_values('ORDER_DATE')
            
            # Method 1: Last 25% of orders (or minimum 3 orders)
            total_orders = customer_data['SALES_ORDER_NO'].nunique()
            recent_order_count = max(3, int(total_orders * 0.25))
            
            # Get recent order numbers
            recent_order_numbers = customer_data['SALES_ORDER_NO'].unique()[-recent_order_count:]
            
            # Calculate recent return rate
            recent_data = customer_data[customer_data['SALES_ORDER_NO'].isin(recent_order_numbers)]
            recent_items_returned = (recent_data['RETURN_QTY'] > 0).sum()
            recent_total_items = len(recent_data)
            recent_return_rate = recent_items_returned / recent_total_items if recent_total_items > 0 else 0
            
            # Overall return rate for this customer
            overall_return_rate = customer_agg.loc[customer_id, 'RETURN_RATE']
            
            # Calculate ratio (recent vs average)
            if overall_return_rate > 0:
                recent_vs_avg_ratio = recent_return_rate / overall_return_rate
            else:
                recent_vs_avg_ratio = 1.0 if recent_return_rate == 0 else 5.0  # High value if started returning
            
            recent_vs_avg_ratios.append({
                'CUSTOMER_EMAILID': customer_id,
                'RECENT_RETURN_RATE': recent_return_rate,
                'RECENT_VS_AVG_RATIO': recent_vs_avg_ratio,
                'RECENT_ORDER_COUNT': recent_order_count
            })
        
        # Convert to DataFrame and merge
        recent_behavior_df = pd.DataFrame(recent_vs_avg_ratios).set_index('CUSTOMER_EMAILID')
        
        # Add to customer_agg (with defaults for customers with <10 orders)
        customer_agg['RECENT_RETURN_RATE'] = recent_behavior_df['RECENT_RETURN_RATE'].fillna(0)
        customer_agg['RECENT_VS_AVG_RATIO'] = recent_behavior_df['RECENT_VS_AVG_RATIO'].fillna(1.0)
        customer_agg['RECENT_ORDER_COUNT'] = recent_behavior_df['RECENT_ORDER_COUNT'].fillna(0)
        
        # Churn risk indicators
        customer_agg['RETURN_TREND_INCREASING'] = (customer_agg['RECENT_VS_AVG_RATIO'] > 1.5).astype(int)
        customer_agg['RETURN_TREND_DECREASING'] = (customer_agg['RECENT_VS_AVG_RATIO'] < 0.5).astype(int)

        # Return timing analysis (if valid return dates exist)
        if len(returned_items) > 0:
            return_timing = returned_items.groupby('CUSTOMER_EMAILID')['DAYS_TO_RETURN'].agg(['mean', 'std'])
            customer_agg['AVG_DAYS_TO_RETURN'] = return_timing['mean'].fillna(0)
            customer_agg['RETURN_TIMING_CONSISTENCY'] = return_timing['std'].fillna(0)
            
            # First return timing
            first_return_timing = returned_items.groupby('CUSTOMER_EMAILID')['DAYS_TO_RETURN'].min()
            customer_agg['DAYS_TO_FIRST_RETURN'] = first_return_timing.fillna(0)
            
            # Return timing spread
            return_spread = returned_items.groupby('CUSTOMER_EMAILID')['DAYS_TO_RETURN'].agg(
                lambda x: x.max() - x.min() if len(x) > 1 else 0
            )
            customer_agg['RETURN_TIMING_SPREAD'] = return_spread.fillna(0)
            
        else:
            # Use placeholder values when no return dates available
            customer_agg['AVG_DAYS_TO_RETURN'] = 0
            customer_agg['RETURN_TIMING_CONSISTENCY'] = 0
            customer_agg['DAYS_TO_FIRST_RETURN'] = 0
            customer_agg['RETURN_TIMING_SPREAD'] = 0
        
        # Advanced return behavior patterns
        
        # Batch return behavior - items returned on same order
        returns_per_order = analysis_df[analysis_df['RETURN_QTY'] > 0].groupby(
            ['CUSTOMER_EMAILID', 'SALES_ORDER_NO']
        ).size().reset_index(name='items_returned_per_order')
        
        avg_returns_per_order = returns_per_order.groupby('CUSTOMER_EMAILID')['items_returned_per_order'].mean()
        customer_agg['AVG_RETURNS_PER_ORDER'] = avg_returns_per_order.fillna(0)
        
        # Return frequency relative to purchase frequency
        customer_agg['RETURN_FREQUENCY_RATIO'] = (
            customer_agg['ITEMS_RETURNED_COUNT'] / customer_agg['SALES_ORDER_NO_nunique']
        )
        
        # Partial vs full returns - average return intensity per returned item
        partial_return_analysis = analysis_df[analysis_df['RETURN_QTY'] > 0].copy()
        if len(partial_return_analysis) > 0:
            partial_return_analysis['RETURN_INTENSITY'] = (
                partial_return_analysis['RETURN_QTY'] / partial_return_analysis['SALES_QTY']
            )
            avg_return_intensity = partial_return_analysis.groupby('CUSTOMER_EMAILID')['RETURN_INTENSITY'].mean()
            customer_agg['AVG_RETURN_INTENSITY'] = avg_return_intensity.fillna(0)
        else:
            customer_agg['AVG_RETURN_INTENSITY'] = 0
        
        # Fill any remaining NaN values
        customer_agg = customer_agg.fillna(0)
        
        # Select features for clustering
        clustering_features = [
            'SALES_ORDER_NO_nunique',         # Order frequency
            'SKU_nunique',                    # Product variety purchased
            'RETURN_RATE',                    # Items returned / total items
            'RETURN_RATIO',                   # Quantity returned / quantity purchased
            'ITEMS_RETURNED_COUNT',           # Total items returned
            'RETURN_PRODUCT_VARIETY',         # Variety of products returned
            'CUSTOMER_LIFETIME_DAYS',         # Customer tenure
            'RECENT_ORDERS',                  # Recent purchase activity
            'RECENT_RETURNS',                 # Recent return activity
            'SALES_QTY_mean',                 # Average purchase quantity per item
            'AVG_DAYS_TO_RETURN',             # Average return timing
            'RETURN_TIMING_SPREAD',           # Return timing variability
            'AVG_RETURNS_PER_ORDER',          # Batch return behavior
            'RETURN_FREQUENCY_RATIO',         # Returns per order ratio
            'AVG_RETURN_INTENSITY',           # Partial vs full return tendency
            'RECENT_VS_AVG_RATIO',            # Recent vs historical return behavior
            'RETURN_TREND_INCREASING',        # Binary: Return trend going up
        ]
        
        print(f"\nUsing features for clustering: {clustering_features}")
        
        self.customer_features = customer_agg[clustering_features].copy()
        self.customer_features = self.customer_features.round(3)
        
        print(f"Customer features created for {len(self.customer_features):,} customers")
        
        return self.customer_features
    
    def find_optimal_clusters(self, max_clusters=10):
        """Find optimal number of clusters using elbow method and silhouette score"""
        
        if self.customer_features is None:
            self.prepare_customer_features()
        
        # Remove CLUSTER column if it exists from previous runs
        features_for_clustering = self.customer_features.copy()
        if 'CLUSTER' in features_for_clustering.columns:
            features_for_clustering = features_for_clustering.drop('CLUSTER', axis=1)
        
        # Store features for future usage
        self._optimiation_features = features_for_clustering.copy()

        # Scale features
        X_scaled = self.scaler.fit_transform(features_for_clustering)
        
        inertias = []
        silhouette_scores = []
        k_range = range(2, max_clusters + 1)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X_scaled)
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))
        
        # Plot results
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Elbow curve
        ax1.plot(k_range, inertias, 'bo-')
        ax1.set_xlabel('Number of Clusters')
        ax1.set_ylabel('Inertia')
        ax1.set_title('Elbow Method for Optimal k')
        ax1.grid(True)
        
        # Silhouette scores
        ax2.plot(k_range, silhouette_scores, 'ro-')
        ax2.set_xlabel('Number of Clusters')
        ax2.set_ylabel('Silhouette Score')
        ax2.set_title('Silhouette Score by Number of Clusters')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # Recommend optimal k
        optimal_k = k_range[np.argmax(silhouette_scores)]
        print(f"Recommended number of clusters: {optimal_k}")
        print(f"Silhouette score: {max(silhouette_scores):.3f}")
        
        return optimal_k, silhouette_scores
    
    def perform_clustering(self, n_clusters=5):
        """Perform K-means clustering"""
        
        if hasattr(self, '_optimization_features'):
            features_for_clustering = self._optimization_features.copy()
            print("Using features from optimization run")
        else:
            if self.customer_features is None:
                self.prepare_customer_features()
            
            features_for_clustering = self.customer_features.copy()
            if 'CLUSTER' in features_for_clustering.columns:
                features_for_clustering = features_for_clustering.drop('CLUSTER', axis=1)
            print("No optimization run found - using current features")

        # Debug: print actual feature columns
        print(f"Features for clustering: {list(features_for_clustering.columns)}")
        print(f"Number of features: {len(features_for_clustering.columns)}")
        
        # Use transform() instead of fit_transform() to use existing scaler
        try:
            X_scaled = self.scaler.transform(features_for_clustering)
            print("Using existing scaler parameters")
        except:
            X_scaled = self.scaler.fit_transform(features_for_clustering)
            print("Fitting new scaler parameters")
        
        print(f"Scaled data shape: {X_scaled.shape}")

        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_scaled)
        
        # Add clusters to customer features
        self.customer_features['CLUSTER'] = cluster_labels
        
        # Calculate cluster centers in original scale
        cluster_centers = pd.DataFrame(
            self.scaler.inverse_transform(kmeans.cluster_centers_),
            columns=features_for_clustering.columns,
            index=[f'Cluster_{i}' for i in range(n_clusters)]
        )
        
        return cluster_labels, cluster_centers
    
    def analyze_clusters(self):
        """Analyze and interpret clusters"""
        
        if 'CLUSTER' not in self.customer_features.columns:
            print("Please run perform_clustering() first")
            return
        
        cluster_summary = self.customer_features.groupby('CLUSTER').agg({
            'SALES_ORDER_NO_nunique': ['mean', 'count'],
            'RETURN_RATE': 'mean',
            'RETURN_RATIO': 'mean',
            'ITEMS_RETURNED_COUNT': 'mean',
            'CUSTOMER_LIFETIME_DAYS': 'mean',
            'RECENT_ORDERS': 'mean',
            'RECENT_RETURNS': 'mean',
            'SKU_nunique': 'mean',
            'SALES_QTY_mean': 'mean',
            'AVG_DAYS_TO_RETURN': 'mean',
            'RETURN_TIMING_SPREAD': 'mean',
            'AVG_RETURNS_PER_ORDER': 'mean',
            'RETURN_FREQUENCY_RATIO': 'mean',
            'AVG_RETURN_INTENSITY': 'mean',
            'RECENT_VS_AVG_RATIO': 'mean',
            'RETURN_TREND_INCREASING': 'mean',
        }).round(3)
        
        # Flatten column names
        cluster_summary.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col 
            for col in cluster_summary.columns]
        
        # Rename for clarity
        rename_dict = {
            'SALES_ORDER_NO_nunique_mean': 'AVG_ORDERS',
            'SALES_ORDER_NO_nunique_count': 'CUSTOMER_COUNT',
            'RETURN_RATE_mean': 'AVG_RETURN_RATE',
            'RETURN_RATIO_mean': 'AVG_RETURN_RATIO',
            'ITEMS_RETURNED_COUNT_mean': 'AVG_ITEMS_RETURNED',
            'CUSTOMER_LIFETIME_DAYS_mean': 'AVG_LIFETIME_DAYS',
            'RECENT_ORDERS_mean': 'AVG_RECENT_ORDERS',
            'RECENT_RETURNS_mean': 'AVG_RECENT_RETURNS',
            'SKU_nunique_mean': 'AVG_PRODUCT_VARIETY',
            'SALES_QTY_mean_mean': 'AVG_ORDER_SIZE',
            'AVG_DAYS_TO_RETURN_mean': 'AVG_DAYS_TO_RETURN',
            'RETURN_TIMING_SPREAD_mean': 'AVG_RETURN_TIMING_SPREAD',
            'AVG_RETURNS_PER_ORDER_mean': 'AVG_RETURNS_PER_ORDER',
            'RETURN_FREQUENCY_RATIO_mean': 'AVG_RETURN_FREQUENCY_RATIO',
            'AVG_RETURN_INTENSITY_mean': 'AVG_RETURN_INTENSITY',
            'RECENT_VS_AVG_RATIO_mean': 'AVG_RECENT_VS_AVG_RATIO',
            'RETURN_TREND_INCREASING_mean': 'AVG_RETURN_TREND_INCREASING'
        }
        
        cluster_summary = cluster_summary.rename(columns=rename_dict)
        
        print("=== CLUSTER ANALYSIS ===")
        print(cluster_summary)
        
        # Calculate percentiles for better thresholds
        return_rate_75 = self.customer_features['RETURN_RATE'].quantile(0.75)
        return_rate_25 = self.customer_features['RETURN_RATE'].quantile(0.25)
        return_ratio_75 = self.customer_features['RETURN_RATIO'].quantile(0.75)
        orders_75 = self.customer_features['SALES_ORDER_NO_nunique'].quantile(0.75)
        recent_orders_25 = self.customer_features['RECENT_ORDERS'].quantile(0.25)
        return_intensity_75 = self.customer_features['AVG_RETURN_INTENSITY'].quantile(0.75)
        returns_per_order_75 = self.customer_features['AVG_RETURNS_PER_ORDER'].quantile(0.75)
        if 'RECENT_VS_AVG_RATIO' in self.customer_features.columns:
            recent_vs_avg_75 = self.customer_features['RECENT_VS_AVG_RATIO'].quantile(0.75)
            recent_vs_avg_25 = self.customer_features['RECENT_VS_AVG_RATIO'].quantile(0.25)
        else:
            recent_vs_avg_75 = 2.0  # Default fallback
            recent_vs_avg_25 = 0.5
        
        print(f"\nData-driven thresholds:")
        print(f"Return Rate - 75th percentile: {return_rate_75:.3f}")
        print(f"Return Ratio - 75th percentile: {return_ratio_75:.3f}")
        print(f"Orders - 75th percentile: {orders_75:.0f}")
        print(f"Recent Orders - 25th percentile: {recent_orders_25:.1f}")
        print(f"Return Intensity - 75th percentile: {return_intensity_75:.3f}")
        print(f"Returns Per Order - 75th percentile: {returns_per_order_75:.2f}")
        print(f"Recent vs Avg Ratio - 75th percentile: {recent_vs_avg_75:.2f}")


        # Create cluster interpretations with data-driven thresholds
        interpretations = {}
        for cluster in cluster_summary.index:
            profile = cluster_summary.loc[cluster]
            
            # Determine cluster characteristics based on combined order-return behavior
            if ('RECENT_VS_AVG_RATIO' in cluster_summary.columns and 
                profile.get('RECENT_VS_AVG_RATIO', 1.0) > recent_vs_avg_75 and
                profile['AVG_RECENT_ORDERS'] <= recent_orders_25):
                cluster_type = "🚨 CHURN ALERT - Return Behavior Worsening"
                action = "Immediate intervention - analyze recent order issues"
            if (profile['AVG_RECENT_VS_AVG_RATIO'] > recent_vs_avg_75 and
                profile['AVG_RETURN_TREND_INCREASING'] > 0.3):  # 30%+ have increasing trend
                cluster_type = "⚠️ CHURN ALERT - Increasing Return Trend"
                action = "Immediate intervention - return behavior worsening"
            if (profile['AVG_RETURN_RATE'] >= return_rate_75 and 
                profile['AVG_RETURN_RATIO'] >= return_ratio_75):
                cluster_type = "HIGH RISK - Heavy Returners"
                action = "Immediate retention intervention needed"
            elif (profile['AVG_RETURNS_PER_ORDER'] >= returns_per_order_75 and 
                profile['AVG_RETURN_INTENSITY'] >= return_intensity_75):
                cluster_type = "BULK RETURNERS - High Volume Per Order"
                action = "Order size limits + return policy review"
            elif (profile['AVG_RETURN_INTENSITY'] >= return_intensity_75 and 
                profile['AVG_DAYS_TO_RETURN'] <= 7):
                cluster_type = "FAST FULL RETURNERS - Quick Complete Returns"
                action = "Product education + sizing guides"
            elif (profile['AVG_RETURN_RATE'] >= return_rate_75 and 
                profile['AVG_RECENT_ORDERS'] <= recent_orders_25):
                cluster_type = "CHURN RISK - High Returns + Low Engagement"
                action = "Re-engagement campaign with better recommendations"
            elif (profile['AVG_ORDERS'] >= orders_75 and 
                profile['AVG_RETURN_RATE'] <= return_rate_25):
                cluster_type = "VIP - High Volume Low Returns"
                action = "Premium loyalty rewards program"
            elif (profile['AVG_ORDERS'] >= orders_75 and 
                    profile['AVG_RETURN_RATE'] >= return_rate_75):
                cluster_type = "COMPLEX - High Volume High Returns"
                action = "Personalized service + product quality review"
            
            elif profile['AVG_PRODUCT_VARIETY'] >= self.customer_features['SKU_nunique'].quantile(0.75):
                cluster_type = "EXPLORERS - High Product Variety"
                action = "Product discovery campaigns"
            elif profile['AVG_RECENT_ORDERS'] <= recent_orders_25:
                cluster_type = "AT RISK - Low Recent Activity"
                action = "Win-back campaign"
            else:
                cluster_type = "STANDARD - Balanced Customers"
                action = "Standard marketing engagement"
            
            interpretations[cluster] = {
                'type': cluster_type,
                'action': action,
                'customers': int(profile['CUSTOMER_COUNT'])
            }
        
        print("\n=== CLUSTER INTERPRETATIONS ===")
        for cluster, info in interpretations.items():
            print(f"\nCluster {cluster}: {info['type']}")
            print(f"  Customers: {info['customers']:,}")
            print(f"  Recommended Action: {info['action']}")
        
        return cluster_summary, interpretations
    
    def visualize_clusters(self):
        """Create visualizations for cluster analysis"""
        
        if 'CLUSTER' not in self.customer_features.columns:
            print("Please run perform_clustering() first")
            return
        
        # Get features without CLUSTER column for PCA
        features_for_viz = self.customer_features.drop('CLUSTER', axis=1)
        
        # PCA for 2D visualization
        X_scaled = self.scaler.transform(features_for_viz)
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. PCA scatter plot
        scatter = axes[0, 0].scatter(X_pca[:, 0], X_pca[:, 1], 
            c=self.customer_features['CLUSTER'], 
            cmap='viridis', alpha=0.6)
        axes[0, 0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        axes[0, 0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        axes[0, 0].set_title('Customer Clusters in PCA Space')
        plt.colorbar(scatter, ax=axes[0, 0])
        
        # 2. Return Rate vs Order Frequency
        axes[0, 1].scatter(self.customer_features['SALES_ORDER_NO_nunique'], 
            self.customer_features['RETURN_RATE'],
            c=self.customer_features['CLUSTER'], 
            cmap='viridis', alpha=0.6)
        axes[0, 1].set_xlabel('Number of Orders')
        axes[0, 1].set_ylabel('Return Rate')
        axes[0, 1].set_title('Return Rate vs Order Frequency')
        
        # 3. Return Ratio vs Customer Lifetime
        axes[1, 0].scatter(self.customer_features['CUSTOMER_LIFETIME_DAYS'], 
            self.customer_features['RETURN_RATIO'],
            c=self.customer_features['CLUSTER'], 
            cmap='viridis', alpha=0.6)
        axes[1, 0].set_xlabel('Customer Lifetime (Days)')
        axes[1, 0].set_ylabel('Return Ratio')
        axes[1, 0].set_title('Return Ratio vs Customer Lifetime')
        
        # 4. Cluster size distribution
        cluster_counts = self.customer_features['CLUSTER'].value_counts().sort_index()
        axes[1, 1].bar(cluster_counts.index, cluster_counts.values, color='skyblue')
        axes[1, 1].set_xlabel('Cluster')
        axes[1, 1].set_ylabel('Number of Customers')
        axes[1, 1].set_title('Cluster Size Distribution')
        
        plt.tight_layout()
        plt.show()
    
    def get_cluster_customers(self, cluster_id):
        """Get customer emails for a specific cluster"""
        cluster_customers = self.customer_features[
            self.customer_features['CLUSTER'] == cluster_id
        ].index.tolist()
        return cluster_customers
    
    def export_results(self, filename='customer_clusters.csv'):
        """Export clustering results"""
        if 'CLUSTER' not in self.customer_features.columns:
            print("Please run perform_clustering() first")
            return
        
        # Prepare export data
        export_data = self.customer_features.copy()
        export_data.index.name = 'CUSTOMER_EMAILID'
        export_data = export_data.reset_index()
        
        # Add cluster interpretations
        cluster_summary, interpretations = self.analyze_clusters()
        
        # Map cluster types
        cluster_type_map = {i: interpretations[i]['type'] for i in interpretations.keys()}
        export_data['CLUSTER_TYPE'] = export_data['CLUSTER'].map(cluster_type_map)
        
        # Export
        export_data.to_csv(filename, index=False)
        print(f"Results exported to {filename}")
        
        return export_data