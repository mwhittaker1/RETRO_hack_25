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
        """
        self.df = df.copy()
        self.customer_features = None
        self.clusters = None
        self.scaler = StandardScaler()
        
    def prepare_customer_features(self):
        """Create customer-level features for clustering"""
        
        # Convert dates
        self.df['ORDER_DATE'] = pd.to_datetime(self.df['ORDER_DATE'])
        
        # Identify actual returns using UNITS_RETURNED_FLAG and RETURN_QTY
        print("Analyzing data structure...")
        print(f"UNITS_RETURNED_FLAG distribution:")
        print(self.df['UNITS_RETURNED_FLAG'].value_counts())
        print(f"\nRecords with RETURN_QTY > 0: {(self.df['RETURN_QTY'] > 0).sum():,}")
        
        # Method 1: Use UNITS_RETURNED_FLAG = 'Y' to identify returns
        actual_returns = self.df[self.df['UNITS_RETURNED_FLAG'] == 'Yes'].copy()
        
        # Verify this matches RETURN_QTY > 0
        returns_by_qty = self.df[self.df['RETURN_QTY'] > 0].copy()
        
        print(f"Returns by FLAG='Yes': {len(actual_returns):,}")
        print(f"Returns by QTY>0: {len(returns_by_qty):,}")
        
        # Should match, but if not, use the larger dataset.
        if len(returns_by_qty) > len(actual_returns):
            print("Using RETURN_QTY > 0 method (more comprehensive)")
            actual_returns = returns_by_qty
        else:
            print("Using UNITS_RETURNED_FLAG = 'Yes' method")
        
        print(f"Final return records: {len(actual_returns):,} out of {len(self.df):,} total records")
        
        # convert return dates to datetime
        actual_returns['RETURN_DATE'] = pd.to_datetime(actual_returns['RETURN_DATE'], errors='coerce')

        # Get customers who have made returns
        customers_with_returns = actual_returns['CUSTOMER_EMAILID'].unique()
        
        # Get ALL records (sales + returns) for these customers
        all_customer_records = self.df[
            self.df['CUSTOMER_EMAILID'].isin(customers_with_returns)
        ].copy()
        
        print(f"Analyzing {len(customers_with_returns):,} customers who have made returns")
        print(f"Total records for these customers: {len(all_customer_records):,}")
        
        # Separate sales and returns for better analysis
        customer_sales = all_customer_records[all_customer_records['UNITS_RETURNED_FLAG'] != 'Yes']
        customer_returns = all_customer_records[all_customer_records['UNITS_RETURNED_FLAG'] == 'Yes']
        
        print(f"Sales records: {len(customer_sales):,}")
        print(f"Return records: {len(customer_returns):,}")
        
        # Clean return_date "-" values
        customer_returns['RETURN_DATE'] = pd.to_datetime(customer_returns['RETURN_DATE'], errors='coerce')
        print(f"Valid return dates: {customer_returns['RETURN_DATE'].notna().sum():,}")

        # Aggregate sales data by customer
        sales_agg = customer_sales.groupby('CUSTOMER_EMAILID').agg({
            'SALES_ORDER_NO': 'nunique',  # Total unique orders
            'SKU': 'nunique',  # Product variety purchased
            'SALES_QTY': ['sum', 'mean'],  # Total and average purchase quantities
            'ORDER_DATE': ['min', 'max'],  # Customer lifetime
        })
        
        # Aggregate returns data by customer  
        returns_agg = customer_returns.groupby('CUSTOMER_EMAILID').agg({
            'RETURN_QTY': ['sum', 'mean', 'count'],  # Return patterns
            'RETURN_NO': 'nunique',  # Number of return transactions
            'SKU': 'nunique',  # Variety of products returned
            'RETURN_DATE': ['min', 'max'],  # Return lifetime
        })
        
        # Flatten DataFrames to single-level columns Fbefore joining
        sales_agg.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in sales_agg.columns]
        returns_agg.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in returns_agg.columns]
        
        # Add prefix to return columns to avoid conflicts
        returns_agg = returns_agg.add_prefix('RETURN_')
        
        # Combine sales and returns data
        customer_agg = sales_agg.join(returns_agg, how='inner')
        
        # Calculate derived features
        customer_agg['CUSTOMER_LIFETIME_DAYS'] = (
            customer_agg['ORDER_DATE_max'] - customer_agg['ORDER_DATE_min']
        ).dt.days
        
        # Return rate = return transactions per order
        customer_agg['RETURN_RATE'] = (
            customer_agg['RETURN_RETURN_NO_nunique'] / customer_agg['SALES_ORDER_NO_nunique']
        )
        
        # Return ratio = quantity returned vs purchased
        customer_agg['RETURN_RATIO'] = (
            customer_agg['RETURN_RETURN_QTY_sum'] / customer_agg['SALES_QTY_sum']
        )
        
        # Return frequency = individual return events
        customer_agg['RETURN_FREQUENCY'] = customer_agg['RETURN_RETURN_QTY_count']
        
        # Product return diversity = how many different SKUs returned
        customer_agg['RETURN_PRODUCT_VARIETY'] = customer_agg['RETURN_SKU_nunique']
        
        # Recent activity (last 90 days)
        recent_date = all_customer_records['ORDER_DATE'].max() - timedelta(days=90)
        recent_sales = customer_sales[customer_sales['ORDER_DATE'] >= recent_date].groupby('CUSTOMER_EMAILID').size()
        valid_return_dates = customer_returns['RETURN_DATE'].notna()
        recent_returns = customer_returns[
            valid_return_dates & (customer_returns['RETURN_DATE'] >= recent_date)
        ].groupby('CUSTOMER_EMAILID').size()
        
        customer_agg['RECENT_ORDERS'] = recent_sales.fillna(0)
        customer_agg['RECENT_RETURNS'] = recent_returns.fillna(0)
        
        # Cross-order return analysis
        print("Calculating cross-order return patterns...")

        # Get return timing relative to ALL orders (not just the returned order)
        customer_orders = all_customer_records.groupby('CUSTOMER_EMAILID')['ORDER_DATE'].agg(['min', 'max', 'count'])
        customer_returns_timing = customer_returns.groupby('CUSTOMER_EMAILID')['ORDER_DATE'].agg(['min', 'max', 'count'])

        # Cross-order return analysis
        print("Calculating cross-order return patterns...")

        # Get return timing relative to ALL orders (not just the returned order)
        customer_orders = all_customer_records.groupby('CUSTOMER_EMAILID')['ORDER_DATE'].agg(['min', 'max', 'count'])
        customer_returns_timing = customer_returns.groupby('CUSTOMER_EMAILID')['ORDER_DATE'].agg(['min', 'max', 'count'])

        # Since RETURN_DATE is invalid, use ORDER_DATE for return timing analysis
        print("Using ORDER_DATE for return timing analysis (RETURN_DATE unavailable)")

        # Time between first order and first return
        customer_agg['DAYS_TO_FIRST_RETURN'] = (
            customer_returns_timing['min'] - customer_orders['min']
        ).dt.days

        # Return concentration - do they return everything at once?
        return_date_spread = customer_returns.groupby('CUSTOMER_EMAILID')['ORDER_DATE'].agg(
            lambda x: (x.max() - x.min()).days if len(x) > 1 else 0
        )
        customer_agg['RETURN_DATE_SPREAD'] = return_date_spread.fillna(0)

        # Stockpiling indicators
        customer_agg['ORDERS_BEFORE_FIRST_RETURN'] = customer_orders['count'] - customer_returns_timing['count']

        # Batch return behavior - returns per distinct return date
        returns_per_date = customer_returns.groupby(['CUSTOMER_EMAILID', 'ORDER_DATE']).size().reset_index(name='returns_per_date')
        avg_returns_per_batch = returns_per_date.groupby('CUSTOMER_EMAILID')['returns_per_date'].mean()
        customer_agg['AVG_RETURNS_PER_BATCH'] = avg_returns_per_batch.fillna(0)

        # Return velocity - returns per day during active return period
        customer_agg['RETURN_VELOCITY'] = np.where(
            customer_agg['RETURN_DATE_SPREAD'] > 0,
            customer_agg['RETURN_FREQUENCY'] / customer_agg['RETURN_DATE_SPREAD'],
            customer_agg['RETURN_FREQUENCY']  # If all returns same day
        )

        # Fill NaN values
        customer_agg = customer_agg.fillna(0)
        
        # Select features for clustering
        clustering_features = [
            'SALES_ORDER_NO_nunique',      # Order frequency
            'SKU_nunique',                 # Product variety purchased
            'RETURN_RATE',                 # Return rate (returns per order)
            'RETURN_RATIO',                # Return intensity (qty returned/purchased)
            'RETURN_FREQUENCY',            # Total return events
            'RETURN_PRODUCT_VARIETY',      # Variety of products returned
            'CUSTOMER_LIFETIME_DAYS',      # Customer tenure
            'RECENT_ORDERS',               # Recent purchase activity
            'RECENT_RETURNS',              # Recent return activity
            'SALES_QTY_mean',              # Average order size
            'DAYS_TO_FIRST_RETURN',        # Cross-order: time to first return
            'RETURN_DATE_SPREAD',          # Stockpiling: return timing spread
            'ORDERS_BEFORE_FIRST_RETURN',  # Stockpiling: orders before returning
            'AVG_RETURNS_PER_BATCH',       # Batch behavior
            'RETURN_VELOCITY',             # Returns per day intensity
        ]
        
        print(f"\nUsing features for clustering: {clustering_features}")
        
        self.customer_features = customer_agg[clustering_features].copy()
        self.customer_features = self.customer_features.round(2)
        
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
        
        if self.customer_features is None:
            self.prepare_customer_features()
        
        # Remove CLUSTER column if it exists from previous runs
        features_for_clustering = self.customer_features.copy()
        if 'CLUSTER' in features_for_clustering.columns:
            features_for_clustering = features_for_clustering.drop('CLUSTER', axis=1)
        
        # Debug: print actual feature columns
        print(f"Features for clustering:")
        for col in features_for_clustering.columns:
            print(f" - {col}\n")
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
        
        # Debug: print cluster centers shape
        print(f"Cluster centers shape: {kmeans.cluster_centers_.shape}")
        print(f"Features columns length: {len(features_for_clustering.columns)}")
        
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
            'CUSTOMER_LIFETIME_DAYS': 'mean',
            'RECENT_ORDERS': 'mean',
            'SKU_nunique': 'mean',
            'SALES_QTY_mean': 'mean',
            'DAYS_TO_FIRST_RETURN': 'mean',
            'RETURN_DATE_SPREAD': 'mean',
            'ORDERS_BEFORE_FIRST_RETURN': 'mean',
            'AVG_RETURNS_PER_BATCH': 'mean',
            'RETURN_VELOCITY': 'mean'
        }).round(2)
        
        # Add DAYS_TO_RETURN if available
        if 'DAYS_TO_RETURN_mean' in self.customer_features.columns:
            days_summary = self.customer_features.groupby('CLUSTER')['DAYS_TO_RETURN_mean'].mean().round(2)
            cluster_summary = cluster_summary.join(days_summary.to_frame('DAYS_TO_RETURN_mean'))
        
        # Flatten column names
        cluster_summary.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in cluster_summary.columns]
        
        # Rename for clarity
        rename_dict = {
            'SALES_ORDER_NO_nunique_mean': 'AVG_ORDERS',
            'SALES_ORDER_NO_nunique_count': 'CUSTOMER_COUNT',
            'RETURN_RATE_mean': 'AVG_RETURN_RATE',
            'RETURN_RATIO_mean': 'AVG_RETURN_RATIO',
            'CUSTOMER_LIFETIME_DAYS_mean': 'AVG_LIFETIME_DAYS',
            'RECENT_ORDERS_mean': 'AVG_RECENT_ORDERS',
            'SKU_nunique_mean': 'AVG_PRODUCT_VARIETY',
            'SALES_QTY_mean_mean': 'AVG_ORDER_SIZE',
            'DAYS_TO_RETURN_mean': 'AVG_DAYS_TO_RETURN',
            'DAYS_TO_FIRST_RETURN_mean': 'AVG_DAYS_TO_FIRST_RETURN',
            'RETURN_DATE_SPREAD_mean': 'AVG_RETURN_DATE_SPREAD',
            'ORDERS_BEFORE_FIRST_RETURN_mean': 'AVG_ORDERS_BEFORE_FIRST_RETURN',
            'AVG_RETURNS_PER_BATCH_mean': 'AVG_RETURNS_PER_BATCH',
            'RETURN_VELOCITY_mean': 'AVG_RETURN_VELOCITY'
        }
        
        cluster_summary = cluster_summary.rename(columns=rename_dict)
        
        print("=== CLUSTER ANALYSIS ===")
        for col in cluster_summary:
            print(col+"\n")
        
        # Calculate percentiles for better thresholds
        return_rate_75 = self.customer_features['RETURN_RATE'].quantile(0.75)
        return_rate_25 = self.customer_features['RETURN_RATE'].quantile(0.25)
        return_ratio_75 = self.customer_features['RETURN_RATIO'].quantile(0.75)
        orders_75 = self.customer_features['SALES_ORDER_NO_nunique'].quantile(0.75)
        recent_orders_25 = self.customer_features['RECENT_ORDERS'].quantile(0.25)
        
        # stockpile thresholds
        first_return_25 = self.customer_features['DAYS_TO_FIRST_RETURN'].quantile(0.25)
        return_spread_75 = self.customer_features['RETURN_DATE_SPREAD'].quantile(0.75)
        orders_before_return_75 = self.customer_features['ORDERS_BEFORE_FIRST_RETURN'].quantile(0.75)
        returns_per_batch_75 = self.customer_features['AVG_RETURNS_PER_BATCH'].quantile(0.75)
        velocity_75 = self.customer_features['RETURN_VELOCITY'].quantile(0.75)

        print(f"\nData-driven thresholds:")
        print(f"Return Rate - 75th percentile: {return_rate_75:.2f}")
        print(f"Return Ratio - 75th percentile: {return_ratio_75:.2f}")
        print(f"Orders - 75th percentile: {orders_75:.0f}")
        print(f"Recent Orders - 25th percentile: {recent_orders_25:.2f}")
        print(f"Days to First Return - 25th percentile: {first_return_25:.1f}")
        print(f"Return Date Spread - 75th percentile: {return_spread_75:.1f}")
        print(f"Orders Before First Return - 75th percentile: {orders_before_return_75:.1f}")
        print(f"Returns Per Batch - 75th percentile: {returns_per_batch_75:.2f}")
        print(f"Return Velocity - 75th percentile: {velocity_75:.3f}")

        # Create cluster interpretations with data-driven thresholds
        interpretations = {}
        for cluster in cluster_summary.index:
            profile = cluster_summary.loc[cluster]
            
            # Determine cluster characteristics based on your data distribution
            if profile['AVG_RETURN_RATE'] >= return_rate_75 and profile['AVG_RETURN_RATIO'] >= return_ratio_75:
                cluster_type = "HIGH RISK - Heavy Returners"
                action = "Immediate retention intervention needed"
            if (profile['AVG_ORDERS_BEFORE_FIRST_RETURN'] >= orders_before_return_75 and 
                profile['AVG_RETURN_DATE_SPREAD'] <= 7):
                cluster_type = "STOCKPILERS - Order & Bulk Return"
                action = "Return policy review + purchase limits"
            elif (profile['AVG_RETURNS_PER_BATCH'] >= returns_per_batch_75 and 
                profile['RETURN_VELOCITY'] >= velocity_75):
                cluster_type = "BATCH RETURNERS - High Volume Fast Returns"
                action = "Fraud investigation + account review"
            elif profile['DAYS_TO_FIRST_RETURN'] <= first_return_25:
                cluster_type = "IMMEDIATE RETURNERS - Quick Dissatisfaction"
                action = "Pre-purchase education + sizing guides"
            elif (profile['RETURN_DATE_SPREAD'] >= return_spread_75 and 
                profile['AVG_RETURN_RATE'] >= return_rate_75):
                cluster_type = "SYSTEMATIC RETURNERS - Spread Out High Returns"
                action = "Account monitoring + personalized service"
            elif profile['AVG_RETURN_RATE'] >= return_rate_75 and profile['AVG_RETURN_RATIO'] >= return_ratio_75:
                cluster_type = "HIGH RISK - Heavy Returners"
                action = "Immediate retention intervention needed"
            elif profile['AVG_RETURN_RATE'] >= return_rate_75 and profile['AVG_RECENT_ORDERS'] <= recent_orders_25:
                cluster_type = "CHURN RISK - High Return + Low Engagement"
                action = "Re-engagement campaign with product recommendations"
            elif profile['AVG_ORDERS'] >= orders_75 and profile['AVG_RETURN_RATE'] <= return_rate_25:
                cluster_type = "VIP - High Volume Low Returns"
                action = "Premium loyalty rewards program"
            elif profile['AVG_ORDERS'] >= orders_75 and profile['AVG_RETURN_RATE'] >= return_rate_75:
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
        X_scaled = self.scaler.fit_transform(features_for_viz)
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