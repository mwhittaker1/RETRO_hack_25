"""
DuckDB Database Setup and Management
Creates local database with bronze/silver/gold layers for customer clustering analysis
"""

import duckdb
import pandas as pd
from pathlib import Path
import logging
from typing import Optional, List

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CustomerClusteringDB:
    def __init__(self, db_path: str = "customer_clustering.db"):
        """Initialize DuckDB connection and create database structure"""
        self.db_path = db_path
        self.conn = duckdb.connect(db_path)
        logger.info(f"Connected to DuckDB at {db_path}")
        self._setup_database()
    
    def _setup_database(self):
        """Create database schema with bronze, silver, and gold layers"""
        
        bronze_ddl = """
        CREATE TABLE IF NOT EXISTS bronze_return_order_data (
            primary_key VARCHAR PRIMARY KEY, 
            sales_order_no VARCHAR,
            customer_emailid VARCHAR,
            order_date TIMESTAMP,
            sku VARCHAR,
            sales_qty INTEGER,
            return_qty INTEGER,
            units_returned_flag VARCHAR,
            return_date TIMESTAMP,
            return_no VARCHAR,
            return_comment TEXT, 
            orderlink VARCHAR,
            q_cls_id VARCHAR,
            q_sku_desc VARCHAR,
            q_gmm_id VARCHAR,
            q_sku_id VARCHAR,
            class_ VARCHAR,
            division_ VARCHAR,
            brand_ VARCHAR,
            q_clr_dnum VARCHAR,
            q_clr_desc VARCHAR,
            vendor_style VARCHAR,
            size_ VARCHAR
        );
        """
        
        # Create silver layer table for customer features
        silver_ddl = """
        CREATE TABLE IF NOT EXISTS silver_customer_features (
            customer_emailid VARCHAR PRIMARY KEY,
            customer_emailid_cleaned VARCHAR,  -- Standardized email for consolidation
            
            -- BASIC VOLUME METRICS
            sales_order_no_nunique INTEGER,
            sku_nunique INTEGER,
            items_returned_count INTEGER,
            sales_qty_mean DOUBLE,
            avg_order_size DOUBLE,
            
            -- RETURN BEHAVIOR PATTERNS
            return_rate DOUBLE,
            return_ratio DOUBLE,
            return_product_variety INTEGER,
            avg_returns_per_order DOUBLE,
            return_frequency_ratio DOUBLE,
            return_intensity DOUBLE,
            consecutive_returns INTEGER,
            avg_consecutive_returns DOUBLE,
            
            -- TEMPORAL & TIMING PATTERNS
            customer_lifetime_days INTEGER,
            avg_days_to_return DOUBLE,
            return_timing_spread DOUBLE,
            customer_tenure_stage VARCHAR,
            
            -- TREND & RECENCY ANALYSIS
            recent_orders INTEGER,
            recent_returns INTEGER,
            recent_vs_avg_ratio DOUBLE,
            behavior_stability_score DOUBLE,
            
            -- Hey Claude! Look here for the updated data that includes order values!
            -- MONETARY VALUE PATTERNS (PLACEHOLDER)
            avg_order_value DOUBLE DEFAULT NULL,  -- TO BE POPULATED
            avg_return_value DOUBLE DEFAULT NULL,  -- TO BE POPULATED
            high_value_return_affinity DOUBLE DEFAULT NULL,  -- TO BE POPULATED
            
            -- PRODUCT & CATEGORY INTELLIGENCE
            product_category_loyalty DOUBLE,
            category_diversity_score DOUBLE,
            category_loyalty_score DOUBLE,
            high_return_category_affinity DOUBLE,
            
            -- ADJACENCY & REPEAT BEHAVIOR
            sku_adjacency_orders INTEGER,
            sku_adjacency_returns INTEGER,
            sku_adjacency_timing DOUBLE,
            sku_adjacency_return_timing DOUBLE,
            
            -- SEASONAL & TREND SUSCEPTIBILITY
            seasonal_susceptibility_returns DOUBLE,
            seasonal_susceptibility_orders DOUBLE,
            trend_product_category_return_rate DOUBLE,
            trend_product_category_order_rate DOUBLE,
            
            -- Metadata
            feature_calculation_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            data_quality_flags TEXT DEFAULT '',
            total_sales_qty INTEGER,
            total_return_qty INTEGER,
            first_order_date TIMESTAMP,
            last_order_date TIMESTAMP
        );
        """
        
        # Create gold layer table for clustering
        gold_ddl = """
        CREATE TABLE IF NOT EXISTS gold_cluster_processed (
            customer_emailid VARCHAR PRIMARY KEY,
            
            -- Scaled and processed features for clustering
            -- All features will be normalized/scaled appropriately
            sales_order_no_nunique_scaled DOUBLE,
            sku_nunique_scaled DOUBLE,
            items_returned_count_scaled DOUBLE,
            sales_qty_mean_scaled DOUBLE,
            avg_order_size_scaled DOUBLE,
            return_rate_scaled DOUBLE,
            return_ratio_scaled DOUBLE,
            return_product_variety_scaled DOUBLE,
            avg_returns_per_order_scaled DOUBLE,
            return_frequency_ratio_scaled DOUBLE,
            return_intensity_scaled DOUBLE,
            consecutive_returns_scaled DOUBLE,
            avg_consecutive_returns_scaled DOUBLE,
            customer_lifetime_days_scaled DOUBLE,
            avg_days_to_return_scaled DOUBLE,
            return_timing_spread_scaled DOUBLE,
            recent_orders_scaled DOUBLE,
            recent_returns_scaled DOUBLE,
            recent_vs_avg_ratio_scaled DOUBLE,
            behavior_stability_score_scaled DOUBLE,
            product_category_loyalty_scaled DOUBLE,
            category_diversity_score_scaled DOUBLE,
            category_loyalty_score_scaled DOUBLE,
            high_return_category_affinity_scaled DOUBLE,
            sku_adjacency_orders_scaled DOUBLE,
            sku_adjacency_returns_scaled DOUBLE,
            sku_adjacency_timing_scaled DOUBLE,
            sku_adjacency_return_timing_scaled DOUBLE,
            seasonal_susceptibility_returns_scaled DOUBLE,
            seasonal_susceptibility_orders_scaled DOUBLE,
            trend_product_category_return_rate_scaled DOUBLE,
            trend_product_category_order_rate_scaled DOUBLE,
            
            -- Hey Claude! Look here for the updated data that includes order values!
            -- MONETARY FEATURES (PLACEHOLDER - SCALED)
            avg_order_value_scaled DOUBLE DEFAULT NULL,
            avg_return_value_scaled DOUBLE DEFAULT NULL,
            high_value_return_affinity_scaled DOUBLE DEFAULT NULL,
            
            -- Clustering metadata
            outlier_score DOUBLE,
            feature_completeness_score DOUBLE,
            processing_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            scaling_method VARCHAR DEFAULT 'RobustScaler',
            data_quality_flags TEXT DEFAULT ''
        );
        """
        
        # Create helper tables
        helper_tables = """
        -- Email consolidation tracking
        CREATE TABLE IF NOT EXISTS email_consolidation_candidates (
            email_group_id INTEGER,
            customer_emailid VARCHAR,
            customer_emailid_cleaned VARCHAR,
            similarity_score DOUBLE,
            consolidation_suggested BOOLEAN DEFAULT FALSE,
            review_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Data quality issues tracking
        CREATE TABLE IF NOT EXISTS data_quality_issues (
            issue_id INTEGER PRIMARY KEY,
            issue_type VARCHAR,
            issue_description TEXT,
            affected_records INTEGER,
            severity VARCHAR, -- HIGH, MEDIUM, LOW
            detection_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            resolved BOOLEAN DEFAULT FALSE
        );
        
        -- Feature calculation logs
        CREATE TABLE IF NOT EXISTS feature_calculation_log (
            log_id INTEGER PRIMARY KEY,
            feature_name VARCHAR,
            calculation_start TIMESTAMP,
            calculation_end TIMESTAMP,
            execution_time_seconds DOUBLE,
            records_processed INTEGER,
            warnings_count INTEGER,
            errors_count INTEGER,
            status VARCHAR DEFAULT 'PENDING'
        );
        """
        
        # Execute DDL statements
        for ddl in [bronze_ddl, silver_ddl, gold_ddl, helper_tables]:
            self.conn.execute(ddl)
        
        logger.info("Database schema created successfully")
    
    def load_bronze_data(self, csv_path: str, chunk_size: int = 50000):
        """Load CSV data into bronze layer with data quality checks"""
        logger.info(f"Loading data from {csv_path} into bronze layer")
        # Read CSV in chunks for memory efficiency
        total_rows = 0
        data_quality_issues = []

        # Define the exact column order and types for bronze table
        bronze_columns = [
            'primary_key', 'sales_order_no', 'customer_emailid', 'order_date', 'sku',
            'sales_qty', 'return_qty', 'units_returned_flag', 'return_date', 'return_no',
            'return_comment', 'orderlink', 'q_cls_id', 'q_sku_desc', 'q_gmm_id', 'q_sku_id',
            'class_', 'division_', 'brand_', 'q_clr_dnum', 'q_clr_desc', 'vendor_style', 'size_'
        ]
        bronze_types = {
            'primary_key': str,
            'sales_order_no': str,
            'customer_emailid': str,
            'order_date': 'datetime64[ns]',
            'sku': str,
            'sales_qty': 'Int64',
            'return_qty': 'Int64',
            'units_returned_flag': str,
            'return_date': 'datetime64[ns]',
            'return_no': str,
            'return_comment': str,
            'orderlink': str,
            'q_cls_id': str,
            'q_sku_desc': str,
            'q_gmm_id': str,
            'q_sku_id': str,
            'class_': str,
            'division_': str,
            'brand_': str,
            'q_clr_dnum': str,
            'q_clr_desc': str,
            'vendor_style': str,
            'size_': str
        }

        for chunk_num, chunk in enumerate(pd.read_csv(csv_path, chunksize=chunk_size)):
            logger.info(f"Processing chunk {chunk_num + 1}, rows: {len(chunk)}")
            # Clean and standardize data
            chunk_processed = self._clean_bronze_data(chunk, data_quality_issues)
            # Add primary_key if not present
            if 'primary_key' not in chunk_processed.columns:
                chunk_processed['primary_key'] = chunk_processed['q_sku_id'].astype(str) + '_' + chunk_processed['sales_order_no'].astype(str)
            # Reorder columns and cast types
            for col in bronze_columns:
                if col not in chunk_processed.columns:
                    chunk_processed[col] = pd.NA
            chunk_processed = chunk_processed[bronze_columns]
            for col, typ in bronze_types.items():
                if typ == 'datetime64[ns]':
                    chunk_processed[col] = pd.to_datetime(chunk_processed[col], errors='coerce')
                elif typ == 'Int64':
                    chunk_processed[col] = pd.to_numeric(chunk_processed[col], errors='coerce').astype('Int64')
                else:
                    chunk_processed[col] = chunk_processed[col].astype(str).fillna('')
            # Insert into database
            self.conn.register('chunk_processed', chunk_processed)
            self.conn.execute("INSERT OR REPLACE INTO bronze_return_order_data SELECT * FROM chunk_processed")
            total_rows += len(chunk_processed)
        # Log data quality issues
        self._log_data_quality_issues(data_quality_issues)
        logger.info(f"Loaded {total_rows} rows into bronze_return_order_data")
        return total_rows
    
    def _clean_bronze_data(self, df: pd.DataFrame, issues_list: List) -> pd.DataFrame:
        """Clean and validate bronze layer data"""
        df_clean = df.copy()
        
        df_clean['primary_key'] = df_clean['Q_SKU_ID'].astype(str) + '_' + df_clean['SALES_ORDER_NO'].astype(str)
        
        # Standardize column names and ensure we have all expected columns
        df_clean = df_clean.rename(columns={col: col.lower() for col in df_clean.columns})

        # Ensure we have exactly the columns we expect
        expected_columns = [
            'sales_order_no', 'customer_emailid', 'order_date', 'sku', 'sales_qty',
            'return_qty', 'units_returned_flag', 'return_date', 'return_no', 'return_comment',
            'orderlink', 'q_cls_id', 'q_sku_desc', 'q_gmm_id', 'q_sku_id', 'class_',
            'division_', 'brand_', 'q_clr_dnum', 'q_clr_desc', 'vendor_style', 'size_'
        ]

        # Keep only the columns we expect (in case there are extras)
        available_columns = [col for col in expected_columns if col in df_clean.columns]
        df_clean = df_clean[available_columns]
        
        # Clean email addresses - convert to lowercase
        df_clean['customer_emailid'] = df_clean['customer_emailid'].str.lower().str.strip()
        
        # Parse dates
        try:
            df_clean['order_date'] = pd.to_datetime(df_clean['order_date'])
            df_clean['return_date'] = pd.to_datetime(df_clean['return_date'])
        except Exception as e:
            issues_list.append(('DATE_PARSING', f'Date parsing error: {str(e)}', len(df_clean), 'MEDIUM'))
        
        # Fix return dates before order dates
        invalid_return_dates = df_clean['return_date'] < df_clean['order_date']
        if invalid_return_dates.sum() > 0:
            issues_list.append(('INVALID_RETURN_DATES', 
                              f'Return dates before order dates: {invalid_return_dates.sum()} records', 
                              invalid_return_dates.sum(), 'HIGH'))
            # Fix: set return_date to order_date + 1 day
            df_clean.loc[invalid_return_dates, 'return_date'] = df_clean.loc[invalid_return_dates, 'order_date'] + pd.Timedelta(days=1)
        
        # Handle missing return dates when return_qty > 0
        missing_return_dates = (df_clean['return_qty'] > 0) & df_clean['return_date'].isna()
        if missing_return_dates.sum() > 0:
            issues_list.append(('MISSING_RETURN_DATES', 
                              f'Missing return dates with return qty > 0: {missing_return_dates.sum()} records', 
                              missing_return_dates.sum(), 'MEDIUM'))
            # Fix: set return_date to order_date + 1 day
            df_clean.loc[missing_return_dates, 'return_date'] = df_clean.loc[missing_return_dates, 'order_date'] + pd.Timedelta(days=1)
        
        # Add data quality flags
        df_clean['data_quality_flags'] = ''
        df_clean.loc[invalid_return_dates, 'data_quality_flags'] += 'RETURN_DATE_FIXED;'
        df_clean.loc[missing_return_dates, 'data_quality_flags'] += 'RETURN_DATE_IMPUTED;'
        
        return df_clean
    
    def _log_data_quality_issues(self, issues_list: List):
        """Log data quality issues to tracking table"""
        for issue_type, description, affected_records, severity in issues_list:
            self.conn.execute("""
                INSERT INTO data_quality_issues (issue_type, issue_description, affected_records, severity)
                VALUES (?, ?, ?, ?)
            """, [issue_type, description, affected_records, severity])
        
        logger.info(f"Logged {len(issues_list)} data quality issues")
    
    def get_similar_emails(self, similarity_threshold: float = 0.8) -> pd.DataFrame:
        """Find similar email addresses for consolidation review"""
        query = """
        WITH email_pairs AS (
            SELECT DISTINCT 
                a.customer_emailid as email1,
                b.customer_emailid as email2,
                -- Simple similarity: Levenshtein distance
                (1.0 - CAST(levenshtein(a.customer_emailid, b.customer_emailid) AS DOUBLE) / 
                 GREATEST(length(a.customer_emailid), length(b.customer_emailid))) as similarity
            FROM bronze_return_order_data a
            CROSS JOIN bronze_return_order_data b
            WHERE a.customer_emailid != b.customer_emailid
            AND a.customer_emailid < b.customer_emailid  -- Avoid duplicates
        )
        SELECT email1, email2, similarity
        FROM email_pairs 
        WHERE similarity >= ?
        ORDER BY similarity DESC;
        """
        
        try:
            result = self.conn.execute(query, [similarity_threshold]).fetchdf()
            logger.info(f"Found {len(result)} email pairs with similarity >= {similarity_threshold}")
            return result
        except Exception as e:
            logger.warning(f"Email similarity analysis failed: {e}")
            # Fallback: simple case-based grouping
            return self._simple_email_grouping()
    
    def _simple_email_grouping(self) -> pd.DataFrame:
        """Fallback email grouping based on case differences"""
        query = """
        SELECT 
            lower(customer_emailid) as email_standardized,
            list(DISTINCT customer_emailid) as email_variants,
            count(DISTINCT customer_emailid) as variant_count
        FROM bronze_return_order_data
        GROUP BY lower(customer_emailid)
        HAVING count(DISTINCT customer_emailid) > 1
        ORDER BY variant_count DESC;
        """
        
        result = self.conn.execute(query).fetchdf()
        logger.info(f"Found {len(result)} email groups with case variations")
        return result
    
    def get_data_summary(self) -> dict:
        """Get summary statistics for all layers"""
        summary = {}
        
        # Bronze layer summary
        bronze_stats = self.conn.execute("""
            SELECT 
                count(*) as total_records,
                count(DISTINCT customer_emailid) as unique_customers,
                count(DISTINCT sales_order_no) as unique_orders,
                count(DISTINCT q_sku_id) as unique_skus,
                min(order_date) as earliest_order,
                max(order_date) as latest_order,
                sum(sales_qty) as total_sales_qty,
                sum(return_qty) as total_return_qty,
                count(*) FILTER (WHERE return_qty > 0) as records_with_returns
            FROM bronze_return_order_data;
        """).fetchone()
        
        summary['bronze'] = dict(zip([
            'total_records', 'unique_customers', 'unique_orders', 'unique_skus',
            'earliest_order', 'latest_order', 'total_sales_qty', 'total_return_qty',
            'records_with_returns'
        ], bronze_stats))
        
        # Silver layer summary (if exists)
        try:
            silver_stats = self.conn.execute("""
                SELECT count(*) as customer_features_count
                FROM silver_customer_features;
            """).fetchone()
            summary['silver'] = {'customer_features_count': silver_stats[0]}
        except:
            summary['silver'] = {'customer_features_count': 0}
        
        # Gold layer summary (if exists)
        try:
            gold_stats = self.conn.execute("""
                SELECT count(*) as processed_customers_count
                FROM gold_cluster_processed;
            """).fetchone()
            summary['gold'] = {'processed_customers_count': gold_stats[0]}
        except:
            summary['gold'] = {'processed_customers_count': 0}
        
        return summary
    
    def close(self):
        """Close database connection"""
        self.conn.close()
        logger.info("Database connection closed")

# Convenience functions
def setup_database(csv_path: Optional[str] = None, db_path: str = "customer_clustering.db") -> CustomerClusteringDB:
    """Setup database and optionally load initial data"""
    db = CustomerClusteringDB(db_path)
    
    if csv_path and Path(csv_path).exists():
        db.load_bronze_data(csv_path)
    
    return db

def get_connection(db_path: str = "customer_clustering.db") -> duckdb.DuckDBPyConnection:
    """Get a simple DuckDB connection"""
    return duckdb.connect(db_path)

if __name__ == "__main__":
    # Example usage
    db = setup_database("paste.txt")  # Replace with actual CSV path
    
    print("Database Summary:")
    summary = db.get_data_summary()
    for layer, stats in summary.items():
        print(f"\n{layer.upper()} Layer:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
    
    print("\nSimilar Emails:")
    similar_emails = db.get_similar_emails()
    print(similar_emails.head())
    
    db.close()
