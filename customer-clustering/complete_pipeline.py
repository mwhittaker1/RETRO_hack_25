"""
Complete Pipeline for Importing, Processing, and Checking Data
This script:
1. Imports new CSV data to bronze layer
2. Updates silver layer features
3. Updates gold layer scaling
4. Checks for duplicates in all layers
5. Exports results
"""

import duckdb
import pandas as pd
import os
import sys
import logging
from datetime import datetime
from pathlib import Path
import time

# Setup logging
log_filename = f'complete_pipeline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Constants
DB_PATH = "customer_clustering.db"
NEW_DATA_PATH = "../data/random1_FINAL_SENT.csv"  # Path to new CSV file
OUTPUT_DIR = "."
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

def get_database_connection():
    """Connect to the database"""
    try:
        conn = duckdb.connect(DB_PATH)
        logger.info(f"Connected to database: {DB_PATH}")
        return conn
    except Exception as e:
        logger.error(f"Error connecting to database: {str(e)}")
        raise

def check_for_duplicates(conn):
    """Check for duplicates in all layers and report"""
    try:
        logger.info("Checking for duplicates in all layers")
        
        # Check bronze layer
        bronze_total = conn.execute("SELECT COUNT(*) FROM bronze_return_order_data").fetchone()[0]
        bronze_unique = conn.execute("SELECT COUNT(DISTINCT primary_key) FROM bronze_return_order_data").fetchone()[0]
        bronze_dups = bronze_total - bronze_unique
        
        logger.info(f"Bronze layer: {bronze_total} total rows, {bronze_unique} unique primary keys, {bronze_dups} duplicates")
        
        if bronze_dups > 0:
            # Find duplicate primary keys
            bronze_dup_keys = conn.execute("""
            SELECT primary_key, COUNT(*) as count
            FROM bronze_return_order_data
            GROUP BY primary_key
            HAVING COUNT(*) > 1
            ORDER BY COUNT(*) DESC
            LIMIT 10
            """).fetchdf()
            
            logger.warning(f"Found duplicate primary keys in bronze layer, sample: {bronze_dup_keys}")
            
            # Remove duplicates
            logger.info("Removing duplicates from bronze layer")
            conn.execute("""
            CREATE TABLE bronze_temp AS
            SELECT DISTINCT * FROM bronze_return_order_data
            """)
            
            # Swap tables
            conn.execute("ALTER TABLE bronze_return_order_data RENAME TO bronze_old")
            conn.execute("ALTER TABLE bronze_temp RENAME TO bronze_return_order_data")
            conn.execute("DROP TABLE bronze_old")
            
            # Count after dedupe
            bronze_after = conn.execute("SELECT COUNT(*) FROM bronze_return_order_data").fetchone()[0]
            logger.info(f"Bronze layer after deduplication: {bronze_after} rows")
        
        # Check silver layer
        silver_total = conn.execute("SELECT COUNT(*) FROM silver_customer_features").fetchone()[0]
        silver_unique = conn.execute("SELECT COUNT(DISTINCT customer_emailid) FROM silver_customer_features").fetchone()[0]
        silver_dups = silver_total - silver_unique
        
        logger.info(f"Silver layer: {silver_total} total rows, {silver_unique} unique emails, {silver_dups} duplicates")
        
        if silver_dups > 0:
            # Find duplicate emails
            silver_dup_emails = conn.execute("""
            SELECT customer_emailid, COUNT(*) as count
            FROM silver_customer_features
            GROUP BY customer_emailid
            HAVING COUNT(*) > 1
            ORDER BY COUNT(*) DESC
            LIMIT 10
            """).fetchdf()
            
            logger.warning(f"Found duplicate emails in silver layer, sample: {silver_dup_emails}")
            
            # Remove duplicates
            logger.info("Removing duplicates from silver layer")
            conn.execute("""
            CREATE TABLE silver_temp AS
            SELECT DISTINCT * FROM silver_customer_features
            """)
            
            # Swap tables
            conn.execute("ALTER TABLE silver_customer_features RENAME TO silver_old")
            conn.execute("ALTER TABLE silver_temp RENAME TO silver_customer_features")
            conn.execute("DROP TABLE silver_old")
            
            # Count after dedupe
            silver_after = conn.execute("SELECT COUNT(*) FROM silver_customer_features").fetchone()[0]
            logger.info(f"Silver layer after deduplication: {silver_after} rows")
        
        # Check gold layer
        gold_total = conn.execute("SELECT COUNT(*) FROM gold_cluster_processed").fetchone()[0]
        gold_unique = conn.execute("SELECT COUNT(DISTINCT customer_emailid) FROM gold_cluster_processed").fetchone()[0]
        gold_dups = gold_total - gold_unique
        
        logger.info(f"Gold layer: {gold_total} total rows, {gold_unique} unique emails, {gold_dups} duplicates")
        
        if gold_dups > 0:
            # Find duplicate emails
            gold_dup_emails = conn.execute("""
            SELECT customer_emailid, COUNT(*) as count
            FROM gold_cluster_processed
            GROUP BY customer_emailid
            HAVING COUNT(*) > 1
            ORDER BY COUNT(*) DESC
            LIMIT 10
            """).fetchdf()
            
            logger.warning(f"Found duplicate emails in gold layer, sample: {gold_dup_emails}")
            
            # Remove duplicates
            logger.info("Removing duplicates from gold layer")
            conn.execute("""
            CREATE TABLE gold_temp AS
            SELECT DISTINCT * FROM gold_cluster_processed
            """)
            
            # Swap tables
            conn.execute("ALTER TABLE gold_cluster_processed RENAME TO gold_old")
            conn.execute("ALTER TABLE gold_temp RENAME TO gold_cluster_processed")
            conn.execute("DROP TABLE gold_old")
            
            # Count after dedupe
            gold_after = conn.execute("SELECT COUNT(*) FROM gold_cluster_processed").fetchone()[0]
            logger.info(f"Gold layer after deduplication: {gold_after} rows")
        
        return True
    except Exception as e:
        logger.error(f"Error checking for duplicates: {str(e)}")
        return False

def import_to_bronze_layer(conn):
    """Import new CSV data to bronze layer"""
    try:
        logger.info(f"Importing data from {NEW_DATA_PATH}")
        
        # Check if file exists
        if not os.path.exists(NEW_DATA_PATH):
            logger.error(f"File not found: {NEW_DATA_PATH}")
            return False
        
        # Count rows before import
        before_count = conn.execute("SELECT COUNT(*) FROM bronze_return_order_data").fetchone()[0]
        logger.info(f"Bronze layer has {before_count} rows before import")
        
        # Create temporary table
        conn.execute("DROP TABLE IF EXISTS temp_import")
        conn.execute("DROP TABLE IF EXISTS temp_formatted")
        
        # Load CSV directly into temporary table
        logger.info("Loading CSV into temporary table")
        load_query = f"""
        CREATE TABLE temp_import AS 
        SELECT * FROM read_csv_auto('{NEW_DATA_PATH}', header=true, sample_size=1000, 
                                   all_varchar=true)
        """
        
        conn.execute(load_query)
        
        # Check columns in the temporary table
        temp_columns = conn.execute("PRAGMA table_info(temp_import);").fetchdf()
        logger.info(f"Temporary table has {len(temp_columns)} columns")
        
        # Map columns to bronze layer format and generate primary key
        logger.info("Mapping columns and generating primary keys")
        
        # Get bronze layer columns
        bronze_columns = conn.execute("PRAGMA table_info(bronze_return_order_data);").fetchdf()
        bronze_column_names = bronze_columns['name'].tolist()
        logger.info(f"Bronze layer has {len(bronze_column_names)} columns: {bronze_column_names}")
        
        # Check which columns from temp_import match bronze layer
        temp_column_names = temp_columns['name'].tolist()
        matching_columns = [col for col in bronze_column_names if col.lower() in [c.lower() for c in temp_column_names]]
        
        # Generate mapping SQL
        column_mapping = []
        for bronze_col in bronze_column_names:
            if bronze_col.lower() == 'primary_key':
                # Generate primary key from other columns
                column_mapping.append(f"CONCAT(CAST(COALESCE(sales_order_no, '') AS VARCHAR), '-', CAST(COALESCE(sku, '') AS VARCHAR), '-', CAST(COALESCE(customer_emailid, '') AS VARCHAR)) AS primary_key")
            else:
                # Look for matching column (case-insensitive)
                matched = False
                for temp_col in temp_column_names:
                    if temp_col.lower() == bronze_col.lower():
                        # Handle special cases
                        if bronze_col.lower() == 'units_returned_flag':
                            column_mapping.append(f"CAST({temp_col} AS VARCHAR) AS {bronze_col}")
                        elif 'date' in bronze_col.lower():
                            column_mapping.append(f"TRY_CAST({temp_col} AS TIMESTAMP) AS {bronze_col}")
                        elif bronze_col.lower() in ['sales_qty', 'return_qty']:
                            column_mapping.append(f"TRY_CAST({temp_col} AS INTEGER) AS {bronze_col}")
                        elif bronze_col.lower() == 'gross':
                            column_mapping.append(f"TRY_CAST({temp_col} AS DOUBLE) AS {bronze_col}")
                        else:
                            column_mapping.append(f"{temp_col} AS {bronze_col}")
                        matched = True
                        break
                
                if not matched:
                    # Column not found, use NULL
                    column_mapping.append(f"NULL AS {bronze_col}")
        
        # Create properly formatted table with all needed columns
        logger.info("Creating formatted table with all required columns")
        format_query = f"""
        CREATE TABLE temp_formatted AS
        SELECT
            {', '.join(column_mapping)}
        FROM temp_import
        """
        
        conn.execute(format_query)
        
        # Insert into bronze layer, avoiding duplicates
        logger.info("Inserting data into bronze layer")
        insert_query = """
        INSERT OR IGNORE INTO bronze_return_order_data
        SELECT * FROM temp_formatted
        """
        
        conn.execute(insert_query)
        
        # Count rows after import
        after_count = conn.execute("SELECT COUNT(*) FROM bronze_return_order_data").fetchone()[0]
        rows_added = after_count - before_count
        
        logger.info(f"Added {rows_added} new rows to bronze layer")
        
        # Clean up temporary tables
        conn.execute("DROP TABLE IF EXISTS temp_import")
        conn.execute("DROP TABLE IF EXISTS temp_formatted")
        
        return rows_added > 0
    except Exception as e:
        logger.error(f"Error importing to bronze layer: {str(e)}")
        return False

def update_silver_layer(conn):
    """Update silver layer with new features"""
    try:
        logger.info("Updating silver layer features")
        
        # Get unique category count for diversity score
        unique_categories = conn.execute("SELECT COUNT(DISTINCT class_) FROM bronze_return_order_data").fetchone()[0]
        if unique_categories <= 0:
            unique_categories = 20  # Default if no categories found
        
        logger.info(f"Found {unique_categories} unique product categories in bronze layer")
        
        # Update the category diversity score
        logger.info("Updating category_diversity_score")
        diversity_query = f"""
        -- Create a temporary table with customer_emailid and new category_diversity_score
        CREATE TEMPORARY TABLE temp_category_diversity AS
        WITH customer_categories AS (
            SELECT 
                customer_emailid,
                class_ as category,
                COUNT(*) as category_purchases,
                SUM(CASE WHEN return_qty > 0 THEN 1 ELSE 0 END) as category_returns,
                SUM(sales_qty) as category_sales
            FROM bronze_return_order_data
            GROUP BY customer_emailid, class_
        ),
        customer_totals AS (
            SELECT 
                customer_emailid,
                COUNT(DISTINCT category) as unique_categories,
                SUM(category_purchases) as total_purchases,
                SUM(category_returns) as total_returns
            FROM customer_categories
            GROUP BY customer_emailid
        )
        SELECT 
            customer_emailid,
            -- Update formula to use actual unique categories count
            CAST(unique_categories AS DOUBLE) / {unique_categories}.0 as category_diversity_score
        FROM customer_totals;
        """
        
        conn.execute(diversity_query)
        
        # Find new customers not yet in silver layer
        logger.info("Identifying new customers to add to silver layer")
        new_customers_query = """
        SELECT DISTINCT b.customer_emailid
        FROM bronze_return_order_data b
        LEFT JOIN silver_customer_features s ON b.customer_emailid = s.customer_emailid
        WHERE s.customer_emailid IS NULL
        """
        
        new_customers_df = conn.execute(new_customers_query).fetchdf()
        new_customer_count = len(new_customers_df)
        logger.info(f"Found {new_customer_count} new customers to add to silver layer")
        
        # Update existing customers in silver layer
        logger.info("Updating existing customers in silver layer")
        update_query = """
        UPDATE silver_customer_features
        SET category_diversity_score = t.category_diversity_score
        FROM temp_category_diversity t
        WHERE silver_customer_features.customer_emailid = t.customer_emailid
        """
        
        conn.execute(update_query)
        
        # Count updated rows
        updated_rows = conn.execute("""
            SELECT COUNT(*) 
            FROM silver_customer_features s
            JOIN temp_category_diversity t ON s.customer_emailid = t.customer_emailid
        """).fetchone()[0]
        
        logger.info(f"Updated {updated_rows} existing customers in silver layer")
        
        # For new customers, we need to run full feature engineering
        # This is a placeholder - in a real implementation, you would call 
        # the complete feature engineering process for new customers
        
        # Drop temporary table
        conn.execute("DROP TABLE IF EXISTS temp_category_diversity")
        
        return True
    except Exception as e:
        logger.error(f"Error updating silver layer: {str(e)}")
        return False

def update_gold_layer(conn):
    """Update gold layer with scaled features"""
    try:
        logger.info("Updating gold layer scaled features")
        
        # Update the scaled diversity score
        logger.info("Updating category_diversity_score_scaled in gold layer")
        
        # Update the gold layer to match the updated silver layer
        update_query = """
        UPDATE gold_cluster_processed g
        SET category_diversity_score_scaled = (
            SELECT (s.category_diversity_score - MIN(s2.category_diversity_score)) / 
                  (MAX(s2.category_diversity_score) - MIN(s2.category_diversity_score))
            FROM silver_customer_features s, silver_customer_features s2
            WHERE s.customer_emailid = g.customer_emailid
            AND s2.category_diversity_score > 0
            GROUP BY s.customer_emailid, s.category_diversity_score
        )
        WHERE EXISTS (
            SELECT 1 FROM silver_customer_features s
            WHERE s.customer_emailid = g.customer_emailid
            AND s.category_diversity_score > 0
        )
        """
        
        conn.execute(update_query)
        
        # Get number of updated rows
        updated_rows = conn.execute("""
        SELECT COUNT(*) FROM gold_cluster_processed
        WHERE category_diversity_score_scaled IS NOT NULL
        """).fetchone()[0]
        
        logger.info(f"Updated category_diversity_score_scaled for {updated_rows} customers in gold layer")
        
        return True
    except Exception as e:
        logger.error(f"Error updating gold layer: {str(e)}")
        return False

def export_layers(conn):
    """Export silver and gold layers to CSV and Excel"""
    try:
        logger.info("Exporting silver and gold layers")
        
        # Export silver layer
        silver_csv = f"silver_customer_features_{TIMESTAMP}.csv"
        logger.info(f"Exporting silver layer to {silver_csv}")
        
        silver_df = conn.execute("SELECT * FROM silver_customer_features").fetchdf()
        silver_df.to_csv(silver_csv, index=False)
        logger.info(f"Exported {len(silver_df)} rows to {silver_csv}")
        
        # Export gold layer
        gold_csv = f"gold_cluster_processed_{TIMESTAMP}.csv"
        logger.info(f"Exporting gold layer to {gold_csv}")
        
        gold_df = conn.execute("SELECT * FROM gold_cluster_processed").fetchdf()
        gold_df.to_csv(gold_csv, index=False)
        logger.info(f"Exported {len(gold_df)} rows to {gold_csv}")
        
        return silver_csv, gold_csv
    except Exception as e:
        logger.error(f"Error exporting layers: {str(e)}")
        return None, None

def process_new_customers(conn):
    """Process new customers from bronze layer into silver layer"""
    try:
        logger.info("Processing new customers into silver layer")
        
        # Identify new customers
        new_customers_query = """
        SELECT DISTINCT b.customer_emailid
        FROM bronze_return_order_data b
        LEFT JOIN silver_customer_features s ON b.customer_emailid = s.customer_emailid
        WHERE s.customer_emailid IS NULL
        """
        
        new_customers_df = conn.execute(new_customers_query).fetchdf()
        new_customer_count = len(new_customers_df)
        
        if new_customer_count == 0:
            logger.info("No new customers to process")
            return True
        
        logger.info(f"Processing {new_customer_count} new customers into silver layer")
        
        # Get basic features for new customers
        logger.info("Calculating basic features for new customers")
        
        # Create a temporary table with category diversity scores for all customers
        logger.info("Calculating category diversity scores")
        unique_categories = conn.execute("SELECT COUNT(DISTINCT class_) FROM bronze_return_order_data").fetchone()[0]
        
        diversity_query = f"""
        CREATE TEMPORARY TABLE temp_new_customer_features AS
        WITH customer_categories AS (
            SELECT 
                customer_emailid,
                class_ as category,
                COUNT(*) as category_purchases,
                SUM(CASE WHEN return_qty > 0 THEN 1 ELSE 0 END) as category_returns,
                SUM(sales_qty) as category_sales
            FROM bronze_return_order_data
            GROUP BY customer_emailid, class_
            HAVING customer_emailid IN (
                SELECT customer_emailid FROM ({new_customers_query})
            )
        ),
        customer_totals AS (
            SELECT 
                customer_emailid,
                COUNT(DISTINCT category) as unique_categories,
                SUM(category_purchases) as total_purchases,
                SUM(category_returns) as total_returns
            FROM customer_categories
            GROUP BY customer_emailid
        ),
        customer_orders AS (
            SELECT 
                customer_emailid,
                COUNT(DISTINCT sales_order_no) as order_count,
                MIN(order_date) as first_order_date,
                MAX(order_date) as last_order_date
            FROM bronze_return_order_data
            GROUP BY customer_emailid
            HAVING customer_emailid IN (
                SELECT customer_emailid FROM ({new_customers_query})
            )
        ),
        customer_returns AS (
            SELECT 
                customer_emailid,
                COUNT(DISTINCT return_no) as return_count,
                COUNT(DISTINCT CASE WHEN return_qty > 0 THEN sales_order_no ELSE NULL END) as orders_with_returns
            FROM bronze_return_order_data
            GROUP BY customer_emailid
            HAVING customer_emailid IN (
                SELECT customer_emailid FROM ({new_customers_query})
            )
        )
        SELECT 
            t.customer_emailid,
            CAST(t.unique_categories AS DOUBLE) / {unique_categories}.0 as category_diversity_score,
            o.order_count as sales_order_no_nunique,
            t.unique_categories as sku_nunique,
            r.return_count as items_returned_count,
            t.total_returns as return_qty_sum,
            t.total_purchases as sales_qty_sum,
            CASE WHEN o.order_count > 0 THEN CAST(t.total_purchases AS DOUBLE) / o.order_count ELSE 0 END as sales_qty_mean,
            CASE WHEN t.total_purchases > 0 THEN CAST(t.total_returns AS DOUBLE) / t.total_purchases ELSE 0 END as return_rate,
            CASE WHEN o.order_count > 0 THEN CAST(r.return_count AS DOUBLE) / o.order_count ELSE 0 END as avg_returns_per_order,
            CASE WHEN o.order_count > 0 THEN CAST(r.orders_with_returns AS DOUBLE) / o.order_count ELSE 0 END as return_frequency_ratio,
            DATEDIFF('day', o.first_order_date, o.last_order_date) as customer_lifetime_days
        FROM customer_totals t
        JOIN customer_orders o ON t.customer_emailid = o.customer_emailid
        JOIN customer_returns r ON t.customer_emailid = r.customer_emailid
        """
        
        conn.execute(diversity_query)
        
        # Insert new customers into silver layer
        logger.info("Inserting new customers into silver layer")
        
        # Get silver layer columns
        silver_columns = conn.execute("PRAGMA table_info(silver_customer_features);").fetchdf()
        silver_column_names = silver_columns['name'].tolist()
        
        # Create column list for insert
        temp_columns = conn.execute("PRAGMA table_info(temp_new_customer_features);").fetchdf()
        temp_column_names = temp_columns['name'].tolist()
        
        # Map temp columns to silver columns
        column_mapping = []
        for silver_col in silver_column_names:
            if silver_col in temp_column_names:
                column_mapping.append(f"t.{silver_col}")
            else:
                # Use NULL for missing columns
                column_mapping.append("NULL")
        
        # Insert into silver layer
        insert_query = f"""
        INSERT INTO silver_customer_features
        SELECT {', '.join(column_mapping)}
        FROM temp_new_customer_features t
        """
        
        conn.execute(insert_query)
        
        # Count inserted rows
        inserted_count = conn.execute(f"""
        SELECT COUNT(*) FROM silver_customer_features
        WHERE customer_emailid IN (
            SELECT customer_emailid FROM ({new_customers_query})
        )
        """).fetchone()[0]
        
        logger.info(f"Inserted {inserted_count} new customers into silver layer")
        
        # Drop temporary table
        conn.execute("DROP TABLE IF EXISTS temp_new_customer_features")
        
        return True
    except Exception as e:
        logger.error(f"Error processing new customers: {str(e)}")
        return False

def run_pipeline():
    """Run the complete pipeline"""
    start_time = time.time()
    logger.info("Starting complete pipeline")
    
    try:
        # Connect to database
        conn = get_database_connection()
        
        # 1. Check for and remove duplicates
        if not check_for_duplicates(conn):
            logger.error("Failed to check for duplicates")
            return False
        
        # 2. Import new data to bronze layer
        if not import_to_bronze_layer(conn):
            logger.error("Failed to import to bronze layer")
            return False
        
        # 3. Update silver layer
        if not update_silver_layer(conn):
            logger.error("Failed to update silver layer")
            return False
            
        # 3b. Process new customers into silver layer
        if not process_new_customers(conn):
            logger.warning("Warning: Failed to process new customers into silver layer")
            # Continue processing - this is not a critical failure
        
        # 4. Update gold layer
        if not update_gold_layer(conn):
            logger.error("Failed to update gold layer")
            return False
        
        # 5. Check for duplicates again after all operations
        if not check_for_duplicates(conn):
            logger.warning("Warning: Duplicate check after updates failed")
        
        # 6. Export layers
        silver_csv, gold_csv = export_layers(conn)
        if silver_csv and gold_csv:
            logger.info(f"Successfully exported silver and gold layers")
        else:
            logger.warning("Warning: Layer export failed")
        
        end_time = time.time()
        total_time = end_time - start_time
        logger.info(f"Complete pipeline finished in {total_time:.2f} seconds")
        
        print("\n========== PIPELINE SUMMARY ==========")
        print(f"Pipeline completed successfully in {total_time:.2f} seconds")
        print(f"Log file: {log_filename}")
        if silver_csv and gold_csv:
            print(f"Silver layer exported to: {silver_csv}")
            print(f"Gold layer exported to: {gold_csv}")
        print("======================================")
        
        return True
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        return False
    finally:
        if 'conn' in locals():
            conn.close()
            logger.info("Database connection closed")

if __name__ == "__main__":
    run_pipeline()
