"""
Update and Export Pipeline Runner
This script updates the database with new sentiment data, recalculates product diversity scores,
and exports the updated silver layer to CSV.
"""

import os
import sys
import logging
import pandas as pd
import duckdb
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'update_database_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Constants
DB_PATH = "customer_clustering.db"
NEW_DATA_PATH = "sentiment_and_raw/new_base_returns_sku_reasoncodes_sent.csv"
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

def import_new_data(conn):
    """Import the new sentiment data to a temporary table"""
    try:
        logger.info(f"Importing new data from: {NEW_DATA_PATH}")
        
        # Check if file exists
        if not os.path.exists(NEW_DATA_PATH):
            logger.error(f"File not found: {NEW_DATA_PATH}")
            return False
        
        # Read the CSV file
        try:
            df = pd.read_csv(NEW_DATA_PATH, low_memory=False)
            logger.info(f"Read {len(df)} rows from CSV file")
            
            # Check column count
            expected_cols = 24  # Number of columns in bronze layer
            if len(df.columns) != expected_cols:
                logger.warning(f"Column count mismatch. Expected {expected_cols}, got {len(df.columns)}")
        except Exception as e:
            logger.error(f"Error reading CSV file: {str(e)}")
            return False
            
        # Clean column names and create a clean DataFrame with required columns
        df_clean = pd.DataFrame()
        
        # Generate a primary key column if it doesn't exist
        if 'primary_key' not in df.columns:
            logger.warning("Added empty column primary_key")
            df_clean['primary_key'] = None
        
        # Map columns from source to destination (case-insensitive)
        bronze_columns = [
            'sales_order_no', 'customer_emailid', 'order_date', 'sku',
            'sales_qty', 'gross', 'return_qty', 'units_returned_flag',
            'return_date', 'return_no', 'return_comment', 'orderlink',
            'q_cls_id', 'q_sku_desc', 'q_gmm_id', 'q_sku_id', 'class_',
            'division_', 'brand_', 'q_clr_dnum', 'q_clr_desc', 'vendor_style', 
            'size_'
        ]
        
        # Map columns case-insensitively
        for col in bronze_columns:
            mapped = False
            for source_col in df.columns:
                if source_col.lower() == col.lower():
                    logger.info(f"Mapped column {source_col} to {col}")
                    df_clean[col] = df[source_col]
                    mapped = True
                    break
            
            if not mapped:
                logger.warning(f"Could not find match for column: {col}")
                df_clean[col] = None
        
        # Generate primary keys
        if all(col in df_clean.columns for col in ['sales_order_no', 'sku', 'customer_emailid']):
            df_clean['primary_key'] = (
                df_clean['sales_order_no'].astype(str) + '-' + 
                df_clean['sku'].astype(str) + '-' + 
                df_clean['customer_emailid'].astype(str)
            )
            logger.info("Generated primary keys for data")
        else:
            logger.error("Missing required columns for primary key generation")
            return False
        
        # Ensure units_returned_flag is string
        if 'units_returned_flag' in df_clean.columns:
            df_clean['units_returned_flag'] = df_clean['units_returned_flag'].astype(str)
            logger.info("Converted units_returned_flag to string")
        
        # Create a temporary table
        temp_table = "temp_bronze_import"
        conn.execute(f"DROP TABLE IF EXISTS {temp_table}")
        
        # Register DataFrame with DuckDB
        conn.register("df_clean", df_clean)
        
        # Create temporary table with the right schema
        create_temp_table = f"""
        CREATE TABLE {temp_table} (
            primary_key VARCHAR,
            sales_order_no VARCHAR,
            customer_emailid VARCHAR,
            order_date TIMESTAMP,
            sku VARCHAR,
            sales_qty INTEGER,
            gross DOUBLE,
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
        )
        """
        conn.execute(create_temp_table)
        
        # Insert data with proper casting
        insert_query = f"""
        INSERT INTO {temp_table}
        SELECT 
            primary_key,
            sales_order_no,
            customer_emailid,
            TRY_CAST(order_date AS TIMESTAMP),
            sku,
            TRY_CAST(sales_qty AS INTEGER),
            TRY_CAST(gross AS DOUBLE),
            TRY_CAST(return_qty AS INTEGER),
            units_returned_flag,
            TRY_CAST(return_date AS TIMESTAMP),
            return_no,
            return_comment,
            orderlink,
            q_cls_id,
            q_sku_desc,
            q_gmm_id,
            q_sku_id,
            class_,
            division_,
            brand_,
            q_clr_dnum,
            q_clr_desc,
            vendor_style,
            size_
        FROM df_clean
        """
        conn.execute(insert_query)
        
        # Check how many rows were inserted
        temp_count = conn.execute(f"SELECT COUNT(*) FROM {temp_table}").fetchone()[0]
        logger.info(f"Inserted {temp_count} rows into temporary table")
        
        # Find new records (not already in bronze)
        conn.execute(f"""
        CREATE TEMPORARY TABLE new_records AS
        SELECT t.* 
        FROM {temp_table} t
        LEFT JOIN bronze_return_order_data b ON t.primary_key = b.primary_key
        WHERE b.primary_key IS NULL
        """)
        
        # Count new records
        new_count = conn.execute("SELECT COUNT(*) FROM new_records").fetchone()[0]
        logger.info(f"Found {new_count} new records to add to bronze layer")
        
        # Get current bronze count
        before_count = conn.execute("SELECT COUNT(*) FROM bronze_return_order_data").fetchone()[0]
        
        # Insert new records into bronze
        conn.execute("""
        INSERT INTO bronze_return_order_data
        SELECT * FROM new_records
        """)
        
        # Get new bronze count
        after_count = conn.execute("SELECT COUNT(*) FROM bronze_return_order_data").fetchone()[0]
        rows_added = after_count - before_count
        
        logger.info(f"Added {rows_added} new rows to bronze_return_order_data")
        
        # Clean up temporary tables
        conn.execute(f"DROP TABLE IF EXISTS {temp_table}")
        conn.execute("DROP TABLE IF EXISTS new_records")
        
        return True
    except Exception as e:
        logger.error(f"Error updating bronze layer: {str(e)}")
        return False

def update_product_diversity_score(conn):
    """Update the product_diversity_score calculation in the database"""
    try:
        logger.info("Updating category_diversity_score calculation")
        
        # Get unique category count
        unique_categories = conn.execute("SELECT COUNT(DISTINCT class_) FROM bronze_return_order_data").fetchone()[0]
        if unique_categories <= 0:
            unique_categories = 20  # Default if no categories found
        
        logger.info(f"Found {unique_categories} unique product categories in bronze layer")
        
        # Create SQL to update the score
        update_query = f"""
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
        
        -- Update silver_customer_features with new scores
        UPDATE silver_customer_features
        SET category_diversity_score = t.category_diversity_score
        FROM temp_category_diversity t
        WHERE silver_customer_features.customer_emailid = t.customer_emailid;
        
        -- Insert new customers not yet in silver layer (informational only)
        SELECT COUNT(*) as new_customers_count
        FROM temp_category_diversity t
        WHERE NOT EXISTS (
            SELECT 1 FROM silver_customer_features s
            WHERE s.customer_emailid = t.customer_emailid
        );
        
        -- Drop temporary table
        DROP TABLE temp_category_diversity;
        """
        
        conn.execute("BEGIN TRANSACTION")
        
        # Run the query
        new_customers = conn.execute(update_query).fetchone()[0]
        
        # Get number of customers with updated scores
        affected_rows = conn.execute("""
            SELECT COUNT(*) 
            FROM silver_customer_features 
            WHERE category_diversity_score > 0
        """).fetchone()[0]
        
        conn.execute("COMMIT")
        
        logger.info(f"Updated category_diversity_score for {affected_rows} customers")
        logger.info(f"Found {new_customers} new customers not yet in silver layer")
        
        return True
    except Exception as e:
        logger.error(f"Error updating category_diversity_score: {str(e)}")
        conn.execute("ROLLBACK")
        return False

def export_silver_layer(conn):
    """Export the silver layer to CSV"""
    try:
        output_file = f"silver_customer_features_updated_{TIMESTAMP}.csv"
        
        logger.info(f"Exporting silver layer to {output_file}")
        
        # Get the data
        df = conn.execute("SELECT * FROM silver_customer_features").fetchdf()
        logger.info(f"Retrieved {len(df)} rows with {len(df.columns)} columns")
        
        # Export to CSV
        df.to_csv(output_file, index=False)
        logger.info(f"Successfully exported to CSV: {output_file}")
        return output_file
    except Exception as e:
        logger.error(f"Error exporting silver layer: {str(e)}")
        return None

def main():
    """Main execution function"""
    try:
        logger.info("Starting update and export process")
        
        # Connect to the database
        conn = get_database_connection()
        
        # Step 1: Update bronze layer with new data
        logger.info("Step 1: Updating bronze layer with new sentiment data")
        if import_new_data(conn):
            logger.info("Successfully updated bronze layer")
            
            # Step 2: Update product diversity score
            logger.info("Step 2: Updating product diversity score")
            if update_product_diversity_score(conn):
                logger.info("Successfully updated product diversity score")
                
                # Step 3: Export updated silver layer
                logger.info("Step 3: Exporting updated silver layer")
                output_file = export_silver_layer(conn)
                
                if output_file:
                    logger.info(f"Successfully exported silver layer to {output_file}")
                    print(f"\nProcess completed successfully!\nOutput file: {output_file}")
                else:
                    logger.error("FAILED: Failed to export silver layer")
                    print("\nFAILED: Failed to export silver layer")
            else:
                logger.error("FAILED: Failed to update product diversity score")
                print("\nFAILED: Failed to update product diversity score")
        else:
            logger.error("FAILED: Failed to update bronze layer")
            print("\nFAILED: Failed to update bronze layer")
            
    except Exception as e:
        logger.error(f"Process failed: {str(e)}")
        print(f"\nProcess failed: {str(e)}")
    finally:
        if 'conn' in locals():
            conn.close()
            logger.info("Database connection closed")

if __name__ == "__main__":
    main()
