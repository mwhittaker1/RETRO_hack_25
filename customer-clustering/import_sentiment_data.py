"""
Import New Sentiment Data to Bronze Layer and Update Silver/Gold Layers
This script imports sentiment data from CSV, updates the bronze layer,
and recalculates product diversity score.
"""

import duckdb
import pandas as pd
import os
import sys
import logging
from datetime import datetime
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'import_sentiment_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Constants
DB_PATH = "customer_clustering.db"
NEW_DATA_PATH = "sentiment_and_raw/new_base_returns_sku_reasoncodes_sent.csv"
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

def check_units_returned_flag_column_type(conn):
    """Check if units_returned_flag is VARCHAR, alter if needed"""
    try:
        # Get column type
        schema_info = conn.execute("""
        SELECT column_name, data_type 
        FROM information_schema.columns 
        WHERE table_name = 'bronze_return_order_data' 
        AND column_name = 'units_returned_flag'
        """).fetchall()
        
        if schema_info and schema_info[0][1].upper() != 'VARCHAR':
            logger.warning(f"units_returned_flag column is type {schema_info[0][1]}, needs to be VARCHAR")
            
            # Create a backup of the existing data
            conn.execute("CREATE TABLE bronze_return_order_data_backup AS SELECT * FROM bronze_return_order_data")
            logger.info("Created backup of bronze_return_order_data")
            
            # Drop the old table and recreate with correct schema
            conn.execute("DROP TABLE bronze_return_order_data")
            
            # Create the table with VARCHAR for units_returned_flag
            conn.execute(f"""
            CREATE TABLE bronze_return_order_data (
                primary_key VARCHAR PRIMARY KEY, 
                sales_order_no VARCHAR,
                customer_emailid VARCHAR,
                order_date TIMESTAMP,
                sku VARCHAR,
                sales_qty INTEGER,
                gross DOUBLE,
                return_qty INTEGER,
                units_returned_flag VARCHAR,  -- Changed to VARCHAR
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
            """)
            
            # Copy data from backup with explicit casting for units_returned_flag
            conn.execute(f"""
            INSERT INTO bronze_return_order_data
            SELECT 
                primary_key,
                sales_order_no,
                customer_emailid,
                order_date,
                sku,
                sales_qty,
                gross,
                return_qty,
                CAST(units_returned_flag AS VARCHAR) AS units_returned_flag,
                return_date,
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
            FROM bronze_return_order_data_backup
            """)
            
            # Check the row count matches
            original_count = conn.execute("SELECT COUNT(*) FROM bronze_return_order_data_backup").fetchone()[0]
            new_count = conn.execute("SELECT COUNT(*) FROM bronze_return_order_data").fetchone()[0]
            
            if original_count == new_count:
                logger.info(f"Successfully migrated {new_count} rows with updated schema")
                conn.execute("DROP TABLE bronze_return_order_data_backup")
            else:
                logger.error(f"Row count mismatch after migration: {original_count} vs {new_count}")
                # Restore from backup
                conn.execute("DROP TABLE bronze_return_order_data")
                conn.execute("ALTER TABLE bronze_return_order_data_backup RENAME TO bronze_return_order_data")
                return False
                
        return True
    except Exception as e:
        logger.error(f"Error checking units_returned_flag column type: {str(e)}")
        return False

def import_sentiment_data(conn):
    """Import the sentiment data to bronze layer"""
    try:
        logger.info(f"Importing data from {NEW_DATA_PATH}")
        
        # Check if file exists
        if not os.path.exists(NEW_DATA_PATH):
            logger.error(f"File not found: {NEW_DATA_PATH}")
            return False
        
        # Read the file
        try:
            df = pd.read_csv(NEW_DATA_PATH, low_memory=False)
            logger.info(f"Read {len(df)} rows from CSV file")
        except Exception as e:
            logger.error(f"Error reading CSV file: {str(e)}")
            return False
        
        # Check column names and standardize
        original_columns = df.columns.tolist()
        logger.info(f"Original columns: {original_columns}")
        
        # Create a mapping of original case-insensitive column names to bronze layer columns
        bronze_columns_map = {
            "sales_order_no": "sales_order_no",
            "customer_emailid": "customer_emailid", 
            "order_date": "order_date",
            "sku": "sku",
            "sales_qty": "sales_qty",
            "gross": "gross",
            "return_qty": "return_qty",
            "units_returned_flag": "units_returned_flag",
            "return_date": "return_date",
            "return_no": "return_no",
            "return_comment": "return_comment",
            "orderlink": "orderlink",
            "q_cls_id": "q_cls_id",
            "q_sku_desc": "q_sku_desc",
            "q_gmm_id": "q_gmm_id",
            "q_sku_id": "q_sku_id",
            "class_": "class_",
            "division_": "division_",
            "brand_": "brand_",
            "q_clr_dnum": "q_clr_dnum",
            "q_clr_desc": "q_clr_desc",
            "vendor_style": "vendor_style",
            "size_": "size_"
        }
        
        # Create a new DataFrame with the standardized column names
        mapped_df = pd.DataFrame()
        
        # Map original columns to bronze columns (case-insensitive)
        for bronze_col in bronze_columns_map.keys():
            found = False
            for orig_col in original_columns:
                if orig_col.lower() == bronze_col.lower():
                    logger.info(f"Mapped column {orig_col} to {bronze_col}")
                    mapped_df[bronze_col] = df[orig_col]
                    found = True
                    break
            
            if not found:
                logger.warning(f"Could not find match for column: {bronze_col}")
                mapped_df[bronze_col] = None
        
        # Generate primary key from other columns (order_no + sku + email)
        if ("sales_order_no" in mapped_df.columns and 
            "sku" in mapped_df.columns and 
            "customer_emailid" in mapped_df.columns):
            mapped_df['primary_key'] = mapped_df['sales_order_no'].astype(str) + '-' + mapped_df['sku'].astype(str) + '-' + mapped_df['customer_emailid'].astype(str)
            logger.info("Generated primary key column")
        else:
            logger.error("Missing required columns for primary key generation")
            return False
        
        # Handle data types
        # Ensure units_returned_flag is treated as VARCHAR
        if 'units_returned_flag' in mapped_df.columns:
            mapped_df['units_returned_flag'] = mapped_df['units_returned_flag'].astype(str)
            logger.info("Converted units_returned_flag to string")
        
        # Convert dates
        try:
            if 'order_date' in mapped_df.columns:
                mapped_df['order_date'] = pd.to_datetime(mapped_df['order_date'], errors='coerce')
            if 'return_date' in mapped_df.columns:
                mapped_df['return_date'] = pd.to_datetime(mapped_df['return_date'], errors='coerce')
        except Exception as e:
            logger.warning(f"Error converting dates: {str(e)}")
        
        # Convert numeric fields
        numeric_columns = ['sales_qty', 'gross', 'return_qty']
        for col in numeric_columns:
            if col in mapped_df.columns:
                mapped_df[col] = pd.to_numeric(mapped_df[col], errors='coerce')
        
        # Add primary key to the beginning of our columns list
        bronze_columns = ["primary_key"] + list(bronze_columns_map.keys())
        
        # Select only the columns we need in the right order
        try:
            mapped_df = mapped_df[bronze_columns]
            logger.info(f"Prepared DataFrame with {len(mapped_df)} rows and {len(mapped_df.columns)} columns")
        except Exception as e:
            logger.error(f"Error selecting columns: {str(e)}")
            logger.error(f"Available columns: {mapped_df.columns.tolist()}")
            logger.error(f"Required columns: {bronze_columns}")
            return False
        
        # Create a temporary table for the new data with explicit types
        temp_table_create = """
        CREATE TEMPORARY TABLE temp_new_data (
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
        conn.execute(temp_table_create)
        
        # Register DataFrame with DuckDB for the insert
        conn.register("df_new_data", mapped_df)
        
        # Insert data into temporary table with proper casting, ensuring units_returned_flag is VARCHAR
        try:
            conn.execute("""
            INSERT INTO temp_new_data
            SELECT 
                primary_key,
                sales_order_no,
                customer_emailid,
                CAST(order_date AS TIMESTAMP),
                sku,
                CAST(sales_qty AS INTEGER),
                CAST(gross AS DOUBLE),
                CAST(return_qty AS INTEGER),
                CAST(units_returned_flag AS VARCHAR),
                CAST(return_date AS TIMESTAMP),
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
            FROM df_new_data
            """)
            
            logger.info(f"Inserted {conn.execute('SELECT COUNT(*) FROM temp_new_data').fetchone()[0]} rows into temporary table")
        except Exception as e:
            logger.error(f"Error inserting into temporary table: {str(e)}")
            return False
            
        # Get current count
        before_count = conn.execute("SELECT COUNT(*) FROM bronze_return_order_data").fetchone()[0]
        logger.info(f"Bronze layer has {before_count} rows before import")
        
        # Insert data from temporary table into bronze_return_order_data, ignoring duplicates
        try:
            conn.execute("""
            INSERT INTO bronze_return_order_data
            SELECT * FROM temp_new_data
            WHERE primary_key NOT IN (SELECT primary_key FROM bronze_return_order_data)
            """)
            
            # Get new count
            after_count = conn.execute("SELECT COUNT(*) FROM bronze_return_order_data").fetchone()[0]
            rows_added = after_count - before_count
            
            logger.info(f"Added {rows_added} new rows to bronze layer")
            
            # Clean up
            conn.execute("DROP TABLE temp_new_data")
            
            # Check if any rows were added
            if rows_added == 0:
                logger.warning("No new rows were added - possible duplicates")
            
            return True
        except Exception as e:
            logger.error(f"Error inserting data: {str(e)}")
            return False
            
    except Exception as e:
        logger.error(f"Error importing sentiment data: {str(e)}")
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
        logger.info("Starting sentiment data import and diversity score update")
        
        # Connect to the database
        conn = get_database_connection()
        
        # Ensure units_returned_flag is VARCHAR
        if not check_units_returned_flag_column_type(conn):
            logger.error("Failed to ensure units_returned_flag is VARCHAR")
            return
        
        # Import sentiment data
        if import_sentiment_data(conn):
            logger.info("Successfully imported sentiment data")
            
            # Update product diversity score
            if update_product_diversity_score(conn):
                logger.info("Successfully updated product diversity score")
                
                # Export silver layer
                output_file = export_silver_layer(conn)
                if output_file:
                    logger.info(f"Process completed successfully. Output file: {output_file}")
                    print(f"\nProcess completed successfully!\nOutput file: {output_file}")
                else:
                    logger.error("Failed to export silver layer")
            else:
                logger.error("Failed to update product diversity score")
        else:
            logger.error("Failed to import sentiment data")
            
    except Exception as e:
        logger.error(f"Process failed: {str(e)}")
        print(f"\nProcess failed: {str(e)}")
    finally:
        if 'conn' in locals():
            conn.close()
            logger.info("Database connection closed")

if __name__ == "__main__":
    main()
