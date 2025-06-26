"""
Repair database schema and insert new data
"""

import duckdb
import pandas as pd
import logging
import sys
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

DB_PATH = "customer_clustering.db"
NEW_DATA_PATH = "sentiment_and_raw/new_base_returns_sku_reasoncodes_sent.csv"

def main():
    """Repair database schema and import data"""
    try:
        # Connect to database
        conn = duckdb.connect(DB_PATH)
        logger.info(f"Connected to database: {DB_PATH}")
        
        # Check current tables
        tables = conn.execute("SHOW TABLES").fetchdf()
        logger.info(f"Current tables: {tables['name'].tolist()}")
        
        # Back up the bronze table
        logger.info("Creating backup of bronze_return_order_data")
        conn.execute("CREATE TABLE IF NOT EXISTS bronze_return_order_data_backup AS SELECT * FROM bronze_return_order_data")
        
        # Get the row count to verify backup worked
        backup_count = conn.execute("SELECT COUNT(*) FROM bronze_return_order_data_backup").fetchone()[0]
        logger.info(f"Backup has {backup_count} rows")
        
        # Recreate the bronze table with correct schema
        logger.info("Recreating bronze_return_order_data with correct schema")
        conn.execute("DROP TABLE bronze_return_order_data")
        
        # Create with correct schema
        conn.execute("""
        CREATE TABLE bronze_return_order_data (
            primary_key VARCHAR PRIMARY KEY, 
            sales_order_no VARCHAR,
            customer_emailid VARCHAR,
            order_date TIMESTAMP,
            sku VARCHAR,
            sales_qty INTEGER,
            gross DOUBLE,
            return_qty INTEGER,
            units_returned_flag VARCHAR,  -- VARCHAR, not TIMESTAMP
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
        
        # Restore data from backup with explicit casting
        logger.info("Restoring data from backup with explicit type casting")
        conn.execute("""
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
            CAST(units_returned_flag AS VARCHAR),  -- Explicit cast to VARCHAR
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
        
        # Verify data was restored
        restored_count = conn.execute("SELECT COUNT(*) FROM bronze_return_order_data").fetchone()[0]
        logger.info(f"Restored {restored_count} rows to bronze_return_order_data")
        
        if restored_count != backup_count:
            logger.error(f"Row count mismatch! Backup: {backup_count}, Restored: {restored_count}")
            return
        
        # Verify column type is now VARCHAR
        col_type = conn.execute("""
        SELECT data_type 
        FROM information_schema.columns 
        WHERE table_name = 'bronze_return_order_data' 
        AND column_name = 'units_returned_flag'
        """).fetchone()[0]
        
        logger.info(f"units_returned_flag column type is now: {col_type}")
        
        # Now try importing the new data
        logger.info(f"Reading new data from {NEW_DATA_PATH}")
        
        # Read CSV
        df = pd.read_csv(NEW_DATA_PATH, low_memory=False)
        logger.info(f"Read {len(df)} rows from CSV")
        
        # Prepare the data for import
        # Map columns by case-insensitive name
        df_columns = df.columns.tolist()
        logger.info(f"CSV has {len(df_columns)} columns: {df_columns}")
        
        # Create a new DataFrame with standardized column names
        import_df = pd.DataFrame()
        
        # Map columns
        bronze_columns = [
            'sales_order_no', 'customer_emailid', 'order_date', 'sku',
            'sales_qty', 'gross', 'return_qty', 'units_returned_flag',
            'return_date', 'return_no', 'return_comment', 'orderlink',
            'q_cls_id', 'q_sku_desc', 'q_gmm_id', 'q_sku_id', 'class_',
            'division_', 'brand_', 'q_clr_dnum', 'q_clr_desc', 'vendor_style',
            'size_'
        ]
        
        for bronze_col in bronze_columns:
            for orig_col in df_columns:
                if orig_col.lower() == bronze_col.lower():
                    logger.info(f"Mapped {orig_col} to {bronze_col}")
                    import_df[bronze_col] = df[orig_col]
                    break
            if bronze_col not in import_df.columns:
                logger.warning(f"No match found for {bronze_col}")
                import_df[bronze_col] = None
        
        # Generate primary key
        import_df['primary_key'] = (
            import_df['sales_order_no'].astype(str) + '-' +
            import_df['sku'].astype(str) + '-' +
            import_df['customer_emailid'].astype(str)
        )
        
        # Ensure units_returned_flag is string
        import_df['units_returned_flag'] = import_df['units_returned_flag'].astype(str)
        
        # Order columns with primary_key first
        import_cols = ['primary_key'] + bronze_columns
        import_df = import_df[import_cols]
        
        # Create a temporary table for the import
        logger.info("Creating temporary table for import")
        conn.execute("DROP TABLE IF EXISTS temp_import")
        
        # Register the DataFrame
        conn.register("import_df", import_df)
        
        # Create the temporary table
        conn.execute("""
        CREATE TABLE temp_import (
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
        """)
        
        # Insert with explicit casting
        conn.execute("""
        INSERT INTO temp_import
        SELECT 
            primary_key,
            sales_order_no,
            customer_emailid,
            TRY_CAST(order_date AS TIMESTAMP),
            sku,
            TRY_CAST(sales_qty AS INTEGER),
            TRY_CAST(gross AS DOUBLE),
            TRY_CAST(return_qty AS INTEGER),
            units_returned_flag,  -- Already string
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
        FROM import_df
        """)
        
        # Check how many rows were inserted
        temp_count = conn.execute("SELECT COUNT(*) FROM temp_import").fetchone()[0]
        logger.info(f"Inserted {temp_count} rows into temp_import")
        
        # Find records that don't already exist in bronze
        conn.execute("""
        CREATE TEMPORARY TABLE new_records AS
        SELECT t.* 
        FROM temp_import t
        LEFT JOIN bronze_return_order_data b ON t.primary_key = b.primary_key
        WHERE b.primary_key IS NULL
        """)
        
        # Count new records
        new_count = conn.execute("SELECT COUNT(*) FROM new_records").fetchone()[0]
        logger.info(f"Found {new_count} new records")
        
        # Get bronze count before insert
        before_count = conn.execute("SELECT COUNT(*) FROM bronze_return_order_data").fetchone()[0]
        
        # Insert new records
        conn.execute("""
        INSERT INTO bronze_return_order_data
        SELECT * FROM new_records
        """)
        
        # Get bronze count after insert
        after_count = conn.execute("SELECT COUNT(*) FROM bronze_return_order_data").fetchone()[0]
        
        logger.info(f"Added {after_count - before_count} new records to bronze_return_order_data")
        
        # Clean up
        logger.info("Cleaning up temporary tables")
        conn.execute("DROP TABLE IF EXISTS temp_import")
        conn.execute("DROP TABLE IF EXISTS new_records")
        
        # Update diversity score
        logger.info("Updating category diversity score")
        
        # Get unique category count
        unique_categories = conn.execute("SELECT COUNT(DISTINCT class_) FROM bronze_return_order_data").fetchone()[0]
        logger.info(f"Found {unique_categories} unique product categories")
        
        # Update the score
        conn.execute(f"""
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
        FROM customer_totals
        """)
        
        # Update silver layer
        conn.execute("""
        UPDATE silver_customer_features
        SET category_diversity_score = t.category_diversity_score
        FROM temp_category_diversity t
        WHERE silver_customer_features.customer_emailid = t.customer_emailid
        """)
        
        # Count updated records
        updated_count = conn.execute("""
        SELECT COUNT(*) 
        FROM silver_customer_features 
        WHERE category_diversity_score > 0
        """).fetchone()[0]
        
        logger.info(f"Updated category_diversity_score for {updated_count} customers")
        
        # Export silver layer
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"silver_customer_features_updated_{timestamp}.csv"
        
        logger.info(f"Exporting silver layer to {output_file}")
        
        # Get the data
        silver_df = conn.execute("SELECT * FROM silver_customer_features").fetchdf()
        silver_df.to_csv(output_file, index=False)
        
        logger.info(f"Exported {len(silver_df)} rows to {output_file}")
        
        print(f"\nProcess completed successfully!\nOutput file: {output_file}")
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
    finally:
        if 'conn' in locals():
            conn.close()
            logger.info("Database connection closed")

if __name__ == "__main__":
    main()
