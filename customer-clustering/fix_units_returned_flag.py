"""
Check and update units_returned_flag column in bronze layer
"""

import duckdb
import logging
import sys

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

def main():
    """Check and fix units_returned_flag column type"""
    try:
        # Connect to database
        conn = duckdb.connect(DB_PATH)
        logger.info(f"Connected to database: {DB_PATH}")
        
        # Check column type
        schema_info = conn.execute("""
        SELECT column_name, data_type 
        FROM information_schema.columns 
        WHERE table_name = 'bronze_return_order_data' 
        AND column_name = 'units_returned_flag'
        """).fetchall()
        
        if not schema_info:
            logger.error("units_returned_flag column not found")
            return
        
        logger.info(f"Current units_returned_flag column type: {schema_info[0][1]}")
        
        if schema_info[0][1].upper() != 'VARCHAR':
            logger.info("Changing units_returned_flag column to VARCHAR...")
            
            # Create a backup
            conn.execute("CREATE TABLE bronze_return_order_data_backup AS SELECT * FROM bronze_return_order_data")
            logger.info("Created backup table")
            
            # Check that backup was created properly
            backup_count = conn.execute("SELECT COUNT(*) FROM bronze_return_order_data_backup").fetchone()[0]
            logger.info(f"Backup table has {backup_count} rows")
            
            # Drop and recreate bronze table with updated schema
            conn.execute("DROP TABLE bronze_return_order_data")
            logger.info("Dropped original table")
            
            # Create new table with VARCHAR for units_returned_flag
            create_table_sql = """
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
            """
            conn.execute(create_table_sql)
            logger.info("Created new table with VARCHAR for units_returned_flag")
            
            # Copy data with explicit casting
            copy_data_sql = """
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
                CAST(units_returned_flag AS VARCHAR),
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
            """
            conn.execute(copy_data_sql)
            logger.info("Copied data to new table with explicit casting")
            
            # Verify row count
            new_count = conn.execute("SELECT COUNT(*) FROM bronze_return_order_data").fetchone()[0]
            logger.info(f"New table has {new_count} rows")
            
            if backup_count == new_count:
                logger.info("Row counts match, dropping backup table")
                conn.execute("DROP TABLE bronze_return_order_data_backup")
            else:
                logger.error(f"Row count mismatch: backup={backup_count}, new={new_count}")
                logger.error("Keeping backup table for reference")
        else:
            logger.info("units_returned_flag is already VARCHAR, no changes needed")
        
        # Check final schema to confirm
        final_schema = conn.execute("""
        SELECT column_name, data_type 
        FROM information_schema.columns 
        WHERE table_name = 'bronze_return_order_data' 
        AND column_name = 'units_returned_flag'
        """).fetchall()
        
        logger.info(f"Final units_returned_flag column type: {final_schema[0][1]}")
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
    finally:
        if 'conn' in locals():
            conn.close()
            logger.info("Database connection closed")

if __name__ == "__main__":
    main()
