"""
Create a new bronze layer table from the CSV file
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
    """Import new data directly to bronze layer using DuckDB's COPY command"""
    try:
        # Connect to database
        conn = duckdb.connect(DB_PATH)
        logger.info(f"Connected to database: {DB_PATH}")
        
        # Read the CSV file into a pandas DataFrame
        df = pd.read_csv(NEW_DATA_PATH, low_memory=False)
        logger.info(f"Read {len(df)} rows from CSV file")
        
        # Print original columns
        logger.info(f"Original columns: {df.columns.tolist()}")
        
        # Rename columns to lowercase
        df.columns = [col.lower() for col in df.columns]
        
        # Generate primary key
        df['primary_key'] = df['sales_order_no'].astype(str) + '-' + df['sku'].astype(str) + '-' + df['customer_emailid'].astype(str)
        
        # Explicitly convert units_returned_flag to string
        df['units_returned_flag'] = df['units_returned_flag'].astype(str)
        
        # Create a new temporary table with the correct schema
        temp_table_name = "new_sentiment_data"
        
        # Drop if exists
        conn.execute(f"DROP TABLE IF EXISTS {temp_table_name}")
        
        # Create table with proper schema
        conn.execute(f"""
        CREATE TABLE {temp_table_name} (
            primary_key VARCHAR PRIMARY KEY,
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
        
        # Register the DataFrame with DuckDB
        conn.register("df_new_data", df)
        
        # Insert the data using explicit column mapping and type casting
        conn.execute(f"""
        INSERT INTO {temp_table_name} (
            primary_key, sales_order_no, customer_emailid, order_date, sku,
            sales_qty, gross, return_qty, units_returned_flag, return_date,
            return_no, return_comment, orderlink, q_cls_id, q_sku_desc,
            q_gmm_id, q_sku_id, class_, division_, brand_, q_clr_dnum,
            q_clr_desc, vendor_style, size_
        )
        SELECT
            primary_key,
            sales_order_no,
            customer_emailid,
            TRY_CAST(order_date AS TIMESTAMP),
            sku,
            TRY_CAST(sales_qty AS INTEGER),
            TRY_CAST(gross AS DOUBLE),
            TRY_CAST(return_qty AS INTEGER),
            units_returned_flag,  -- Already string from DataFrame
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
        FROM df_new_data
        """)
        
        # Get count of rows inserted
        inserted_count = conn.execute(f"SELECT COUNT(*) FROM {temp_table_name}").fetchone()[0]
        logger.info(f"Inserted {inserted_count} rows into {temp_table_name}")
        
        # Get count of rows in bronze layer
        bronze_count = conn.execute("SELECT COUNT(*) FROM bronze_return_order_data").fetchone()[0]
        logger.info(f"Current bronze layer has {bronze_count} rows")
        
        # Insert the new data into the bronze layer, avoiding duplicates
        conn.execute(f"""
        INSERT INTO bronze_return_order_data
        SELECT * FROM {temp_table_name}
        WHERE primary_key NOT IN (SELECT primary_key FROM bronze_return_order_data)
        """)
        
        # Get new count
        new_bronze_count = conn.execute("SELECT COUNT(*) FROM bronze_return_order_data").fetchone()[0]
        rows_added = new_bronze_count - bronze_count
        
        logger.info(f"Added {rows_added} new rows to bronze layer")
        
        # Clean up temporary table
        conn.execute(f"DROP TABLE {temp_table_name}")
        
        logger.info("Import completed successfully")
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
    finally:
        if 'conn' in locals():
            conn.close()
            logger.info("Database connection closed")

if __name__ == "__main__":
    main()
