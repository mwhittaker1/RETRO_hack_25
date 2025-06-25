"""
Update the bronze_return_order_data table to include the GROSS column
"""

import duckdb
import pandas as pd
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def update_bronze_table_schema(db_path: str = "customer_clustering.db"):
    """Add GROSS column to bronze_return_order_data table"""
    try:
        # Connect to database
        conn = duckdb.connect(db_path)
        logger.info(f"Connected to database at {db_path}")
        
        # Check if GROSS column already exists
        has_gross = conn.execute("""
            SELECT COUNT(*) 
            FROM information_schema.columns 
            WHERE table_name='bronze_return_order_data' AND column_name='gross'
        """).fetchone()[0]
        
        if has_gross > 0:
            logger.info("GROSS column already exists in bronze_return_order_data table")
            return True
        
        # Add GROSS column to the table
        conn.execute("""
            ALTER TABLE bronze_return_order_data
            ADD COLUMN gross DOUBLE DEFAULT 0.0
        """)
        logger.info("Added GROSS column to bronze_return_order_data table")
          # Check if the returns data CSV exists with GROSS values
        csv_path = Path("base_returns_sku_reasoncodes_sent.csv")
        if not csv_path.exists():
            logger.warning(f"CSV file not found: {csv_path}")
            return False
        
        # Read CSV to get GROSS values
        df = pd.read_csv(csv_path)
        
        # Check if GROSS column exists in the CSV
        if 'GROSS' not in df.columns:
            logger.warning("GROSS column not found in CSV file")
            return False
        
        # For each row, update the GROSS value in the database
        logger.info("Updating GROSS values from CSV file...")
        
        # Convert to lowercase for consistency
        df.columns = [col.lower() for col in df.columns]
        
        # Create temporary table with the CSV data
        conn.execute("CREATE TEMPORARY TABLE temp_data AS SELECT * FROM df")
        
        # Update GROSS values in bronze_return_order_data
        conn.execute("""
            UPDATE bronze_return_order_data SET gross = t.gross
            FROM temp_data t
            WHERE bronze_return_order_data.sales_order_no = t.sales_order_no
            AND bronze_return_order_data.customer_emailid = t.customer_emailid
            AND bronze_return_order_data.sku = t.sku
        """)
        
        # Count number of rows updated
        updated_count = conn.execute("""
            SELECT COUNT(*) FROM bronze_return_order_data WHERE gross > 0
        """).fetchone()[0]
        
        logger.info(f"Updated GROSS values for {updated_count} rows")
        
        # Clean up
        conn.execute("DROP TABLE temp_data")
        conn.close()
        
        return True
        
    except Exception as e:
        logger.error(f"Error updating bronze table schema: {str(e)}")
        return False

if __name__ == "__main__":
    success = update_bronze_table_schema()
    if success:
        logger.info("Successfully updated bronze_return_order_data schema with GROSS column")
    else:
        logger.error("Failed to update bronze_return_order_data schema")
