"""
Update the bronze_return_order_data table to include the GROSS column with values
"""

import duckdb
import pandas as pd
import logging
from pathlib import Path
import time
import sys
import traceback

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def update_bronze_table_schema(db_path: str = "customer_clustering.db", force_update: bool = False):
    """Add GROSS column to bronze_return_order_data table and populate it with values from CSV"""
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
        
        if has_gross > 0 and not force_update:
            logger.info("GROSS column already exists in bronze_return_order_data table")
            
            # Check if GROSS values are all zero
            all_zeros = conn.execute("""
                SELECT COUNT(*) = 0
                FROM bronze_return_order_data
                WHERE gross > 0
            """).fetchone()[0]
            
            if all_zeros:
                logger.warning("GROSS column exists but all values are zero. Will update values.")
            else:
                logger.info("GROSS values are already populated. Set force_update=True to override.")
                return True
        else:
            # Add GROSS column to the table if it doesn't exist
            if has_gross == 0:
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
        logger.info(f"Reading GROSS values from {csv_path}...")
        
        # Read in chunks to handle large files
        chunk_size = 100000
        total_updated = 0
        start_time = time.time()
        
        # First, check if CSV has the GROSS column
        sample = pd.read_csv(csv_path, nrows=5)
        if 'GROSS' not in sample.columns:
            logger.warning("GROSS column not found in CSV file")
            return False
        
        for chunk_idx, chunk in enumerate(pd.read_csv(csv_path, chunksize=chunk_size)):
            # Convert column names to lowercase
            chunk.columns = [col.lower() for col in chunk.columns]
            
            # Create a temporary table with the chunk data
            conn.execute("CREATE TEMPORARY TABLE IF NOT EXISTS temp_gross_data AS SELECT * FROM chunk LIMIT 0")
            conn.execute("DELETE FROM temp_gross_data")
            conn.execute("INSERT INTO temp_gross_data SELECT * FROM chunk")
            
            # Update GROSS values in bronze_return_order_data using a more precise matching
            conn.execute("""
                UPDATE bronze_return_order_data 
                SET gross = t.gross
                FROM temp_gross_data t
                WHERE bronze_return_order_data.sales_order_no = t.sales_order_no
                AND bronze_return_order_data.customer_emailid = t.customer_emailid
                AND bronze_return_order_data.sku = t.sku
            """)
            
            # Check how many rows were updated in this chunk
            updated_in_chunk = conn.execute("SELECT COUNT(*) FROM bronze_return_order_data WHERE gross > 0").fetchone()[0] - total_updated
            total_updated += updated_in_chunk
            
            logger.info(f"Chunk {chunk_idx+1}: Updated {updated_in_chunk} rows. Total updated so far: {total_updated}")
        
        # Get statistics on updated GROSS values
        stats = conn.execute("""
            SELECT 
                COUNT(*) as total_rows,
                COUNT(*) FILTER (WHERE gross > 0) as rows_with_gross,
                MIN(gross) as min_gross,
                MAX(gross) as max_gross,
                AVG(gross) as avg_gross
            FROM bronze_return_order_data
        """).fetchdf()
        
        elapsed_time = time.time() - start_time
        logger.info(f"GROSS column update completed in {elapsed_time:.2f} seconds")
        logger.info(f"GROSS value statistics: {stats.to_dict('records')[0]}")
        
        if stats['rows_with_gross'][0] == 0:
            logger.error("Failed to update any GROSS values. Data may not match between CSV and database.")
            return False
            
        # Clean up
        conn.execute("DROP TABLE IF EXISTS temp_gross_data")
        conn.close()
        
        return True
        
    except Exception as e:
        logger.error(f"Error updating bronze table schema: {str(e)}")
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Update bronze table schema to include GROSS column')
    parser.add_argument('--force', action='store_true', help='Force update of GROSS values even if column exists')
    args = parser.parse_args()
    
    success = update_bronze_table_schema(force_update=args.force)
    if success:
        logger.info("Successfully updated bronze_return_order_data schema with GROSS column")
    else:
        logger.error("Failed to update bronze_return_order_data schema")
