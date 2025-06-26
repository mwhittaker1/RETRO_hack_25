"""
Update Product Diversity Score
This script updates the category_diversity_score calculation in the silver layer
using the actual count of unique categories in the bronze layer.
"""

import duckdb
import pandas as pd
import logging
from datetime import datetime
import sys

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'update_diversity_score_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Constants
DB_PATH = "customer_clustering.db"

def get_database_connection():
    """Connect to the database"""
    try:
        conn = duckdb.connect(DB_PATH)
        logger.info(f"Connected to database: {DB_PATH}")
        return conn
    except Exception as e:
        logger.error(f"Error connecting to database: {str(e)}")
        raise

def get_category_counts(conn):
    """Get total number of unique categories in the database"""
    try:
        query = """
        SELECT COUNT(DISTINCT class_) as unique_categories 
        FROM bronze_return_order_data
        """
        result = conn.execute(query).fetchone()
        unique_categories = result[0] if result[0] > 0 else 20  # Default to 20 if no categories found
        logger.info(f"Found {unique_categories} unique product categories in bronze layer")
        return unique_categories
    except Exception as e:
        logger.error(f"Error getting category counts: {str(e)}")
        return 20  # Default to 20 if query fails

def update_category_diversity_calculation(conn, unique_categories):
    """Update the category_diversity_score calculation in the database"""
    try:
        logger.info("Updating category_diversity_score calculation...")
        
        # Update the SQL query for calculating category_diversity_score
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
        
        -- Drop temporary table
        DROP TABLE temp_category_diversity;
        """
        
        conn.execute("BEGIN TRANSACTION")
        conn.execute(update_query)
        
        # Verify update
        affected_rows = conn.execute("""
            SELECT COUNT(*) 
            FROM silver_customer_features 
            WHERE category_diversity_score > 0
        """).fetchone()[0]
        
        conn.execute("COMMIT")
        logger.info(f"Updated category_diversity_score for {affected_rows} customers")
        return True
        
    except Exception as e:
        logger.error(f"Error updating category_diversity_score: {str(e)}")
        conn.execute("ROLLBACK")
        return False

def export_silver_layer(conn):
    """Export the silver layer to CSV"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"silver_customer_features_updated_{timestamp}.csv"
        
        logger.info(f"Exporting silver layer to {output_file}...")
        
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
        logger.info("Starting diversity score update process")
        
        # Connect to the database
        conn = get_database_connection()
        
        # Get unique category count
        unique_categories = get_category_counts(conn)
        
        # Update category_diversity_score calculation
        if update_category_diversity_calculation(conn, unique_categories):
            # Export updated silver layer
            output_file = export_silver_layer(conn)
            if output_file:
                logger.info(f"Process completed successfully. Output file: {output_file}")
                print(f"\nProcess completed successfully!\nOutput file: {output_file}")
            else:
                logger.error("Failed to export silver layer")
        else:
            logger.error("Failed to update category_diversity_score")
            
    except Exception as e:
        logger.error(f"Process failed: {str(e)}")
        print(f"\nProcess failed: {str(e)}")
    finally:
        if 'conn' in locals():
            conn.close()
            logger.info("Database connection closed")

if __name__ == "__main__":
    main()
