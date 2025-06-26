"""
Update product diversity score and export silver layer
This script updates the product diversity score in the silver layer
and exports the silver_customer_features table to both CSV and Excel formats.
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
        logging.FileHandler(f'silver_export_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)
import logging
from datetime import datetime

# Constants
DB_PATH = "customer_clustering.db"
OUTPUT_DIR = Path(".")
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

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
        conn.execute(update_query)
        new_customers = conn.execute("""
        SELECT COUNT(*) as new_customers_count
        FROM (
            SELECT DISTINCT customer_emailid 
            FROM bronze_return_order_data 
            WHERE customer_emailid NOT IN (SELECT customer_emailid FROM silver_customer_features)
        ) as new_customers
        """).fetchone()[0]
        
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
    """Export the silver layer to CSV and Excel"""
    try:
        # Export to CSV
        csv_file = OUTPUT_DIR / f"silver_customer_features_{TIMESTAMP}.csv"
        logger.info(f"Exporting to CSV: {csv_file}")
        
        # Get the data
        logger.info("Retrieving silver layer data...")
        df = conn.execute("SELECT * FROM silver_customer_features").fetchdf()
        logger.info(f"Retrieved {len(df)} rows with {len(df.columns)} columns")
        
        # Save to CSV
        df.to_csv(csv_file, index=False)
        logger.info(f"Successfully exported to CSV: {csv_file}")
        
        # Try to export to Excel if not too large
        try:
            if len(df) <= 1000000:  # Excel has a limit of ~1M rows
                excel_file = OUTPUT_DIR / f"silver_customer_features_{TIMESTAMP}.xlsx"
                logger.info(f"Exporting to Excel: {excel_file}")
                df.to_excel(excel_file, index=False, engine='openpyxl')
                logger.info(f"Successfully exported to Excel: {excel_file}")
                return str(csv_file), str(excel_file)
            else:
                logger.warning("Data too large for Excel export, CSV only")
                return str(csv_file), None
        except Exception as excel_err:
            logger.warning(f"Excel export failed, CSV only: {str(excel_err)}")
            return str(csv_file), None
    except Exception as e:
        logger.error(f"Error exporting silver layer: {str(e)}")
        return None, None

def main():
    """Main execution function"""
    try:
        logger.info("Starting product diversity score update and silver layer export")
        
        # Connect to database
        conn = duckdb.connect(DB_PATH)
        logger.info(f"Connected to database: {DB_PATH}")
        
        # Update product diversity score
        if update_product_diversity_score(conn):
            logger.info("Successfully updated product diversity score")
            
            # Export silver layer
            csv_file, excel_file = export_silver_layer(conn)
            
            if csv_file:
                logger.info("Export completed successfully")
                print("\nProcess completed successfully!")
                print(f"CSV output: {csv_file}")
                if excel_file:
                    print(f"Excel output: {excel_file}")
            else:
                logger.error("Failed to export silver layer")
                print("\nFailed to export silver layer")
        else:
            logger.error("Failed to update product diversity score")
            print("\nFailed to update product diversity score")
            
    except Exception as e:
        logger.error(f"Process failed: {str(e)}")
        print(f"\nProcess failed: {str(e)}")
    finally:
        if 'conn' in locals():
            conn.close()
            logger.info("Database connection closed")

if __name__ == "__main__":
    main()
