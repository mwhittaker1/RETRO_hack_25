"""
Export Gold Layer to CSV and Excel
This script updates the category_diversity_score_scaled in the gold layer
and exports the gold_cluster_processed table to both CSV and Excel formats.
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
        logging.FileHandler(f'gold_export_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)
# Constants
DB_PATH = "customer_clustering.db"
OUTPUT_DIR = Path(".")
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

def update_category_diversity_scaled(conn):
    """Update the category_diversity_score_scaled column in the gold layer"""
    try:
        logger.info("Updating category_diversity_score_scaled in gold layer")
        
        # Check if gold layer table exists
        table_exists = conn.execute("""
        SELECT COUNT(*) FROM information_schema.tables 
        WHERE table_name = 'gold_cluster_processed'
        """).fetchone()[0]
        
        if table_exists == 0:
            logger.error("gold_cluster_processed table does not exist")
            return False
        
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
        
        conn.execute("BEGIN TRANSACTION")
        
        # Run the update query
        conn.execute(update_query)
        
        # Get number of updated rows
        updated_rows = conn.execute("""
        SELECT COUNT(*) FROM gold_cluster_processed
        WHERE category_diversity_score_scaled IS NOT NULL
        """).fetchone()[0]
        
        conn.execute("COMMIT")
        
        logger.info(f"Updated category_diversity_score_scaled for {updated_rows} customers in gold layer")
        return True
    except Exception as e:
        logger.error(f"Error updating category_diversity_score_scaled: {str(e)}")
        conn.execute("ROLLBACK")
        return False

def export_gold_layer(conn):
    """Export the gold layer to CSV and Excel"""
    try:
        # Export to CSV
        csv_file = OUTPUT_DIR / f"gold_cluster_processed_{TIMESTAMP}.csv"
        logger.info(f"Exporting to CSV: {csv_file}")
        
        # Get the data
        logger.info("Retrieving gold layer data...")
        df = conn.execute("SELECT * FROM gold_cluster_processed").fetchdf()
        logger.info(f"Retrieved {len(df)} rows with {len(df.columns)} columns")
        
        # Save to CSV
        df.to_csv(csv_file, index=False)
        logger.info(f"Successfully exported to CSV: {csv_file}")
        
        # Try to export to Excel if not too large
        try:
            if len(df) <= 1000000:  # Excel has a limit of ~1M rows
                excel_file = OUTPUT_DIR / f"gold_cluster_processed_{TIMESTAMP}.xlsx"
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
        logger.error(f"Error exporting gold layer: {str(e)}")
        return None, None

def main():
    """Main execution function"""
    try:
        logger.info("Starting gold layer update and export process")
        
        # Connect to database
        conn = duckdb.connect(DB_PATH)
        logger.info(f"Connected to database: {DB_PATH}")
        
        # Update the scaled diversity score in gold layer
        if update_category_diversity_scaled(conn):
            logger.info("Successfully updated category_diversity_score_scaled in gold layer")
            
            # Export gold layer
            csv_file, excel_file = export_gold_layer(conn)
            
            if csv_file:
                logger.info("Export completed successfully")
                print("\nGold layer export completed successfully!")
                print(f"CSV output: {csv_file}")
                if excel_file:
                    print(f"Excel output: {excel_file}")
            else:
                logger.error("Failed to export gold layer")
                print("\nFailed to export gold layer")
        else:
            logger.warning("Failed to update category_diversity_score_scaled in gold layer")
            logger.info("Attempting to export gold layer anyway")
            
            # Export gold layer
            csv_file, excel_file = export_gold_layer(conn)
            
            if csv_file:
                logger.info("Export completed successfully")
                print("\nGold layer export completed successfully (without updates)!")
                print(f"CSV output: {csv_file}")
                if excel_file:
                    print(f"Excel output: {excel_file}")
            else:
                logger.error("Failed to export gold layer")
                print("\nFailed to export gold layer")
            
    except Exception as e:
        logger.error(f"Process failed: {str(e)}")
        print(f"\nProcess failed: {str(e)}")
    finally:
        if 'conn' in locals():
            conn.close()
            logger.info("Database connection closed")

if __name__ == "__main__":
    main()
