"""
Export Gold Layer to XLSX or CSV
This script exports the gold_cluster_processed table to either XLSX or CSV
based on the size of the data.
"""

import duckdb
import pandas as pd
import os
from pathlib import Path
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
MAX_EXCEL_ROWS = 1000000  # Excel has a limit of ~1M rows
DB_PATH = "customer_clustering.db"
OUTPUT_DIR = Path(".")
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

def check_and_export_gold_layer():
    """Check gold layer size and export to appropriate format"""
    try:
        # Connect to database
        conn = duckdb.connect(DB_PATH)
        logger.info(f"Connected to database: {DB_PATH}")
        
        # Get row count
        row_count = conn.execute("SELECT COUNT(*) FROM gold_cluster_processed").fetchone()[0]
        logger.info(f"Gold layer contains {row_count} rows")
        
        # Get the data
        logger.info("Retrieving gold layer data...")
        df = conn.execute("SELECT * FROM gold_cluster_processed").fetchdf()
        logger.info(f"Retrieved {len(df)} rows with {len(df.columns)} columns")
        
        # Determine export format based on size
        if row_count <= MAX_EXCEL_ROWS:
            # Export to Excel
            export_path = OUTPUT_DIR / f"gold_cluster_processed_{TIMESTAMP}.xlsx"
            logger.info(f"Exporting to Excel: {export_path}")
            df.to_excel(export_path, index=False, engine='openpyxl')
            logger.info(f"Successfully exported to Excel: {export_path}")
            return str(export_path)
        else:
            # Export to CSV
            export_path = OUTPUT_DIR / f"gold_cluster_processed_{TIMESTAMP}.csv"
            logger.info(f"Data too large for Excel, exporting to CSV: {export_path}")
            df.to_csv(export_path, index=False)
            logger.info(f"Successfully exported to CSV: {export_path}")
            return str(export_path)
            
    except Exception as e:
        logger.error(f"Error exporting gold layer: {str(e)}")
        raise
    finally:
        conn.close()
        logger.info("Database connection closed")

def main():
    """Main execution function"""
    try:
        logger.info("Starting gold layer export process")
        output_file = check_and_export_gold_layer()
        logger.info(f"Export completed successfully. Output file: {output_file}")
        print(f"\nExport completed successfully!\nOutput file: {output_file}")
        return output_file
    except Exception as e:
        logger.error(f"Export process failed: {str(e)}")
        print(f"\nExport process failed: {str(e)}")
        return None

if __name__ == "__main__":
    main()
