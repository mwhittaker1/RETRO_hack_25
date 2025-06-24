# Required imports for Customer Clustering Features Pipeline
import pandas as pd
import duckdb
import numpy as np
from pathlib import Path
import logging
from typing import Union, Optional
import gc
from datetime import datetime
import time
import warnings
from customer_clustering_features import create_customer_clustering_features

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure pandas display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 50)

print("✅ All imports loaded successfully!")
print("🔧 Environment configured for large dataset processing")
print("📊 Ready to run customer clustering features pipeline")

# --- Logging setup for ccf logger ---
import logging
ccf_logger = logging.getLogger("customer_clustering_features")
if not ccf_logger.hasHandlers():
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s %(name)s: %(message)s', datefmt='%H:%M:%S')
    handler.setFormatter(formatter)
    ccf_logger.addHandler(handler)
    ccf_logger.setLevel(logging.INFO)

def run_feature_pipeline():
    start_time = time.time()
    try:
        # Configuration
        config = {
            'file_path': 'data/base_returns_sku_metadata.csv',  # Update with your file path
            'table_name': 'customer_transactions',
            'features_table_name': 'customer_clustering_features',
            'chunk_size': 50000,  # Adjust based on available RAM
            'db_file': 'customer_features.db',  # Persistent database file
            'force_recreate': False  # Set to True to rebuild from scratch
        }
        print("🚀 Starting Customer Clustering Feature Pipeline")
        print(f"📁 File: {config['file_path']}")
        print(f"💾 Database: {config['db_file']}")
        print(f"📊 Features table: {config['features_table_name']}")
        print("-" * 60)
        # Run the complete pipeline
        conn = create_customer_clustering_features(**config)
        # Display results summary
        print("\n" + "="*60)
        print("✅ PIPELINE COMPLETED SUCCESSFULLY")
        print("="*60)
        # Show sample features
        print("\n📋 Sample Customer Features:")
        sample_features = conn.execute(f"""
            SELECT * FROM {config['features_table_name']} 
            ORDER BY SALES_ORDER_NO_nunique DESC 
            LIMIT 5
        """).df()
        print(sample_features.to_string(index=False))
        # Performance summary
        elapsed_time = time.time() - start_time
        total_rows = conn.execute(f"SELECT COUNT(*) FROM {config['table_name']}").fetchone()[0]
        features_count = conn.execute(f"SELECT COUNT(*) FROM {config['features_table_name']}").fetchone()[0]
        print(f"\n⏱️  Processing completed in {elapsed_time:.2f} seconds")
        print(f"📊 Processed {total_rows:,} transaction records")
        print(f"👥 Generated features for {features_count:,} customers")
        print(f"🏃‍♂️ Processing speed: {total_rows/elapsed_time:,.0f} records/second")
        # Export options
        export_csv = input("\n📤 Export features to CSV? (y/n): ").lower().strip() == 'y'
        if export_csv:
            features_df = conn.execute(f"SELECT * FROM {config['features_table_name']}").df()
            csv_filename = 'customer_clustering_features.csv'
            features_df.to_csv(csv_filename, index=False)
            print(f"✅ Features exported to {csv_filename}")
        print(f"\n🎯 Ready for DBSCAN clustering!")
        print(f"💾 Database connection available as 'conn' variable")
        return conn
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        print("Check the logs above for detailed error information")
        raise
    finally:
        elapsed_time = time.time() - start_time
        print(f"\n⏱️  Total execution time: {elapsed_time:.2f} seconds")

def validate_features():
    # Only run feature validation, do not rerun the pipeline
    import duckdb
    db_file = 'customer_features.db'
    features_table_name = 'customer_clustering_features'
    conn = duckdb.connect(db_file)
    from customer_clustering_features import _feature_validation
    _feature_validation(conn, features_table_name)
    print("\nFeature validation complete. See logs for details.")

if __name__ == "__main__":
    validate_features()