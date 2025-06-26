"""
Update Bronze Layer with Sentiment Data and Calculate New Product Diversity Score
This script loads new_base_returns_sku_reasoncodes_sent.csv into the bronze layer
and updates the product_diversity_score calculation using actual category counts.
"""

import duckdb
import pandas as pd
import os
from pathlib import Path
import logging
from datetime import datetime
import sys

# Setup logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'update_database_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Constants
DB_PATH = "customer_clustering.db"
NEW_DATA_PATH = "sentiment_and_raw/new_base_returns_sku_reasoncodes_sent.csv"
MAX_CHUNK_SIZE = 50000  # Smaller chunks to handle large files

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

def update_bronze_layer(conn, file_path):
    """Update bronze layer with new sentiment data"""
    try:
        logger.info(f"Importing new data from: {file_path}")
        
        # Check if file exists
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return False
            
        # Read file in chunks to avoid memory issues
        for chunk_num, chunk in enumerate(pd.read_csv(file_path, chunksize=MAX_CHUNK_SIZE)):
            # Rename columns to match bronze_return_order_data schema if needed
            # Assuming the CSV columns are in the correct order as stated, but names may differ
            bronze_columns = [
                "primary_key", "sales_order_no", "customer_emailid", "order_date", 
                "sku", "sales_qty", "gross", "return_qty", "units_returned_flag", 
                "return_date", "return_no", "return_comment", "orderlink", 
                "q_cls_id", "q_sku_desc", "q_gmm_id", "q_sku_id", "class_", 
                "division_", "brand_", "q_clr_dnum", "q_clr_desc", "vendor_style", "size_"
            ]
            
            # Get actual column names from the chunk
            actual_columns = chunk.columns.tolist()
            
            # Create a mapping from actual columns to bronze columns
            if len(actual_columns) != len(bronze_columns):
                logger.warning(f"Column count mismatch. Expected {len(bronze_columns)}, got {len(actual_columns)}")
                # If columns don't match, try to map by position
                column_mapping = {actual_columns[i]: bronze_columns[i] for i in range(min(len(actual_columns), len(bronze_columns)))}
            else:
                column_mapping = {actual_columns[i]: bronze_columns[i] for i in range(len(actual_columns))}
            
            # Get the required columns
            required_columns = bronze_columns.copy()
            
            # Check if all required columns exist in actual columns
            for col in required_columns:
                if col not in chunk.columns:
                    # Try to find matching column by similar name
                    potential_matches = [actual_col for actual_col in actual_columns 
                                        if col.lower().replace('_', '') in actual_col.lower().replace('_', '')]
                    
                    if potential_matches:
                        # Use the first potential match
                        chunk = chunk.rename(columns={potential_matches[0]: col})
                        logger.info(f"Mapped column {potential_matches[0]} to {col}")
                    else:
                        # If no match found, add empty column
                        chunk[col] = None
                        logger.warning(f"Added empty column {col}")
            
            # Handle data types
            # Convert date columns to proper format
            date_columns = ['order_date', 'return_date']
            for col in date_columns:
                if col in chunk.columns:
                    try:
                        chunk[col] = pd.to_datetime(chunk[col], errors='coerce')
                    except:
                        logger.warning(f"Failed to convert {col} to datetime. Setting to NaT.")
                        chunk[col] = pd.NaT
            
            # Ensure numeric columns are numeric
            numeric_columns = ['sales_qty', 'gross', 'return_qty']
            for col in numeric_columns:
                if col in chunk.columns:
                    try:
                        chunk[col] = pd.to_numeric(chunk[col], errors='coerce')
                    except:
                        logger.warning(f"Failed to convert {col} to numeric. Setting to NaN.")
                        chunk[col] = pd.NA
            
            # Create a primary key if missing
            if 'primary_key' in chunk.columns and chunk['primary_key'].isnull().all():
                # Generate a primary key from other columns
                chunk['primary_key'] = chunk.apply(
                    lambda row: f"{row.get('sales_order_no', '')}-{row.get('sku', '')}-{row.get('customer_emailid', '')}", 
                    axis=1
                )
                logger.info("Generated primary keys for data")
                
            # Ensure units_returned_flag is a string
            if 'units_returned_flag' in chunk.columns:
                chunk['units_returned_flag'] = chunk['units_returned_flag'].astype(str)
                
            # Select only the required columns in the specified order
            chunk = chunk[required_columns]
            
            # Insert into bronze layer
            conn.execute("BEGIN TRANSACTION")
            conn.register("temp_chunk", chunk)
            
            # Create a unique temporary table
            temp_table_name = f"temp_chunk_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
            create_temp_table_sql = f"""
            CREATE TEMPORARY TABLE {temp_table_name} (
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
            );
            """
            conn.execute(create_temp_table_sql)
            
            # Insert directly using SQL with explicit type casting
            columns = ", ".join(bronze_columns)
            placeholders = ", ".join(["?" for _ in bronze_columns])
            
            # Prepare data for insertion with unique primary keys
            primary_keys_seen = set()
            data_to_insert = []
            for _, row in chunk.iterrows():
                row_data = []
                # Make primary key unique
                pk = row['primary_key']
                if pk in primary_keys_seen:
                    # Add a suffix to make it unique
                    pk = f"{pk}-{datetime.now().strftime('%f')}"
                primary_keys_seen.add(pk)
                
                for i, col in enumerate(bronze_columns):
                    if col == 'primary_key':
                        val = pk
                    elif col == 'order_date' or col == 'return_date':
                        # Format dates properly
                        val = row[col]
                        if pd.notna(val):
                            if isinstance(val, str):
                                try:
                                    val = pd.to_datetime(val).strftime('%Y-%m-%d %H:%M:%S')
                                except:
                                    val = None
                            elif isinstance(val, pd.Timestamp):
                                val = val.strftime('%Y-%m-%d %H:%M:%S')
                            else:
                                val = None
                        else:
                            val = None
                    elif col == 'units_returned_flag':
                        # Ensure this is a string
                        val = str(row[col]) if pd.notna(row[col]) else 'Unknown'
                    else:
                        val = row[col] if pd.notna(row[col]) else None
                    row_data.append(val)
                data_to_insert.append(row_data)
            
            # Insert the data into temporary table
            insert_sql = f"INSERT INTO {temp_table_name} ({columns}) VALUES ({placeholders})"
            conn.executemany(insert_sql, data_to_insert)
            
            # Now insert into the main table, avoiding duplicates
            insert_from_temp_sql = f"""
            INSERT INTO bronze_return_order_data
            SELECT t.*
            FROM {temp_table_name} t
            LEFT JOIN bronze_return_order_data b ON t.primary_key = b.primary_key
            WHERE b.primary_key IS NULL;
            """
            conn.execute(insert_from_temp_sql)
            
            # Drop the temporary table
            conn.execute(f"DROP TABLE {temp_table_name}");
            conn.execute("COMMIT")
            
            logger.info(f"Processed chunk {chunk_num+1} with {len(chunk)} rows")
        
        # Get final count
        count = conn.execute("SELECT COUNT(*) FROM bronze_return_order_data").fetchone()[0]
        logger.info(f"Bronze layer now contains {count} rows")
        return True
    
    except Exception as e:
        logger.error(f"Error updating bronze layer: {str(e)}")
        conn.execute("ROLLBACK")
        return False

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
        logger.info("Starting database update process")
        
        # Connect to the database
        conn = get_database_connection()
        
        # Update bronze layer with new data
        if update_bronze_layer(conn, NEW_DATA_PATH):
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
        else:
            logger.error("Failed to update bronze layer")
            
    except Exception as e:
        logger.error(f"Process failed: {str(e)}")
        print(f"\nProcess failed: {str(e)}")
    finally:
        if 'conn' in locals():
            conn.close()
            logger.info("Database connection closed")

if __name__ == "__main__":
    main()
