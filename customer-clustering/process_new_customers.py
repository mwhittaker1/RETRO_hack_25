"""
Process New Customers into Silver Layer
This script identifies customers in the bronze layer not yet in the silver layer,
and processes them with full feature engineering.
"""

import duckdb
import pandas as pd
import os
import sys
import logging
from datetime import datetime

# Setup logging
log_filename = f'process_new_customers_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Constants
DB_PATH = "customer_clustering.db"

def process_new_customers():
    """Process new customers from bronze layer into silver layer"""
    try:
        # Connect to database
        conn = duckdb.connect(DB_PATH)
        logger.info(f"Connected to database: {DB_PATH}")
        
        # Identify new customers
        new_customers_query = """
        SELECT DISTINCT b.customer_emailid
        FROM bronze_return_order_data b
        LEFT JOIN silver_customer_features s ON b.customer_emailid = s.customer_emailid
        WHERE s.customer_emailid IS NULL
        """
        
        new_customers_df = conn.execute(new_customers_query).fetchdf()
        new_customer_count = len(new_customers_df)
        
        if new_customer_count == 0:
            logger.info("No new customers to process")
            return True
        
        logger.info(f"Processing {new_customer_count} new customers into silver layer")
        logger.info(f"New customers: {new_customers_df['customer_emailid'].tolist()}")
        
        # Get basic features for new customers
        logger.info("Calculating basic features for new customers")
        
        # Create a temporary table with category diversity scores for all customers
        logger.info("Calculating category diversity scores")
        unique_categories = conn.execute("SELECT COUNT(DISTINCT class_) FROM bronze_return_order_data").fetchone()[0]
        logger.info(f"Found {unique_categories} unique product categories")
        
        diversity_query = f"""
        CREATE TEMPORARY TABLE temp_new_customer_features AS
        WITH customer_categories AS (
            SELECT 
                customer_emailid,
                class_ as category,
                COUNT(*) as category_purchases,
                SUM(CASE WHEN return_qty > 0 THEN 1 ELSE 0 END) as category_returns,
                SUM(sales_qty) as category_sales
            FROM bronze_return_order_data
            GROUP BY customer_emailid, class_
            HAVING customer_emailid IN (
                SELECT customer_emailid FROM ({new_customers_query})
            )
        ),
        customer_totals AS (
            SELECT 
                customer_emailid,
                COUNT(DISTINCT category) as unique_categories,
                SUM(category_purchases) as total_purchases,
                SUM(category_returns) as total_returns
            FROM customer_categories
            GROUP BY customer_emailid
        ),
        customer_orders AS (
            SELECT 
                customer_emailid,
                COUNT(DISTINCT sales_order_no) as order_count,
                MIN(order_date) as first_order_date,
                MAX(order_date) as last_order_date
            FROM bronze_return_order_data
            GROUP BY customer_emailid
            HAVING customer_emailid IN (
                SELECT customer_emailid FROM ({new_customers_query})
            )
        ),
        customer_returns AS (
            SELECT 
                customer_emailid,
                COUNT(DISTINCT return_no) as return_count,
                COUNT(DISTINCT CASE WHEN return_qty > 0 THEN sales_order_no ELSE NULL END) as orders_with_returns
            FROM bronze_return_order_data
            GROUP BY customer_emailid
            HAVING customer_emailid IN (
                SELECT customer_emailid FROM ({new_customers_query})
            )
        )
        SELECT 
            t.customer_emailid,
            CAST(t.unique_categories AS DOUBLE) / {unique_categories}.0 as category_diversity_score,
            o.order_count as sales_order_no_nunique,
            t.unique_categories as sku_nunique,
            r.return_count as items_returned_count,
            t.total_returns as return_qty_sum,
            t.total_purchases as sales_qty_sum,
            CASE WHEN o.order_count > 0 THEN CAST(t.total_purchases AS DOUBLE) / o.order_count ELSE 0 END as sales_qty_mean,
            CASE WHEN t.total_purchases > 0 THEN CAST(t.total_returns AS DOUBLE) / t.total_purchases ELSE 0 END as return_rate,
            CASE WHEN o.order_count > 0 THEN CAST(r.return_count AS DOUBLE) / o.order_count ELSE 0 END as avg_returns_per_order,
            CASE WHEN o.order_count > 0 THEN CAST(r.orders_with_returns AS DOUBLE) / o.order_count ELSE 0 END as return_frequency_ratio,
            DATEDIFF('day', o.first_order_date, o.last_order_date) as customer_lifetime_days
        FROM customer_totals t
        JOIN customer_orders o ON t.customer_emailid = o.customer_emailid
        JOIN customer_returns r ON t.customer_emailid = r.customer_emailid
        """
        
        conn.execute(diversity_query)
        
        # Show the calculated features
        logger.info("Features calculated for new customers:")
        features_df = conn.execute("SELECT * FROM temp_new_customer_features").fetchdf()
        logger.info(f"Feature columns: {features_df.columns.tolist()}")
        logger.info(f"Sample features:\n{features_df.head()}")
        
        # Get silver layer schema
        silver_columns = conn.execute("PRAGMA table_info(silver_customer_features);").fetchdf()
        silver_column_names = silver_columns['name'].tolist()
        logger.info(f"Silver layer has {len(silver_column_names)} columns")
        
        # Get temp table schema
        temp_columns = conn.execute("PRAGMA table_info(temp_new_customer_features);").fetchdf()
        temp_column_names = temp_columns['name'].tolist()
        logger.info(f"Temp table has {len(temp_column_names)} columns")
        
        # Create template for new customers
        # First get a template row from an existing customer
        template_query = """
        SELECT * FROM silver_customer_features LIMIT 1
        """
        template_df = conn.execute(template_query).fetchdf()
        
        # Now create a new customer template
        for _, row in features_df.iterrows():
            customer_email = row['customer_emailid']
            logger.info(f"Creating silver layer entry for {customer_email}")
            
            # Create a new row based on the template
            new_row = template_df.copy()
            
            # Update with new customer data
            new_row['customer_emailid'] = customer_email
            
            # Update features we have calculated
            for col in temp_column_names:
                if col in silver_column_names:
                    new_row[col] = row[col]
            
            # Insert the new customer into the silver layer
            # Convert to a format that can be inserted
            insert_values = []
            for col in silver_column_names:
                if col in new_row.columns:
                    value = new_row[col].iloc[0]
                    if pd.isna(value):
                        insert_values.append("NULL")
                    elif isinstance(value, str):
                        insert_values.append(f"'{value}'")
                    else:
                        insert_values.append(str(value))
                else:
                    insert_values.append("NULL")
            
            # Insert the new customer
            insert_query = f"""
            INSERT INTO silver_customer_features ({', '.join(silver_column_names)})
            VALUES ({', '.join(insert_values)})
            """
            
            try:
                conn.execute(insert_query)
                logger.info(f"Inserted {customer_email} into silver layer")
            except Exception as e:
                logger.error(f"Error inserting {customer_email}: {str(e)}")
        
        # Verify new customers were added
        added_count = conn.execute(f"""
        SELECT COUNT(*) FROM silver_customer_features
        WHERE customer_emailid IN (
            SELECT customer_emailid FROM ({new_customers_query})
        )
        """).fetchone()[0]
        
        logger.info(f"Added {added_count} out of {new_customer_count} new customers to silver layer")
        
        # Drop temporary table
        conn.execute("DROP TABLE IF EXISTS temp_new_customer_features")
        
        # Now update the gold layer with the new customers
        if added_count > 0:
            logger.info("Updating gold layer with new customers")
            
            # Get gold layer schema
            gold_columns = conn.execute("PRAGMA table_info(gold_cluster_processed);").fetchdf()
            gold_column_names = gold_columns['name'].tolist()
            
            # Create a gold layer template
            template_gold_query = """
            SELECT * FROM gold_cluster_processed LIMIT 1
            """
            template_gold_df = conn.execute(template_gold_query).fetchdf()
            
            # Get silver layer data for new customers
            new_silver_query = f"""
            SELECT * FROM silver_customer_features
            WHERE customer_emailid IN (
                SELECT customer_emailid FROM ({new_customers_query})
            )
            """
            new_silver_df = conn.execute(new_silver_query).fetchdf()
            
            # Process each new customer into gold layer
            for _, row in new_silver_df.iterrows():
                customer_email = row['customer_emailid']
                logger.info(f"Creating gold layer entry for {customer_email}")
                
                # Create a new row based on the template
                new_gold_row = template_gold_df.copy()
                
                # Update with new customer data
                new_gold_row['customer_emailid'] = customer_email
                
                # Copy over features from silver that match gold
                for col in new_silver_df.columns:
                    if col in gold_column_names:
                        new_gold_row[col] = row[col]
                
                # Scale the category_diversity_score
                diversity_score = row['category_diversity_score']
                if diversity_score is not None and not pd.isna(diversity_score):
                    # Get min and max from existing data
                    scaling_stats = conn.execute("""
                    SELECT MIN(category_diversity_score), MAX(category_diversity_score)
                    FROM silver_customer_features
                    WHERE category_diversity_score > 0
                    """).fetchone()
                    
                    min_score, max_score = scaling_stats
                    
                    if min_score is not None and max_score is not None and min_score != max_score:
                        scaled_score = (diversity_score - min_score) / (max_score - min_score)
                        new_gold_row['category_diversity_score_scaled'] = scaled_score
                        logger.info(f"Scaled diversity score for {customer_email}: {diversity_score} -> {scaled_score}")
                
                # Insert the new customer into the gold layer
                insert_values = []
                for col in gold_column_names:
                    if col in new_gold_row.columns:
                        value = new_gold_row[col].iloc[0]
                        if pd.isna(value):
                            insert_values.append("NULL")
                        elif isinstance(value, str):
                            insert_values.append(f"'{value}'")
                        else:
                            insert_values.append(str(value))
                    else:
                        insert_values.append("NULL")
                
                # Insert the new customer
                insert_gold_query = f"""
                INSERT INTO gold_cluster_processed ({', '.join(gold_column_names)})
                VALUES ({', '.join(insert_values)})
                """
                
                try:
                    conn.execute(insert_gold_query)
                    logger.info(f"Inserted {customer_email} into gold layer")
                except Exception as e:
                    logger.error(f"Error inserting {customer_email} into gold layer: {str(e)}")
            
            # Verify new customers were added to gold
            gold_added_count = conn.execute(f"""
            SELECT COUNT(*) FROM gold_cluster_processed
            WHERE customer_emailid IN (
                SELECT customer_emailid FROM ({new_customers_query})
            )
            """).fetchone()[0]
            
            logger.info(f"Added {gold_added_count} out of {added_count} new customers to gold layer")
        
        # Close the connection
        conn.close()
        
        logger.info("Finished processing new customers")
        return True
    except Exception as e:
        logger.error(f"Error processing new customers: {str(e)}")
        if 'conn' in locals():
            conn.close()
        return False

if __name__ == "__main__":
    process_new_customers()
