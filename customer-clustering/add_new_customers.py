"""
Simple script to add new customers to silver and gold layers
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
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Connect to database
conn = duckdb.connect('customer_clustering.db')

try:
    # Identify new customers
    logger.info("Identifying new customers")
    
    new_customers_query = """
    SELECT DISTINCT b.customer_emailid
    FROM bronze_return_order_data b
    LEFT JOIN silver_customer_features s ON b.customer_emailid = s.customer_emailid
    WHERE s.customer_emailid IS NULL
    """
    
    new_customers = conn.execute(new_customers_query).fetchdf()
    logger.info(f"Found {len(new_customers)} new customers")
    logger.info(f"New customers: {new_customers['customer_emailid'].tolist()}")
    
    # Calculate features
    logger.info("Calculating features for new customers")
    
    # Get the actual category count
    unique_categories = conn.execute("SELECT COUNT(DISTINCT class_) FROM bronze_return_order_data").fetchone()[0]
    logger.info(f"Found {unique_categories} unique product categories")
    
    # Create insert statements for silver layer
    for _, customer in new_customers.iterrows():
        email = customer['customer_emailid']
        logger.info(f"Processing {email}")
        
        # Calculate category diversity score
        diversity_query = f"""
        WITH customer_categories AS (
            SELECT 
                customer_emailid,
                class_ as category,
                COUNT(*) as count
            FROM bronze_return_order_data
            WHERE customer_emailid = '{email}'
            GROUP BY customer_emailid, class_
        )
        SELECT 
            COUNT(DISTINCT category) as unique_categories
        FROM customer_categories
        """
        
        unique_customer_categories = conn.execute(diversity_query).fetchone()[0]
        category_diversity_score = float(unique_customer_categories) / float(unique_categories)
        
        logger.info(f"{email} has {unique_customer_categories} unique categories out of {unique_categories} total")
        logger.info(f"Category diversity score: {category_diversity_score}")
        
        # Insert into silver layer with just the customer_emailid and category_diversity_score
        silver_insert = f"""
        INSERT INTO silver_customer_features (customer_emailid, category_diversity_score)
        VALUES ('{email}', {category_diversity_score})
        """
        
        try:
            conn.execute(silver_insert)
            logger.info(f"Added {email} to silver layer")
            
            # Now add to gold layer with scaled diversity score
            # Get min and max for scaling
            scaling_stats = conn.execute("""
            SELECT MIN(category_diversity_score), MAX(category_diversity_score)
            FROM silver_customer_features
            WHERE category_diversity_score > 0
            """).fetchone()
            
            min_score, max_score = scaling_stats
            scaled_score = (category_diversity_score - min_score) / (max_score - min_score)
            
            # Insert into gold layer
            gold_insert = f"""
            INSERT INTO gold_cluster_processed (customer_emailid, category_diversity_score_scaled)
            VALUES ('{email}', {scaled_score})
            """
            
            conn.execute(gold_insert)
            logger.info(f"Added {email} to gold layer with scaled score: {scaled_score}")
        except Exception as e:
            logger.error(f"Error adding {email}: {str(e)}")
    
    # Check if customers were added
    silver_count = conn.execute(f"""
    SELECT COUNT(*) FROM silver_customer_features
    WHERE customer_emailid IN (
        SELECT customer_emailid FROM ({new_customers_query})
    )
    """).fetchone()[0]
    
    gold_count = conn.execute(f"""
    SELECT COUNT(*) FROM gold_cluster_processed
    WHERE customer_emailid IN (
        SELECT customer_emailid FROM ({new_customers_query})
    )
    """).fetchone()[0]
    
    logger.info(f"Final count: {silver_count} customers added to silver, {gold_count} added to gold")
    
except Exception as e:
    logger.error(f"Error: {str(e)}")
finally:
    conn.close()
    logger.info("Database connection closed")
