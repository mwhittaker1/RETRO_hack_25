#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to examine the schema and sample data from the temp_bronze_import table
"""

import duckdb
import pandas as pd

def get_connection(db_path="customer_clustering.db"):
    """Get a connection to the database"""
    try:
        return duckdb.connect(db_path)
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return None

def examine_temp_table():
    """Examine the temp_bronze_import table"""
    conn = get_connection()
    if conn is None:
        return
    
    try:
        # Get schema
        print("Schema of temp_bronze_import:")
        print("-" * 60)
        schema = conn.execute("DESCRIBE temp_bronze_import").fetchdf()
        print(schema)
        
        # Get sample data
        print("\nSample data from temp_bronze_import (first 5 rows):")
        print("-" * 60)
        sample = conn.execute("SELECT * FROM temp_bronze_import LIMIT 5").fetchdf()
        pd.set_option('display.max_columns', None)
        print(sample)
        
        # Check for unique customers
        print("\nUnique customer count in temp_bronze_import:")
        print("-" * 60)
        unique_customers = conn.execute("SELECT COUNT(DISTINCT customer_emailid) FROM temp_bronze_import").fetchone()[0]
        print(f"Unique customers: {unique_customers}")
        
        # Check if these customers exist in the main bronze table
        print("\nCustomers from temp_bronze_import existing in main bronze table:")
        print("-" * 60)
        overlap = conn.execute("""
            SELECT COUNT(DISTINCT t.customer_emailid) 
            FROM temp_bronze_import t
            JOIN bronze_return_order_data b ON t.customer_emailid = b.customer_emailid
        """).fetchone()[0]
        print(f"Customers in both tables: {overlap}")
        print(f"New customers only in temp: {unique_customers - overlap}")
        
        # Get date range in temp table
        print("\nDate range in temp_bronze_import:")
        print("-" * 60)
        try:
            date_range = conn.execute("""
                SELECT MIN(order_date), MAX(order_date) FROM temp_bronze_import
            """).fetchone()
            print(f"Earliest order date: {date_range[0]}")
            print(f"Latest order date: {date_range[1]}")
        except:
            print("Could not determine date range (order_date field may not exist)")
        
    except Exception as e:
        print(f"Error examining temp_bronze_import: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    examine_temp_table()
