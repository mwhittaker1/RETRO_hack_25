import duckdb

# Connect to the database
conn = duckdb.connect('customer_clustering.db')

# Tables to examine
tables_to_examine = ['bronze_return_order_data', 'silver_customer_features', 'gold_cluster_processed']

# Get schema for each table
for table in tables_to_examine:
    print(f"\nSchema for {table}:")
    print("=" * (len(f"Schema for {table}:") + 1))
    schema = conn.execute(f"PRAGMA table_info({table});").fetchdf()
    print(schema[['name', 'type']])
    
    # Get sample data
    print(f"\nSample data from {table} (first 5 rows):")
    print("=" * (len(f"Sample data from {table} (first 5 rows):") + 1))
    try:
        sample = conn.execute(f"SELECT * FROM {table} LIMIT 5").fetchdf()
        print(sample)
    except Exception as e:
        print(f"Error getting sample: {str(e)}")

# Customer statistics
print("\nCustomer counts across layers:")
print("==============================")
bronze_customers = conn.execute("SELECT COUNT(DISTINCT customer_emailid) FROM bronze_return_order_data").fetchone()[0]
silver_customers = conn.execute("SELECT COUNT(DISTINCT customer_emailid) FROM silver_customer_features").fetchone()[0]
gold_customers = conn.execute("SELECT COUNT(DISTINCT customer_emailid) FROM gold_cluster_processed").fetchone()[0]

print(f"Bronze layer unique customers: {bronze_customers}")
print(f"Silver layer unique customers: {silver_customers}")
print(f"Gold layer unique customers: {gold_customers}")

# Examine new customers
print("\nNew customers in bronze layer but not in silver:")
print("=============================================")
new_customers = conn.execute("""
SELECT DISTINCT b.customer_emailid
FROM bronze_return_order_data b
LEFT JOIN silver_customer_features s ON b.customer_emailid = s.customer_emailid
WHERE s.customer_emailid IS NULL
LIMIT 10
""").fetchdf()
print(new_customers)

# Close the connection
conn.close()
