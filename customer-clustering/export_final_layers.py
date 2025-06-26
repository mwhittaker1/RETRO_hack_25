import duckdb

# Connect to the database
conn = duckdb.connect('customer_clustering.db')

# Check database layer counts
print('Database layer counts:')
print(f"Bronze: {conn.execute('SELECT COUNT(*) FROM bronze_return_order_data').fetchone()[0]} rows, {conn.execute('SELECT COUNT(DISTINCT customer_emailid) FROM bronze_return_order_data').fetchone()[0]} unique customers")
print(f"Silver: {conn.execute('SELECT COUNT(*) FROM silver_customer_features').fetchone()[0]} rows, {conn.execute('SELECT COUNT(DISTINCT customer_emailid) FROM silver_customer_features').fetchone()[0]} unique customers")
print(f"Gold: {conn.execute('SELECT COUNT(*) FROM gold_cluster_processed').fetchone()[0]} rows, {conn.execute('SELECT COUNT(DISTINCT customer_emailid) FROM gold_cluster_processed').fetchone()[0]} unique customers")

# Export final silver and gold layers
print("\nExporting final layers:")
from datetime import datetime
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

silver_df = conn.execute("SELECT * FROM silver_customer_features").fetchdf()
silver_filename = f"silver_customer_features_final_{timestamp}.csv"
silver_df.to_csv(silver_filename, index=False)
print(f"Exported {len(silver_df)} rows to {silver_filename}")

gold_df = conn.execute("SELECT * FROM gold_cluster_processed").fetchdf()
gold_filename = f"gold_cluster_processed_final_{timestamp}.csv"
gold_df.to_csv(gold_filename, index=False)
print(f"Exported {len(gold_df)} rows to {gold_filename}")

# Close connection
conn.close()
