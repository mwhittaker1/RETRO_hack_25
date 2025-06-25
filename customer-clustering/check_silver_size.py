import duckdb

# Connect to the database
conn = duckdb.connect('customer_clustering.db')

# Get the size of the silver layer
row_count = conn.execute("SELECT COUNT(*) FROM silver_customer_features").fetchone()[0]
column_count = conn.execute("PRAGMA table_info(silver_customer_features)").fetchdf().shape[0]

print(f"Silver layer statistics:")
print(f"Number of rows: {row_count}")
print(f"Number of columns: {column_count}")

# Check if the dataset is too large for Excel (Excel limit ~1M rows)
if row_count > 1000000:
    print("Dataset is too large for Excel, will export as CSV")
else:
    print("Dataset can fit in Excel format")

conn.close()
