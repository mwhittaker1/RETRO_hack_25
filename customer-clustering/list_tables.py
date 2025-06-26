import duckdb

# Connect to the database
conn = duckdb.connect('customer_clustering.db')

# Get all tables
tables = conn.execute("SHOW TABLES").fetchall()

print("Tables in customer_clustering.db:")
print("================================")
for table in tables:
    print(table[0])
    
# Get table statistics
print("\nTable Row Counts:")
print("================")
for table in tables:
    try:
        row_count = conn.execute(f"SELECT COUNT(*) FROM {table[0]}").fetchone()[0]
        print(f"{table[0]}: {row_count} rows")
    except Exception as e:
        print(f"{table[0]}: Error getting count - {str(e)}")

# Close the connection
conn.close()
