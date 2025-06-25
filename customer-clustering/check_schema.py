import duckdb

# Connect to the database
conn = duckdb.connect('customer_clustering.db')

# Get the schema of the clustering_results table
try:
    result = conn.execute("DESCRIBE clustering_results").fetchall()
    print("=== CLUSTERING_RESULTS TABLE SCHEMA ===")
    for column in result:
        print(column)
except Exception as e:
    print(f"Error: {e}")

# Check the SQL for creating the table
try:
    create_sql = conn.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='clustering_results'").fetchone()
    if create_sql:
        print("\n=== CREATE TABLE SQL ===")
        print(create_sql[0])
    else:
        print("\nTable clustering_results doesn't exist yet.")
except Exception as e:
    print(f"Error getting CREATE TABLE SQL: {e}")

# List all tables
try:
    tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    print("\n=== ALL TABLES ===")
    for table in tables:
        print(table[0])
except Exception as e:
    print(f"Error listing tables: {e}")

conn.close()
