import duckdb

# Connect to the database
conn = duckdb.connect('customer_clustering.db')

# Get the schema of the cluster_summary table
try:
    result = conn.execute("DESCRIBE cluster_summary").fetchall()
    print("=== CLUSTER_SUMMARY TABLE SCHEMA ===")
    for column in result:
        print(column)
except Exception as e:
    print(f"Error: {e}")

# Check the SQL for creating the table
try:
    create_sql = conn.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='cluster_summary'").fetchone()
    if create_sql:
        print("\n=== CREATE TABLE SQL ===")
        print(create_sql[0])
    else:
        print("\nTable cluster_summary doesn't exist yet.")
except Exception as e:
    print(f"Error getting CREATE TABLE SQL: {e}")

conn.close()
