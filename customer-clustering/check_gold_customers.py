import duckdb

# Connect to the database
conn = duckdb.connect('customer_clustering.db')

# Check if new customers are in the gold layer
print('Total customers in gold layer:', conn.execute('SELECT COUNT(DISTINCT customer_emailid) FROM gold_cluster_processed').fetchone()[0])
print('\nCheck if new customers were added to gold layer:')

new_emails = [
    'AREVEAL@FREEPEOPLE.COM', 
    'KMOSES@FREEPEOPLE.COM', 
    'MHANNUM@FREEPEOPLE.COM', 
    'LDRONSKI@ANTHROPOLOGIE.COM', 
    'MRAMIREZ-MENDEZ@URBANOUTFITTERS.COM',
    'JCOOPER1@ANTHROPOLOGIE.COM', 
    'LSIMONET@URBANOUTFITTERS.COM', 
    'PLILLIS@URBANOUTFITTERS.COM', 
    'Melissa.dicerbo@gmail.com'
]

for email in new_emails:
    count = conn.execute(f"SELECT COUNT(1) FROM gold_cluster_processed WHERE customer_emailid = '{email}'").fetchone()[0]
    print(f"{email}: {'Found in gold layer' if count > 0 else 'NOT found in gold layer'}")

# Close connection
conn.close()
