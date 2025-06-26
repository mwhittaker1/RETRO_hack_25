import duckdb

# Connect to the database
conn = duckdb.connect('customer_clustering.db')

# Check if new customers are in the silver layer
print('Total customers in silver layer:', conn.execute('SELECT COUNT(DISTINCT customer_emailid) FROM silver_customer_features').fetchone()[0])
print('\nCheck if new customers were added to silver layer:')

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
    count = conn.execute(f"SELECT COUNT(1) FROM silver_customer_features WHERE customer_emailid = '{email}'").fetchone()[0]
    print(f"{email}: {'Found in silver layer' if count > 0 else 'NOT found in silver layer'}")

# Close connection
conn.close()
