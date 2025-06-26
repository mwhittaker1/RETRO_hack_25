import pandas as pd
import duckdb
import matplotlib.pyplot as plt
import seaborn as sns

# Connect to the database
conn = duckdb.connect('customer_clustering.db')

# Get database statistics
print("Database Overview:")
tables = conn.execute("SHOW TABLES").fetchall()
print(f"Tables in database: {[t[0] for t in tables]}")

# Check bronze layer
bronze_count = conn.execute("SELECT COUNT(*) FROM bronze_return_order_data").fetchone()[0]
bronze_customers = conn.execute("SELECT COUNT(DISTINCT customer_emailid) FROM bronze_return_order_data").fetchone()[0]
bronze_orders = conn.execute("SELECT COUNT(DISTINCT sales_order_no) FROM bronze_return_order_data").fetchone()[0]
bronze_returns = conn.execute("SELECT COUNT(DISTINCT return_no) FROM bronze_return_order_data").fetchone()[0]
bronze_categories = conn.execute("SELECT COUNT(DISTINCT class_) FROM bronze_return_order_data").fetchone()[0]

print("\nBronze Layer Statistics:")
print(f"Total rows: {bronze_count}")
print(f"Unique customers: {bronze_customers}")
print(f"Unique orders: {bronze_orders}")
print(f"Unique returns: {bronze_returns}")
print(f"Unique product categories: {bronze_categories}")

# Check silver layer
silver_count = conn.execute("SELECT COUNT(*) FROM silver_customer_features").fetchone()[0]
silver_customers = conn.execute("SELECT COUNT(DISTINCT customer_emailid) FROM silver_customer_features").fetchone()[0]

print("\nSilver Layer Statistics:")
print(f"Total rows: {silver_count}")
print(f"Unique customers: {silver_customers}")

# Get statistics on category_diversity_score
silver_diversity_stats = conn.execute("""
SELECT 
    MIN(category_diversity_score) as min_score,
    MAX(category_diversity_score) as max_score,
    AVG(category_diversity_score) as avg_score,
    STDDEV(category_diversity_score) as std_score
FROM silver_customer_features
""").fetchone()

print("\nCategory Diversity Score Statistics:")
print(f"Min: {silver_diversity_stats[0]}")
print(f"Max: {silver_diversity_stats[1]}")
print(f"Average: {silver_diversity_stats[2]}")
print(f"Standard Deviation: {silver_diversity_stats[3]}")

# Check gold layer
gold_count = conn.execute("SELECT COUNT(*) FROM gold_cluster_processed").fetchone()[0]
gold_customers = conn.execute("SELECT COUNT(DISTINCT customer_emailid) FROM gold_cluster_processed").fetchone()[0]
# gold_clusters = conn.execute("SELECT COUNT(DISTINCT cluster) FROM gold_cluster_processed").fetchone()[0]

print("\nGold Layer Statistics:")
print(f"Total rows: {gold_count}")
print(f"Unique customers: {gold_customers}")
# print(f"Number of clusters: {gold_clusters}")

# Get statistics on category_diversity_score_scaled
gold_diversity_stats = conn.execute("""
SELECT 
    MIN(category_diversity_score_scaled) as min_score,
    MAX(category_diversity_score_scaled) as max_score,
    AVG(category_diversity_score_scaled) as avg_score,
    STDDEV(category_diversity_score_scaled) as std_score
FROM gold_cluster_processed
""").fetchone()

print("\nScaled Category Diversity Score Statistics:")
print(f"Min: {gold_diversity_stats[0]}")
print(f"Max: {gold_diversity_stats[1]}")
print(f"Average: {gold_diversity_stats[2]}")
print(f"Standard Deviation: {gold_diversity_stats[3]}")

# Check for any remaining duplicate primary keys in bronze layer
bronze_dups = conn.execute("""
SELECT primary_key, COUNT(*) as count
FROM bronze_return_order_data
GROUP BY primary_key
HAVING COUNT(*) > 1
ORDER BY COUNT(*) DESC
LIMIT 10
""").fetchdf()

if len(bronze_dups) > 0:
    print("\nWARNING: Found duplicate primary keys in bronze layer:")
    print(bronze_dups)
else:
    print("\nNo duplicate primary keys found in bronze layer")

# Check for any duplicate customer emails in silver layer
silver_dups = conn.execute("""
SELECT customer_emailid, COUNT(*) as count
FROM silver_customer_features
GROUP BY customer_emailid
HAVING COUNT(*) > 1
ORDER BY COUNT(*) DESC
LIMIT 10
""").fetchdf()

if len(silver_dups) > 0:
    print("\nWARNING: Found duplicate customer emails in silver layer:")
    print(silver_dups)
else:
    print("No duplicate customer emails found in silver layer")

# Check for any duplicate customer emails in gold layer
gold_dups = conn.execute("""
SELECT customer_emailid, COUNT(*) as count
FROM gold_cluster_processed
GROUP BY customer_emailid
HAVING COUNT(*) > 1
ORDER BY COUNT(*) DESC
LIMIT 10
""").fetchdf()

if len(gold_dups) > 0:
    print("\nWARNING: Found duplicate customer emails in gold layer:")
    print(gold_dups)
else:
    print("No duplicate customer emails found in gold layer")

# Create a histogram of category_diversity_score in silver layer
plt.figure(figsize=(10, 6))
silver_diversity = conn.execute("SELECT category_diversity_score FROM silver_customer_features").fetchnumpy()['category_diversity_score']
plt.hist(silver_diversity, bins=50)
plt.title('Distribution of Category Diversity Score in Silver Layer')
plt.xlabel('Category Diversity Score')
plt.ylabel('Count')
plt.savefig('category_diversity_score_distribution.png')

# Create a histogram of category_diversity_score_scaled in gold layer
plt.figure(figsize=(10, 6))
gold_diversity = conn.execute("SELECT category_diversity_score_scaled FROM gold_cluster_processed").fetchnumpy()['category_diversity_score_scaled']
plt.hist(gold_diversity, bins=50)
plt.title('Distribution of Scaled Category Diversity Score in Gold Layer')
plt.xlabel('Scaled Category Diversity Score')
plt.ylabel('Count')
plt.savefig('category_diversity_score_scaled_distribution.png')

# Check for new customers in bronze layer not yet in silver layer
new_customers = conn.execute("""
SELECT DISTINCT b.customer_emailid
FROM bronze_return_order_data b
LEFT JOIN silver_customer_features s ON b.customer_emailid = s.customer_emailid
WHERE s.customer_emailid IS NULL
LIMIT 10
""").fetchdf()

print("\nCustomers in bronze layer not yet in silver layer (sample):")
print(new_customers)

# Close the connection
conn.close()

print("\nAnalysis complete. Distribution plots saved to disk.")
