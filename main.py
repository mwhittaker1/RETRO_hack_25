import pandas as pd
import openpyxl


from functions import perform_clustering, find_optimal_clusters, analyze_clusters, visualize_clusters, export_results, get_cluster_customers, ReturnsClusteringAnalysis

# Load your data
df = pd.read_excel('RETRO_SAMPLE.xlsx')

# Initialize analyzer
analyzer = ReturnsClusteringAnalysis(df)

# Find optimal number of clusters
optimal_k, scores = analyzer.find_optimal_clusters()

# Perform clustering
clusters, centers = analyzer.perform_clustering(n_clusters=optimal_k)

# Analyze results
summary, interpretations = analyzer.analyze_clusters()

# Visualize
analyzer.visualize_clusters()

# Export results
results = analyzer.export_results()

# Get customers in specific cluster (e.g., high-risk customers)
high_risk_customers = analyzer.get_cluster_customers(cluster_id=0)