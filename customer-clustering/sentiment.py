import pandas as pd

# Load the CSV files
left_csv_path = 'customer-clustering\sentiment_and_raw\export_202506251146.csv'      # Replace with your actual left CSV file path
right_csv_path = 'customer-clustering\sentiment_and_raw\RETURNS_CODES_SENT.csv'    # Replace with your actual right CSV file path

# Read both CSVs
left_df = pd.read_csv(left_csv_path)
right_df = pd.read_csv(right_csv_path, usecols=['RETURN_REASON', 'SENTIMENT SCORE'])

# Perform the left join on 'order line header'
merged_df = pd.merge(left_df, right_df, on='RETURN_REASON', how='left')

# Save to a new CSV
output_csv_path = 'new_base_returns_sku_reasoncodes_sent.csv'
merged_df.to_csv(output_csv_path, index=False)

print(f"Left join complete. Output saved to '{output_csv_path}'.")