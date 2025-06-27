import duckdb
import os
import pandas as pd
from humanize import naturalsize

def get_table_sizes():
    """Get sizes of all tables in the database"""
    conn = duckdb.connect("customer_clustering.db")
    
    # Get all tables
    tables = conn.execute("SHOW TABLES").fetchdf()
    
    # Create a results dataframe
    results = []
    
    # For each table, get row count and approximate size
    for table_name in tables['name']:
        try:
            # Get row count
            row_count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
            
            # Get column info
            columns = conn.execute(f"""
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_name = '{table_name}'
            """).fetchdf()
            
            # Get a sample row to estimate size
            sample = conn.execute(f"SELECT * FROM {table_name} LIMIT 1").fetchdf()
            
            # Estimate bytes per row (this is approximate)
            bytes_per_row = sum(sample.memory_usage()) / len(sample) if len(sample) > 0 else 0
            
            # Estimate total size
            estimated_size = bytes_per_row * row_count
            
            # Calculate compression ratio (DuckDB compresses data)
            compression_ratio = 0.5  # Estimated compression ratio
            
            # Adjusted size
            adjusted_size = estimated_size * compression_ratio
            
            results.append({
                'table_name': table_name,
                'row_count': row_count,
                'column_count': len(columns),
                'estimated_size_bytes': adjusted_size,
                'human_size': naturalsize(adjusted_size)
            })
        except Exception as e:
            results.append({
                'table_name': table_name,
                'row_count': 'ERROR',
                'column_count': 'ERROR',
                'estimated_size_bytes': 0,
                'human_size': f'Error: {str(e)}'
            })
    
    conn.close()
    
    # Return as DataFrame sorted by size
    df = pd.DataFrame(results)
    return df.sort_values('estimated_size_bytes', ascending=False)

def identify_temp_and_backup_tables():
    """Identify temporary and backup tables"""
    conn = duckdb.connect("customer_clustering.db")
    tables = conn.execute("SHOW TABLES").fetchall()
    conn.close()
    
    all_tables = [table[0] for table in tables]
    
    temp_tables = []
    backup_tables = []
    versioned_tables = []
    
    for table in all_tables:
        # Identify temporary tables
        if "_temp" in table.lower() or "_tmp" in table.lower():
            temp_tables.append(table)
        # Identify backup tables
        elif "_backup" in table.lower() or "_bak" in table.lower():
            backup_tables.append(table)
        # Identify versioned tables with timestamps
        elif "_20" in table:
            versioned_tables.append(table)
    
    return temp_tables, backup_tables, versioned_tables

def analyze_database():
    """Analyze the database and provide recommendations for size reduction"""
    print("=" * 80)
    print("DATABASE SIZE ANALYSIS")
    print("=" * 80)
    
    # Get current database file size
    db_size = os.path.getsize("customer_clustering.db")
    print(f"Current database size: {naturalsize(db_size)} ({db_size:,} bytes)")
    
    # Get table sizes
    print("\nAnalyzing table sizes...")
    table_sizes = get_table_sizes()
    
    # Display top 10 largest tables
    print("\nTop 10 largest tables:")
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(table_sizes.head(10))
    
    # Identify temporary and backup tables
    temp_tables, backup_tables, versioned_tables = identify_temp_and_backup_tables()
    
    print(f"\nFound {len(temp_tables)} temporary tables:")
    for table in temp_tables:
        size_info = table_sizes[table_sizes['table_name'] == table]
        size_str = size_info['human_size'].values[0] if len(size_info) > 0 else "Unknown"
        print(f"  - {table}: {size_str}")
    
    print(f"\nFound {len(backup_tables)} backup tables:")
    for table in backup_tables:
        size_info = table_sizes[table_sizes['table_name'] == table]
        size_str = size_info['human_size'].values[0] if len(size_info) > 0 else "Unknown"
        print(f"  - {table}: {size_str}")
    
    print(f"\nFound {len(versioned_tables)} versioned tables:")
    for table in versioned_tables:
        size_info = table_sizes[table_sizes['table_name'] == table]
        size_str = size_info['human_size'].values[0] if len(size_info) > 0 else "Unknown"
        print(f"  - {table}: {size_str}")
    
    # Calculate potential space savings
    removable_tables = temp_tables + backup_tables + versioned_tables
    
    # Filter table_sizes to only include removable tables
    removable_sizes = table_sizes[table_sizes['table_name'].isin(removable_tables)]
    
    # Calculate total potential savings
    total_savings = removable_sizes['estimated_size_bytes'].sum()
    
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    
    print(f"\nPotential space savings from removing temporary, backup, and versioned tables:")
    print(f"  {naturalsize(total_savings)} ({total_savings:,} bytes)")
    
    potential_new_size = db_size - total_savings
    print(f"\nEstimated database size after cleanup: {naturalsize(potential_new_size)} ({potential_new_size:,} bytes)")
    
    if potential_new_size > 1_000_000_000:  # 1GB
        print("\nAdditional actions needed to get below 1GB:")
        
        # Suggest keeping only essential core tables
        print("\n1. Keep only these essential core tables:")
        core_tables = ['bronze', 'silver', 'gold', 'gold_cluster_processed']
        for table in core_tables:
            size_info = table_sizes[table_sizes['table_name'] == table]
            size_str = size_info['human_size'].values[0] if len(size_info) > 0 else "Unknown"
            print(f"  - {table}: {size_str}")
        
        # Suggest vacuum
        print("\n2. Run VACUUM on the database to reclaim space")
        print("   This can compact the database and reclaim space from deleted data.")
        
        # Suggest table optimization
        print("\n3. For the bronze table (likely the largest):")
        print("   - Consider archiving older data")
        print("   - Convert TEXT columns to VARCHAR")
        print("   - Use appropriate data types for numeric columns")
    else:
        print("\nRemoving temporary and backup tables should be sufficient to get below 1GB.")

if __name__ == "__main__":
    try:
        import humanize
    except ImportError:
        import pip
        pip.main(['install', 'humanize'])
        import humanize
    
    analyze_database()
