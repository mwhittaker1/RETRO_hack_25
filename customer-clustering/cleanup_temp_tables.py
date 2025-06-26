#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to clean up temporary and backup tables from the customer clustering database
"""

import os
import duckdb
from datetime import datetime

def get_connection(db_path="customer_clustering.db"):
    """Get a connection to the database"""
    try:
        return duckdb.connect(db_path)
    except duckdb.IOException as e:
        print(f"\nERROR: Cannot connect to database. It may be in use by another process.")
        print(f"Exception details: {e}")
        print("\nPlease ensure all other processes using the database are closed, including:")
        print("- Any running Jupyter notebooks")
        print("- Python scripts")
        print("- Interactive database sessions")
        print("\nThen run this script again.")
        return None

def list_all_tables():
    """List all tables in the database"""
    conn = get_connection()
    if conn is None:
        return []
    tables = conn.execute("SHOW TABLES").fetchall()
    conn.close()
    return [table[0] for table in tables]

def identify_temp_and_backup_tables(tables):
    """Identify temporary and backup tables"""
    temp_tables = []
    backup_tables = []
    
    for table in tables:
        # Identify temporary tables (those with temp, tmp, or _old in name)
        if "_temp" in table.lower() or "_tmp" in table.lower() or "_old" in table.lower():
            temp_tables.append(table)
        # Identify backup tables (those with backup, bak, or date patterns)
        elif "_backup" in table.lower() or "_bak" in table.lower() or "_20" in table:
            backup_tables.append(table)
            
    return temp_tables, backup_tables

def drop_tables(tables_to_drop, dry_run=True):
    """Drop the specified tables"""
    conn = get_connection()
    if conn is None:
        return
    
    print(f"\n{'DRY RUN - ' if dry_run else ''}Dropping {len(tables_to_drop)} tables:")
    
    for table in tables_to_drop:
        if dry_run:
            print(f"  Would drop: {table}")
        else:
            try:
                conn.execute(f"DROP TABLE IF EXISTS {table}")
                print(f"  Dropped: {table}")
            except Exception as e:
                print(f"  Error dropping {table}: {e}")
    
    conn.close()

def create_backup_of_important_tables():
    """Create backups of important tables before cleaning up"""
    important_tables = ['bronze', 'silver', 'gold', 'gold_cluster_processed']
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    conn = get_connection()
    if conn is None:
        return
    
    print("\nCreating backups of important tables:")
    
    for table in important_tables:
        try:
            backup_table = f"{table}_backup_{timestamp}"
            conn.execute(f"CREATE TABLE IF NOT EXISTS {backup_table} AS SELECT * FROM {table}")
            print(f"  Created backup: {backup_table}")
        except Exception as e:
            print(f"  Error backing up {table}: {e}")
    
    conn.close()

def main():
    print("=" * 60)
    print("DATABASE CLEANUP UTILITY")
    print("=" * 60)
    
    # List all tables
    all_tables = list_all_tables()
    if not all_tables:  # If we couldn't connect to the database
        return
        
    print(f"Found {len(all_tables)} tables in the database.")
    
    # Identify temporary and backup tables
    temp_tables, backup_tables = identify_temp_and_backup_tables(all_tables)
    
    print(f"\nIdentified {len(temp_tables)} temporary tables:")
    for table in temp_tables:
        print(f"  - {table}")
        
    print(f"\nIdentified {len(backup_tables)} backup tables:")
    for table in backup_tables:
        print(f"  - {table}")
    
    # Ask user for confirmation
    user_input = ""
    while user_input.lower() not in ['y', 'n', 'd']:
        user_input = input("\nDo you want to proceed with cleanup? [Y]es/[N]o/[D]ry run: ").lower()
    
    if user_input == 'n':
        print("Cleanup cancelled.")
        return
        
    # Create backups of important tables if not doing dry run
    if user_input == 'y':
        create_backup_of_important_tables()
    
    # Drop tables based on user input
    drop_tables(temp_tables + backup_tables, dry_run=(user_input == 'd'))
    
    if user_input == 'y':
        print("\nCleanup completed successfully!")
    elif user_input == 'd':
        print("\nDry run completed. No changes were made.")

if __name__ == "__main__":
    main()
