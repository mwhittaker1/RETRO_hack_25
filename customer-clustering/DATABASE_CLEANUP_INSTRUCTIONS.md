# Database Cleanup Instructions

## Cleaning up Temporary and Backup Tables

After completing the clustering analysis, you may want to clean up temporary and backup tables from the database to save disk space and keep your database organized.

Follow these steps:

1. **Close all connections to the database**:
   - Close any Jupyter notebooks that are using the database
   - Stop any Python scripts that might be accessing the database
   - Close any database GUI tools

2. **Run the cleanup script**:
   ```powershell
   cd "c:\Code\Local Code\URBN\RETRO_hack_25\customer-clustering"
   python cleanup_temp_tables.py
   ```

3. **Review the tables to be removed**:
   - The script will list all temporary and backup tables it found
   - Temporary tables include those with "_temp", "_tmp", or "_old" in the name
   - Backup tables include those with "_backup", "_bak", or date patterns in the name

4. **Choose an action**:
   - Type `Y` to proceed with removal (creates backups of core tables first)
   - Type `N` to cancel
   - Type `D` for a dry run (just shows what would be removed without actually removing anything)

5. **Verify results**:
   - If you chose to proceed with removal, the script will report which tables were removed
   - You can verify by checking the database size before and after

## Core Tables

The following core tables will NOT be removed and will be backed up before any cleanup operation:

- `bronze`: Raw imported data
- `silver`: Processed and transformed data
- `gold`: Feature-engineered data ready for analysis
- `gold_cluster_processed`: Optimized data used for clustering

## Additional Information

If you encounter any issues with the database being locked or in use, ensure all processes that might be using it are closed. On Windows, you can check which processes are using a file with:

```powershell
Get-Process | Where-Object {$_.Modules.FileName -like "*customer_clustering.db*"}
```

Or use Process Explorer from Sysinternals to search for handles to the database file.
