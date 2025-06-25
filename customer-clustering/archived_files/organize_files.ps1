# PowerShell script to move older files to archived_files directory
# Identify files from older runs by timestamp
$latestRun = "20250625_125"  # Files from runs after 12:50 PM on June 25, 2025 are considered latest
$currentDir = "c:\Code\Local Code\URBN\RETRO_hack_25\customer-clustering"
$archiveDir = "c:\Code\Local Code\URBN\RETRO_hack_25\customer-clustering\archived_files"

# Create a function to determine if a file is part of the latest run
function Is-LatestRunFile {
    param($fileName)
    
    # Keep core files that should never be archived
    $coreFiles = @(
        "create_clusters.ipynb",
        "create_features.py",
        "features.py",
        "db.py",
        "clustering_notebook.py",
        "customer_clustering.db",
        "cluster_preprocessing.py"
    )
    
    if ($coreFiles -contains $fileName) {
        return $true
    }
    
    # Check if the file has today's latest timestamp (after 12:50 PM)
    if ($fileName -match $latestRun) {
        return $true
    }
    
    # Special handling for results directory
    if ($fileName -eq "results") {
        return $true
    }
    
    # Keep Python script files that are part of the codebase
    if ($fileName -match "\.py$" -and -not $fileName.StartsWith("check_")) {
        return $true  
    }
    
    # Keep documentation files
    if ($fileName -match "\.(md|ipynb)$") {
        return $true
    }
    
    return $false
}

# Get all items in the directory
$items = Get-ChildItem -Path $currentDir

# Move older files to archive
foreach ($item in $items) {
    if (-not (Is-LatestRunFile $item.Name)) {
        # Skip directories for now except for old result directories
        if ($item.PSIsContainer -and $item.Name -ne "results" -and $item.Name -ne "archived_files" -and $item.Name -ne "__pycache__") {
            Write-Host "Skipping directory: $($item.Name)"
            continue
        }
        
        # Move the item to archive
        Write-Host "Moving to archive: $($item.Name)"
        try {
            Move-Item -Path "$currentDir\$($item.Name)" -Destination "$archiveDir\$($item.Name)" -Force -ErrorAction Stop
            Write-Host "  Success"
        }
        catch {
            Write-Host "  Failed: $_"
        }
    }
    else {
        Write-Host "Keeping: $($item.Name)"
    }
}

# Also process the root directory
$rootDir = "c:\Code\Local Code\URBN\RETRO_hack_25"
$rootArchiveDir = "c:\Code\Local Code\URBN\RETRO_hack_25\archived_files"

# Core files in the root directory that should be kept
$rootCoreFiles = @(
    "customer_clustering.db",
    "customer_features.db",
    "requirements.txt",
    "features.md",
    ".gitignore"
)

# Get all files in the root directory
$rootItems = Get-ChildItem -Path $rootDir -File

# Move older files to archive
foreach ($item in $rootItems) {
    # Skip core files and directories
    if (($rootCoreFiles -contains $item.Name) -or 
        ($item.Name -match $latestRun) -or 
        ($item.Name -like "*.py")) {
        Write-Host "Keeping root file: $($item.Name)"
        continue
    }
    
    # Check for temporary Excel files
    if ($item.Name -like "~$*") {
        Write-Host "Skipping temporary Excel file: $($item.Name)"
        continue
    }
    
    # Move the item to archive
    Write-Host "Moving to root archive: $($item.Name)"
    try {
        Move-Item -Path "$rootDir\$($item.Name)" -Destination "$rootArchiveDir\$($item.Name)" -Force -ErrorAction Stop
        Write-Host "  Success"
    }
    catch {
        Write-Host "  Failed: $_"
    }
}

Write-Host "File organization complete."
