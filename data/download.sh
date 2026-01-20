#!/bin/bash

# Download splat files in parallel
URLS=(
    "https://huggingface.co/cakewalk/splat-data/resolve/main/truck.splat"
    "https://huggingface.co/cakewalk/splat-data/resolve/main/plush.splat"
    "https://huggingface.co/cakewalk/splat-data/resolve/main/garden.splat"
    "https://huggingface.co/cakewalk/splat-data/resolve/main/treehill.splat"
    "https://huggingface.co/cakewalk/splat-data/resolve/main/stump.splat"
    "https://huggingface.co/cakewalk/splat-data/resolve/main/bicycle.splat"
    "https://media.reshot.ai/models/nike_next/model.splat"
)

# Configuration
MAX_PARALLEL=3
DOWNLOAD_DIR="./splat_downloads"
LOG_FILE="download.log"

# Create directories
mkdir -p "$DOWNLOAD_DIR"

# Function to download a single file
download_file() {
    local url="$1"
    local filename=$(basename "$url")
    local filepath="$DOWNLOAD_DIR/$filename"
    
    echo "[$(date '+%H:%M:%S')] Starting: $filename" >> "$LOG_FILE"
    
    if curl -L -s -o "$filepath" "$url"; then
        if [ -f "$filepath" ]; then
            local size=$(stat -f%z "$filepath" 2>/dev/null || stat -c%s "$filepath" 2>/dev/null)
            echo "[$(date '+%H:%M:%S')] ✓ Completed: $filename ($((size/1024/1024)) MB)" >> "$LOG_FILE"
            return 0
        fi
    fi
    
    echo "[$(date '+%H:%M:%S')] ✗ Failed: $filename" >> "$LOG_FILE"
    return 1
}

# Export function for parallel execution
export -f download_file
export DOWNLOAD_DIR LOG_FILE

# Clear log file
> "$LOG_FILE"

echo "=============================================="
echo "  Parallel Splat Downloader"
echo "=============================================="
echo "Downloading ${#URLS[@]} files"
echo "Max parallel downloads: $MAX_PARALLEL"
echo "Download directory: $DOWNLOAD_DIR"
echo "Log file: $LOG_FILE"
echo "=============================================="
echo ""

# Create argument list for parallel
printf "%s\n" "${URLS[@]}" | xargs -P "$MAX_PARALLEL" -I {} bash -c 'download_file "{}"'

# Display summary
echo ""
echo "=============================================="
echo "Download Summary"
echo "=============================================="

SUCCESS=0
TOTAL=${#URLS[@]}

for url in "${URLS[@]}"; do
    filename=$(basename "$url")
    filepath="$DOWNLOAD_DIR/$filename"
    if [ -f "$filepath" ]; then
        size=$(stat -f%z "$filepath" 2>/dev/null || stat -c%s "$filepath" 2>/dev/null)
        echo "✓ $filename ($((size/1024/1024)) MB)"
        ((SUCCESS++))
    else
        echo "✗ $filename (missing)"
    fi
done

echo ""
echo "Successfully downloaded: $SUCCESS/$TOTAL files"
echo "Check $LOG_FILE for detailed logs"
echo "=============================================="
