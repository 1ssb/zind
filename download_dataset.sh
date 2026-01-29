#!/bin/bash

set -e

# Set your ZInD server token here or pass as environment variable
# Get your token from: https://bridgedataoutput.com/login under API ACCESS tab
SERVER_TOKEN="${ZIND_SERVER_TOKEN:-YOUR_TOKEN_HERE}"
OUTPUT_FOLDER="/home/group/cvml-datasets/zind/data/"
LOG_FILE="logs/download_$(date +%Y%m%d_%H%M%S).log"

if [ "$SERVER_TOKEN" = "YOUR_TOKEN_HERE" ]; then
    echo "Error: Please set ZIND_SERVER_TOKEN environment variable or edit this script"
    echo "Get your token from: https://bridgedataoutput.com/login"
    exit 1
fi

python download_data.py -s "$SERVER_TOKEN" -o "$OUTPUT_FOLDER" 2>&1 | tee "$LOG_FILE"

echo "Download complete. Log saved to: $LOG_FILE"
