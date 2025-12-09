#!/bin/bash
# Sync data from S3 bucket to local

set -e

BUCKET="s3://amzn-s3-proteinbucket"
LOCAL_DIR="$(dirname "$0")/.."

echo "=== Syncing data from S3 ==="
echo "Bucket: $BUCKET"
echo "Local:  $LOCAL_DIR"

# Sync data_raw directory
echo ""
echo "Syncing data_raw/..."
aws s3 sync "$BUCKET/data_raw/" "$LOCAL_DIR/data_raw/" --no-progress

# Sync data_processed if exists
echo ""
echo "Syncing data_processed/..."
aws s3 sync "$BUCKET/data_processed/" "$LOCAL_DIR/data_processed/" --no-progress 2>/dev/null || echo "No data_processed/ found in S3"

echo ""
echo "=== Sync complete ==="
ls -la "$LOCAL_DIR/data_raw/" 2>/dev/null | head -10 || echo "data_raw/ not found"
