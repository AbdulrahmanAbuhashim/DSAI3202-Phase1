#!/bin/bash
set -e

# Upload a file to the project's Azure raw container.
# Usage: bash scripts/upload_to_azure.sh <local-file> <blob-name>

ACCOUNT="amazonlake60301190"
CONTAINER="raw"

if [ "$#" -ne 2 ]; then
    echo "Usage: bash scripts/upload_to_azure.sh <local-file> <blob-name>"
    exit 1
fi

FILE="$1"
BLOB="$2"

if [ ! -f "$FILE" ]; then
    echo "File not found: $FILE"
    exit 1
fi

# Do not overwrite existing Azure files.
EXISTS=$(az storage blob exists \
    --account-name "$ACCOUNT" \
    --container-name "$CONTAINER" \
    --name "$BLOB" \
    --auth-mode login \
    --query exists -o tsv)

if [ "$EXISTS" = "true" ]; then
    echo "File already exists in Azure. Upload cancelled."
    exit 1
fi

az storage blob upload \
    --account-name "$ACCOUNT" \
    --container-name "$CONTAINER" \
    --name "$BLOB" \
    --file "$FILE" \
    --auth-mode login \
    --overwrite false \
    --output none

echo "Upload successful: $BLOB"
