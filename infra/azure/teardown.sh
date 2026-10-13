#!/usr/bin/env bash
# Deletes the whole resource group (everything this project created). Usage: ./teardown.sh <rg> [--yes]
set -euo pipefail
RG="${1:?resource group}"; YES="${2:-}"
if [ "$YES" != "--yes" ]; then
  echo "This deletes resource group '$RG' and everything in it. Re-run with --yes to confirm."; exit 1
fi
az group delete --name "$RG" --yes --no-wait
echo "Deletion started."
