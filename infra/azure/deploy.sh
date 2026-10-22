#!/usr/bin/env bash
# Preview with what-if, then deploy only with --yes. Nothing is created without --yes.
# Usage:
#   ./deploy.sh <resource-group> <location> <parameters.json> [--yes]
set -euo pipefail
RG="${1:?resource group}"; LOC="${2:?location, e.g. westeurope}"; PARAMS="${3:?parameters file}"; YES="${4:-}"
HERE="$(cd "$(dirname "$0")" && pwd)"

if ! az account show >/dev/null 2>&1; then
  echo "Not logged in. Run: az login" >&2; exit 1
fi
echo "Subscription: $(az account show --query name -o tsv)"
az bicep build -f "$HERE/main.bicep" --stdout >/dev/null   # fails early on Bicep errors

if [ "$(az group exists --name "$RG")" = "true" ]; then
  echo "== what-if preview =="
  az deployment group what-if -g "$RG" -f "$HERE/main.bicep" -p "@$PARAMS"
else
  echo "Resource group '$RG' does not exist yet, so a what-if preview is not possible."
  echo "With --yes this script creates it (an empty resource group costs nothing) and deploys."
fi

if [ "$YES" != "--yes" ]; then
  echo
  echo "Preview only. Set the budget alert first (see docs/AZURE_DEPLOY.md), then re-run with --yes."
  exit 0
fi
az group create --name "$RG" --location "$LOC" -o none
az deployment group create -g "$RG" -f "$HERE/main.bicep" -p "@$PARAMS" \
  --query "properties.outputs" -o json
