#!/usr/bin/env bash
# Preview, then deploy only with --yes. Usage:
#   ./deploy.sh <resource-group> <location> <parameters.json> [--yes]
set -euo pipefail
RG="${1:?resource group}"; LOC="${2:?location, e.g. westeurope}"; PARAMS="${3:?parameters file}"; YES="${4:-}"
HERE="$(cd "$(dirname "$0")" && pwd)"

if ! az account show >/dev/null 2>&1; then
  echo "Not logged in. Run: az login" >&2; exit 1
fi
echo "Subscription: $(az account show --query name -o tsv)"

echo "== what-if preview =="
az group create --name "$RG" --location "$LOC" -o none   # an empty resource group costs nothing
az deployment group what-if -g "$RG" -f "$HERE/main.bicep" -p "@$PARAMS"

if [ "$YES" != "--yes" ]; then
  echo
  echo "Preview only. Set the budget alert first (see docs/AZURE_DEPLOY.md), then re-run with --yes."
  exit 0
fi
az deployment group create -g "$RG" -f "$HERE/main.bicep" -p "@$PARAMS" \
  --query "properties.outputs" -o json
