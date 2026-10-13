# Azure deployment guide

Status: the files in `infra/azure/`, `azure-pipelines.yml` and `.github/workflows/ci.yml` were **written and statically checked (YAML parses) but never executed**. The Bicep files were **not compiled and not validated against Azure** because the Bicep CLI was not installed in the build environment (no `az` either). Run `az bicep build -f infra/azure/main.bicep` and the `what-if` preview in `deploy.sh` before trusting them. You are responsible for monitoring cost.

## 1. Which Azure account

Offers read from Microsoft pages on 2026-10-12 (they change; check again before signing up):

| Offer | What the page says | Card |
| --- | --- | --- |
| Azure for Students | "$100 credit for Azure services", usable within 12 months, "No credit card required", sign up with a school email, limited to full-time university students, one account per person, renewable yearly while you remain a student; when the credit runs out you choose pay-as-you-go or the subscription is disabled. Source: azure.microsoft.com/free/students | Not required |
| Standard Azure free account | "$200 credit", "up to 30 days to use your credit", plus free monthly amounts of 65+ always-free services; needs a phone number and "a credit card or a debit card (non-prepaid)". Source: azure.microsoft.com/free | Required |

Recommendation: use Azure for Students first (no card). The standard free account is the alternative and does require a card. Free grants change over time.

Container Apps consumption plan grant (pricing page): "The first 180,000 vCPU-seconds, 360,000 GiB-seconds, and 2 million requests per subscription per month are free." A 0.25 vCPU, 0.5 GiB app that scales to zero stays inside that for light use, but costs can occur if limits are exceeded.

## 2. Design (zero cost target)

* Azure Container Apps, consumption plan, `minReplicas 0`, `maxReplicas 1`, 0.25 vCPU and 0.5 GiB, HTTPS ingress, API key required.
* Storage account (Blob, `Standard_LRS`, Hot) for model artifacts, forecasts and published result tables, a few hundred MB at most. The app reads it with its managed identity (Storage Blob Data Contributor), no keys in the app.
* Log Analytics workspace with a 0.1 GB per day ingestion cap.
* Container images live in GitHub Container Registry (free for public repositories), not Azure Container Registry. Make the package public, or give Container Apps registry credentials.
* The Streamlit app runs locally against the cloud API (`ENERGY_API_URL`, `ENERGY_API_KEY`); hosting it on Azure is off by default.
* The refresh job and the weekly briefing need PyTorch and the language model, so they run on your machine (or a scheduled GitHub Actions job) and write to the same blob container (`ENERGY_STORE=azure`).

## 3. Safety rules (do these in order)

1. Create the budget alert first: `az deployment sub create -l westeurope -f infra/azure/budget.bicep -p contactEmail=you@example.com`.
2. Keep `maxReplicas` at 1.
3. Never expose the API without an API key (the Bicep file requires the `apiKey` parameter).
4. Delete the resource group when finished: `infra/azure/teardown.sh <rg> --yes`.
5. Costs could occur if free limits are exceeded; check Cost Management weekly.

## 4. Deploy by hand

```bash
az login
cp infra/azure/parameters.example.json infra/azure/parameters.json   # edit: image, apiKey (keep out of git)
infra/azure/deploy.sh energy-rg westeurope infra/azure/parameters.json          # what-if only
infra/azure/deploy.sh energy-rg westeurope infra/azure/parameters.json --yes    # deploy
```

Then publish artifacts to the new account. Your user needs the Storage Blob Data Contributor role on the account, then:

```bash
export ENERGY_STORE=azure ENERGY_AZURE_ACCOUNT_URL=https://<account>.blob.core.windows.net ENERGY_CONTAINER=energy
energy publish-models
energy refresh
```

Check: `curl -H "X-API-Key: $KEY" https://<app-fqdn>/v1/forecast?zone=DE`.

## 5. Azure DevOps pipeline (`azure-pipelines.yml`)

Stages: `test` (install, ruff, mypy, pytest with coverage published), `build` (Docker build), `push` (GitHub Container Registry), `deploy` (`az containerapp update` to the new tag, on environment `energy-production` with an approval check). Tests run on pull requests; the later stages run on `main`.

One-time setup:

1. Create an Azure DevOps organisation and project (dev.azure.com).
2. Parallel jobs. Rule read on 2026-10-12 (Microsoft Learn, "Configure and pay for parallel jobs"): the Microsoft-hosted free tier must be enabled by linking the organisation to an Azure subscription ("To receive the free grant of parallel jobs, link your Azure DevOps organization to a valid Azure subscription"); for private projects it gives one free job of up to 60 minutes per run and 1,800 minutes per month. The self-hosted free tier is granted automatically: one parallel job, no time limit. Free alternative: register an agent on your own machine and replace `vmImage: ubuntu-24.04` with `name: <your pool>`.
3. Variable group `energy-forecaster` with `GHCR_USER`, `GHCR_TOKEN` (secret, a GitHub token with `write:packages`), `AZ_RESOURCE_GROUP`, `AZ_APP_NAME`.
4. Service connection `energy-azure` (Azure Resource Manager) scoped to the resource group.
5. Environment `energy-production` with an Approvals check.

The GitHub Actions workflow `.github/workflows/ci.yml` mirrors the `test` and `build` stages (lint, type check, tests with coverage, Docker builds).

## 6. What was and was not verified

Verified locally: the same code path (artifact store, refresh, API) against the local filesystem; Azurite results are in HANDOFF.md. Not verified: Bicep compilation, `what-if`, deployment, the managed identity role assignment, both pipelines, GHCR pushes.
