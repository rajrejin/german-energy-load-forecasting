// Resource group scope. Creates: storage account + container, Log Analytics (capped),
// Container Apps environment and one scale-to-zero API app with managed identity blob access.
// Nothing here is executed by the autonomous build; see docs/AZURE_DEPLOY.md.
targetScope = 'resourceGroup'

@description('Short lowercase prefix, 3-11 letters or digits.')
@minLength(3)
@maxLength(11)
param prefix string

param location string = resourceGroup().location

@description('Image in GitHub Container Registry, for example ghcr.io/<owner>/energy-api:latest')
param image string

@secure()
@description('API key required in the X-API-Key header. Never deploy without one.')
param apiKey string

@description('Daily Log Analytics ingestion cap in GB (0.1 = 100 MB).')
param logCapGb string = '0.1'

var storageName = toLower('${prefix}${uniqueString(resourceGroup().id)}')
var containerName = 'energy'

resource storage 'Microsoft.Storage/storageAccounts@2023-05-01' = {
  name: take(storageName, 24)
  location: location
  sku: { name: 'Standard_LRS' }
  kind: 'StorageV2'
  properties: {
    accessTier: 'Hot'
    allowBlobPublicAccess: false
    minimumTlsVersion: 'TLS1_2'
    supportsHttpsTrafficOnly: true
  }
}

resource blobService 'Microsoft.Storage/storageAccounts/blobServices@2023-05-01' = {
  parent: storage
  name: 'default'
}

resource container 'Microsoft.Storage/storageAccounts/blobServices/containers@2023-05-01' = {
  parent: blobService
  name: containerName
  properties: { publicAccess: 'None' }
}

resource logs 'Microsoft.OperationalInsights/workspaces@2023-09-01' = {
  name: '${prefix}-logs'
  location: location
  properties: {
    sku: { name: 'PerGB2018' }
    retentionInDays: 30
    workspaceCapping: { dailyQuotaGb: json(logCapGb) }
  }
}

resource env 'Microsoft.App/managedEnvironments@2024-03-01' = {
  name: '${prefix}-env'
  location: location
  properties: {
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: {
        customerId: logs.properties.customerId
        sharedKey: logs.listKeys().primarySharedKey
      }
    }
  }
}

resource api 'Microsoft.App/containerApps@2024-03-01' = {
  name: '${prefix}-api'
  location: location
  identity: { type: 'SystemAssigned' }
  properties: {
    managedEnvironmentId: env.id
    configuration: {
      ingress: {
        external: true
        targetPort: 8000
        transport: 'auto'
        allowInsecure: false
      }
      secrets: [
        { name: 'api-key', value: apiKey }
      ]
    }
    template: {
      containers: [
        {
          name: 'api'
          image: image
          resources: { cpu: json('0.25'), memory: '0.5Gi' }
          env: [
            { name: 'ENERGY_STORE', value: 'azure' }
            { name: 'ENERGY_CONTAINER', value: containerName }
            { name: 'ENERGY_AZURE_ACCOUNT_URL', value: storage.properties.primaryEndpoints.blob }
            { name: 'ENERGY_API_KEY', secretRef: 'api-key' }
          ]
          probes: [
            { type: 'Liveness', httpGet: { path: '/v1/health', port: 8000 } }
          ]
        }
      ]
      scale: { minReplicas: 0, maxReplicas: 1 }
    }
  }
}

// Storage Blob Data Contributor for the app's managed identity (read and write forecasts).
var blobContributor = 'ba92f5b4-2d11-453d-a403-e96b0029c9fe'
resource roleAssignment 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(storage.id, api.id, blobContributor)
  scope: storage
  properties: {
    principalId: api.identity.principalId
    principalType: 'ServicePrincipal'
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', blobContributor)
  }
}

output apiUrl string = 'https://${api.properties.configuration.ingress.fqdn}'
output storageAccount string = storage.name
output containerName string = containerName
