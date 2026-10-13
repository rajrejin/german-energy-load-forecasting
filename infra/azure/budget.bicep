// Subscription-scope cost budget with alerts at a very low threshold. Create this FIRST.
// az deployment sub create -l westeurope -f budget.bicep -p contactEmail=you@example.com
targetScope = 'subscription'

param contactEmail string
@description('Monthly budget in the billing currency.')
param amount int = 5
param startDate string = '${utcNow('yyyy-MM')}-01'

resource budget 'Microsoft.Consumption/budgets@2023-05-01' = {
  name: 'energy-forecaster-budget'
  properties: {
    category: 'Cost'
    amount: amount
    timeGrain: 'Monthly'
    timePeriod: { startDate: startDate }
    notifications: {
      actual50: {
        enabled: true
        operator: 'GreaterThan'
        threshold: 50
        contactEmails: [contactEmail]
        thresholdType: 'Actual'
      }
      actual100: {
        enabled: true
        operator: 'GreaterThan'
        threshold: 100
        contactEmails: [contactEmail]
        thresholdType: 'Actual'
      }
    }
  }
}
