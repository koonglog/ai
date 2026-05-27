[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$AiServiceUrl,
    [string]$HostAddress = "0.0.0.0",
    [int]$Port = 8000
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$backendRoot = Join-Path $repoRoot "backend"

$env:AI_SERVICE_URL = $AiServiceUrl.TrimEnd("/")

Write-Host "Starting backend with AI_SERVICE_URL=$env:AI_SERVICE_URL"

Push-Location $backendRoot
try {
    & python -m uvicorn main:app --host $HostAddress --port $Port
}
finally {
    Pop-Location
}
