[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Domain,
    [int]$Port = 8001,
    [string]$NgrokPath = (Join-Path $PSScriptRoot "..\.tools\ngrok.exe"),
    [switch]$SkipHealthCheck
)

$ErrorActionPreference = "Stop"

if (!(Test-Path -LiteralPath $NgrokPath)) {
    throw "ngrok.exe not found at $NgrokPath. Download ngrok first or pass -NgrokPath."
}

$publicUrl = if ($Domain -match "^https?://") { $Domain } else { "https://$Domain" }

if (!$SkipHealthCheck) {
    $healthUrl = "http://127.0.0.1:$Port/health"
    Write-Host "Checking AI health at $healthUrl"
    Invoke-RestMethod -Uri $healthUrl -TimeoutSec 10 | Out-Null
}

Write-Host "Opening fixed ngrok URL: $publicUrl -> http://127.0.0.1:$Port"
& $NgrokPath http $Port --url $publicUrl
