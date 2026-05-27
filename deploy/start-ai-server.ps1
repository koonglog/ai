[CmdletBinding()]
param(
    [string]$HostAddress = "0.0.0.0",
    [int]$Port = 8001,
    [ValidateSet("rule", "hybrid", "lightgbm")]
    [string]$ClassifierBackend = "lightgbm",
    [double]$MinConfidence = 0.6,
    [string]$ModelDir = (Join-Path $PSScriptRoot "..\ai\artifacts\models_pattern_2026_05_11_18_no_accel"),
    [string]$BackendDbPath = (Join-Path $PSScriptRoot "..\_backend_db_expand\kunglog.db"),
    [switch]$Reload
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$resolvedModelDir = (Resolve-Path -LiteralPath $ModelDir).Path

if (Test-Path -LiteralPath $BackendDbPath) {
    $resolvedBackendDbPath = (Resolve-Path -LiteralPath $BackendDbPath).Path
}
else {
    $resolvedBackendDbPath = $BackendDbPath
}

$env:AI_CLASSIFIER_BACKEND = $ClassifierBackend
$env:AI_LGBM_MODEL_DIR = $resolvedModelDir
$env:AI_LGBM_MIN_CONFIDENCE = [string]$MinConfidence
$env:AI_LGBM_SHADOW_MODE = "false"
$env:BACKEND_DB_PATH = $resolvedBackendDbPath
if ([string]::IsNullOrWhiteSpace($env:ENABLE_OPENAI)) {
    $env:ENABLE_OPENAI = "false"
}

$uvicornArgs = @(
    "-m", "uvicorn",
    "ai.dashboard_api:app",
    "--host", $HostAddress,
    "--port", [string]$Port
)

if ($Reload) {
    $uvicornArgs += "--reload"
}

Write-Host "Starting KoongLog AI on http://$HostAddress`:$Port"
Write-Host "Model dir: $resolvedModelDir"
Write-Host "Backend DB: $resolvedBackendDbPath"

Push-Location $repoRoot
try {
    & python @uvicornArgs
}
finally {
    Pop-Location
}
