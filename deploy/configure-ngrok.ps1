[CmdletBinding()]
param(
    [string]$NgrokPath = (Join-Path $PSScriptRoot "..\.tools\ngrok.exe")
)

$ErrorActionPreference = "Stop"

if (!(Test-Path -LiteralPath $NgrokPath)) {
    throw "ngrok.exe not found at $NgrokPath. Download ngrok first or pass -NgrokPath."
}

$token = Read-Host "Paste ngrok authtoken" -AsSecureString
$bstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($token)

try {
    $plainToken = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($bstr)
    & $NgrokPath config add-authtoken $plainToken
}
finally {
    if ($bstr -ne [IntPtr]::Zero) {
        [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr)
    }
}
