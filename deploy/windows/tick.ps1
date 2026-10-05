# Run by the bare-metal-bard-deploy scheduled task every two minutes, as SYSTEM.
# This is what replaces Vercel's deploy-on-push: a push that changes site/ or
# these scripts is built and served within a few minutes, and anything else is
# a fetch and a loopback check.
[CmdletBinding()]
param([string]$Root = 'C:\ProgramData\BareMetalBard')
$ErrorActionPreference = 'Stop'
$log = Join-Path $Root 'logs\poller.log'
try {
    if ((Test-Path -LiteralPath $log) -and (Get-Item -LiteralPath $log).Length -gt 5MB) { Move-Item -LiteralPath $log -Destination ($log + '.previous') -Force }
    & (Join-Path $Root 'ops\deploy.ps1') -Root $Root >> $log 2>&1
} catch {
    Add-Content -LiteralPath $log -Value ((Get-Date).ToString('o') + ' ' + $_.Exception.Message)
    exit 1
}
