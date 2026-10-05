# Point Caddy at a saved release.
#
#   activate.ps1             serve prepared.json (from deploy.ps1 -PrepareOnly)
#   activate.ps1 -Rollback   serve previous.json, the release before the active one
#
# Verifies over HTTPS on loopback before recording the switch; on failure the
# earlier release is restored.
[CmdletBinding()]
param([string]$Root = 'C:\ProgramData\BareMetalBard', [switch]$Rollback, [int]$WaitSeconds = 30)
. (Join-Path $PSScriptRoot 'common.ps1')
Assert-Administrator
$lock = [IO.File]::Open((Join-Path $Root 'deploy.lock'), 'OpenOrCreate', 'ReadWrite', 'None')
try {
    $config = Read-Json (Join-Path $Root 'server.json')
    $old = Read-Json (Join-Path $Root 'active.json')
    $targetFile = if ($Rollback) { 'previous.json' } else { 'prepared.json' }
    $target = Read-Json (Join-Path $Root $targetFile)
    if (-not $target) { throw "No $targetFile to activate." }
    $null = Assert-UnderRoot $target.release (Join-Path $Root 'releases')
    if (-not (Test-ReleaseFiles $target)) { throw 'Saved release is incomplete; traffic is unchanged.' }

    Switch-Caddy $Root $config $target
    if (-not (Test-Served $target.commit $WaitSeconds)) {
        if ($old -and (Test-ReleaseFiles $old)) {
            Switch-Caddy $Root $config $old
            throw "Caddy did not serve $($target.commit); restored $($old.commit)."
        }
        throw "Caddy did not serve $($target.commit) over HTTPS on loopback, and there is no earlier release to restore. Check DNS for $($script:SiteHost) and logs\caddy-reload.log."
    }
    if ($old -and $old.release -ne $target.release) { Write-Json (Join-Path $Root 'previous.json') $old }
    Write-Json (Join-Path $Root 'active.json') $target
    if (-not $Rollback) { Remove-Item -LiteralPath (Join-Path $Root 'prepared.json') -Force -ErrorAction SilentlyContinue }
    Copy-Item -Path (Join-Path $target.release 'ops\*.ps1') -Destination (Join-Path $Root 'ops') -Force
    Write-Output ('Activated ' + $target.commit)
} finally { $lock.Dispose() }
