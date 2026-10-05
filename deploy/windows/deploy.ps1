# Build the site at a commit and serve it.
#
#   deploy.ps1                 build origin/main, switch Caddy, verify, keep previous for rollback
#   deploy.ps1 -PrepareOnly    build and stage it as prepared.json; traffic unchanged
#
# Run as Administrator (tick.ps1 runs it as SYSTEM every two minutes).
[CmdletBinding()]
param([string]$Root = 'C:\ProgramData\BareMetalBard', [string]$Ref = 'origin/main', [switch]$PrepareOnly)
. (Join-Path $PSScriptRoot 'common.ps1')
Assert-Administrator
$lock = $null
try {
    $lock = [IO.File]::Open((Join-Path $Root 'deploy.lock'), 'OpenOrCreate', 'ReadWrite', 'None')
    $config = Read-Json (Join-Path $Root 'server.json')
    if (-not $config) { throw 'No server.json -- run install.ps1 first.' }
    $repo = Join-Path $Root 'repo'
    $log = Join-Path $Root 'logs\deploy.log'
    $env:NEXT_TELEMETRY_DISABLED = '1'
    $env:GIT_TERMINAL_PROMPT = '0'
    $gitOptions = @('-c', ('safe.directory=' + ($repo -replace '\\', '/')), '-C', $repo)

    Invoke-Tool $config.git ($gitOptions + @('fetch', 'origin', 'main')) $log
    $commit = (& $config.git @gitOptions log -1 --format=%H $Ref -- $script:DeployPaths | Out-String).Trim()
    if ($LASTEXITCODE -ne 0 -or $commit -notmatch '^[a-f0-9]{40}$') { throw 'Could not resolve the last site commit.' }

    $active = Read-Json (Join-Path $Root 'active.json')
    if (-not $PrepareOnly -and $active -and $active.commit -eq $commit -and (Test-ReleaseFiles $active)) {
        # Nothing new to build -- but "the files are there" is not "it is being
        # served". finprint's setup.ps1 regenerates the host's main Caddyfile
        # from scratch, dropping every site's import line, and nothing would
        # say so. So check what Caddy actually answers, and re-apply if the
        # import has gone or the answer is wrong.
        $main = [IO.File]::ReadAllText($config.mainCaddyfile)
        if ($main.Contains("# BEGIN $($script:Marker) (managed)") -and (Test-Served $commit 10)) {
            Write-Output "Already serving $commit"; return
        }
        Write-Output "Release $commit is active but not being served; re-applying the Caddy config."
        Switch-Caddy $Root $config $active
        if (-not (Test-Served $commit 30)) { throw "Re-applied $commit, but Caddy still does not serve it." }
        Write-Output "Re-applied $commit."; return
    }
    $dirty = & $config.git @gitOptions status --porcelain
    if ($LASTEXITCODE -ne 0 -or $dirty) { throw 'Deployment checkout is not clean; refusing to overwrite it.' }
    Invoke-Tool $config.git ($gitOptions + @('checkout', '--detach', $commit)) $log

    # Build. The export lands in site\out and is removed first, so a failed
    # build can never leave the previous export looking like this commit's.
    $site = Join-Path $repo 'site'
    $out = Join-Path $site 'out'
    if (Test-Path -LiteralPath $out) { Remove-Item -LiteralPath $out -Recurse -Force }
    Push-Location $site
    try {
        $env:NODE_ENV = 'development'
        Invoke-Tool $config.npm @('ci', '--include=dev', '--no-audit', '--no-fund') $log
        $env:NODE_ENV = 'production'
        Invoke-Tool $config.npm @('run', 'build') $log
    } finally { Pop-Location }
    if (-not (Test-Path -LiteralPath (Join-Path $out 'index.html'))) { throw 'Build produced no index.html; traffic is unchanged.' }

    $releaseId = $commit.Substring(0, 12) + '-' + (Get-Date -Format 'yyyyMMddHHmmss')
    $release = Join-Path $Root ('releases\' + $releaseId)
    $www = Join-Path $release 'www'
    New-Item -ItemType Directory -Path $www, (Join-Path $release 'ops') -Force | Out-Null
    Copy-Item -Path (Join-Path $out '*') -Destination $www -Recurse -Force
    [IO.File]::WriteAllText((Join-Path $www 'version.txt'), $commit + "`n", (New-Object Text.UTF8Encoding $false))
    Copy-Item -Path (Join-Path $repo 'deploy\windows\*.ps1') -Destination (Join-Path $release 'ops') -Force
    $sharedStatic = Join-Path $Root 'static'
    New-Item -ItemType Directory -Path $sharedStatic -Force | Out-Null
    Copy-Item -Path (Join-Path $out '_next\static\*') -Destination $sharedStatic -Recurse -Force

    $next = [pscustomobject]@{ commit = $commit; release = $release; createdAt = (Get-Date).ToUniversalTime().ToString('o') }
    Write-Json (Join-Path $release 'release.json') $next
    if (-not (Test-ReleaseFiles $next)) { throw 'Release files failed verification; traffic is unchanged.' }

    if ($PrepareOnly) {
        Write-Json (Join-Path $Root 'prepared.json') $next
        Write-Output "Prepared $commit at $release. Activate with activate.ps1."
        return
    }

    Switch-Caddy $Root $config $next
    # A first activation waits longer: Caddy has to obtain the certificate.
    if (-not (Test-Served $commit $(if ($active) { 30 } else { 120 }))) {
        if ($active -and (Test-ReleaseFiles $active)) {
            Switch-Caddy $Root $config $active
            throw "Caddy did not serve $commit after the switch; rolled back to $($active.commit)."
        }
        throw "Caddy did not serve $commit over HTTPS on loopback. With no earlier release to fall back to it stays in place -- check that $($script:SiteHost) resolves to this connection and logs\caddy-reload.log."
    }
    if ($active) { Write-Json (Join-Path $Root 'previous.json') $active }
    Write-Json (Join-Path $Root 'active.json') $next
    Copy-Item -Path (Join-Path $release 'ops\*.ps1') -Destination (Join-Path $Root 'ops') -Force
    Remove-OldReleases $Root
    Write-Output "Serving $commit."
} finally {
    if ($lock) { $lock.Dispose() }
}
