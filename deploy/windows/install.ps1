# One-time setup of the runtime root on the home server. Run as Administrator.
#
#   install.ps1 -CaddyExe <path> -MainCaddyfile <path>
#   install.ps1 -CaddyExe <path> -MainCaddyfile <path> -EnableAutoDeploy
#
# -CaddyExe and -MainCaddyfile are the binary and config the host's running
# Caddy already uses (see its command line); this site adds itself to that
# server rather than starting another one, since only one process can own 443.
#
# First rollout, in order:
#   install.ps1 ...                      runtime root, repo clone, tool paths
#   ops\deploy.ps1 -PrepareOnly          build a release; traffic unchanged
#   (point sgemm.ethanyanxu.com at this connection)
#   ops\activate.ps1 -WaitSeconds 180    switch Caddy; it obtains the certificate
#   install.ps1 ... -EnableAutoDeploy    poll origin/main every two minutes
#
# Activating only once DNS points here matters: Caddy starts requesting a
# certificate the moment the site block loads, and each failed Let's Encrypt
# validation counts against a per-hostname rate limit and lengthens Caddy's
# retry backoff.
[CmdletBinding()]
param([string]$Root = 'C:\ProgramData\BareMetalBard', [Parameter(Mandatory)][string]$CaddyExe,
    [Parameter(Mandatory)][string]$MainCaddyfile, [switch]$EnableAutoDeploy)
. (Join-Path $PSScriptRoot 'common.ps1')
Assert-Administrator
foreach ($name in 'logs', 'releases', 'static', 'ops') { New-Item -ItemType Directory -Path (Join-Path $Root $name) -Force | Out-Null }
& icacls.exe $Root '/inheritance:r' '/grant:r' '*S-1-5-18:(OI)(CI)F' '*S-1-5-32-544:(OI)(CI)F' | Out-Null
if ($LASTEXITCODE -ne 0) { throw 'Could not protect the runtime directory.' }

$config = [pscustomobject]@{
    node = (Get-Command node.exe).Source; npm = (Get-Command npm.cmd).Source; git = (Get-Command git.exe).Source
    caddy = (Resolve-Path -LiteralPath $CaddyExe).Path; mainCaddyfile = (Resolve-Path -LiteralPath $MainCaddyfile).Path
}
Write-Json (Join-Path $Root 'server.json') $config

$repo = Join-Path $Root 'repo'
if (-not (Test-Path -LiteralPath (Join-Path $repo '.git'))) {
    Invoke-Tool $config.git @('clone', '--branch', 'main', $script:RepoUrl, $repo) (Join-Path $Root 'logs\install.log')
}
$ops = Join-Path $Root 'ops'
if ([IO.Path]::GetFullPath($PSScriptRoot).TrimEnd('\') -ne [IO.Path]::GetFullPath($ops).TrimEnd('\')) {
    Copy-Item -Path (Join-Path $PSScriptRoot '*.ps1') -Destination $ops -Force
}

if ($EnableAutoDeploy) {
    if (-not (Read-Json (Join-Path $Root 'active.json'))) { throw 'Activate and verify a release before enabling automatic deployment.' }
    $action = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument ('-NoProfile -NonInteractive -ExecutionPolicy Bypass -File "{0}" -Root "{1}"' -f (Join-Path $ops 'tick.ps1'), $Root)
    $triggers = @((New-ScheduledTaskTrigger -AtStartup), (New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(2) -RepetitionInterval (New-TimeSpan -Minutes 2)))
    $settings = New-ScheduledTaskSettingsSet -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Minutes 25) `
        -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
    Register-ScheduledTask -TaskName 'bare-metal-bard-deploy' -Action $action -Trigger $triggers -Settings $settings -User 'SYSTEM' -RunLevel Highest -Force | Out-Null
    if (-not (Get-ScheduledTask -TaskName 'bare-metal-bard-deploy' -ErrorAction SilentlyContinue)) { throw 'Task bare-metal-bard-deploy is not registered.' }
    Write-Output 'Registered bare-metal-bard-deploy (every two minutes, as SYSTEM).'
}
Write-Output "Runtime installed at $Root."
