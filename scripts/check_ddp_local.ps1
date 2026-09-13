# Native Windows correctness checks. Does not pin clocks or rent hardware.
param([string]$OutputDir = 'bench/logs/ddp_diagnostics_5080')
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
function NormalizeLog([string]$Name) {
    # Tee-Object in Windows PowerShell 5.1 writes UTF-16, unlike PowerShell 7.
    # Keep the saved evidence diffable in Git on either host.
    $path = [IO.Path]::GetFullPath((Join-Path $OutputDir "$Name.txt"))
    [IO.File]::WriteAllText($path, [IO.File]::ReadAllText($path), [Text.UTF8Encoding]::new($false))
}
& scripts\measure.bat bench\test_xent.exe 2>&1 |
    Tee-Object -FilePath (Join-Path $OutputDir 'crossentropy.txt')
if ($LASTEXITCODE -ne 0) { throw 'Cross-entropy regression failed' }
NormalizeLog 'crossentropy'
function Check([string]$Name, [int]$Expected, [string[]]$Options) {
    # Windows PowerShell 5.1 represents native stderr as an ErrorRecord even
    # for expected failures. Judge the exit code; still fail on log I/O errors.
    $ErrorActionPreference = 'Continue'
    & scripts\measure.bat bench\test_ddp_gpt.exe @Options 2>&1 |
        Tee-Object -FilePath (Join-Path $OutputDir "$Name.txt") -ErrorAction Stop
    $actual = $LASTEXITCODE
    $ErrorActionPreference = 'Stop'
    NormalizeLog $Name
    if ($actual -ne $Expected) { throw "$Name exited $actual; expected $Expected" }
    Write-Output "CHECK $Name exit=$actual expected=$Expected"
}
Check 'compact_fp32' 0 @('--ranks', '2')
Check 'compact_tf32' 0 @('--ranks', '3', '--tf32')
Check 'production_fp32' 0 @('--ranks', '2', '--production-shape')
Check 'production_tf32' 0 @('--ranks', '2', '--production-shape', '--tf32')
Check 'poison_fp32' 0 @('--ranks', '2', '--production-shape', '--poison-scratch')
Check 'poison_tf32' 0 @('--ranks', '2', '--production-shape', '--tf32', '--poison-scratch')
Check 'forward_tf32' 0 @('--ranks', '2', '--production-shape', '--tf32', '--forward-only')
Check 'forward_poison_tf32' 0 @('--ranks', '2', '--production-shape', '--tf32', '--forward-only', '--poison-scratch')
$savedNoPeer = $env:DDP_NO_P2P
try {
    $env:DDP_NO_P2P = '1'
    Check 'staged_poison_tf32' 0 @('--ranks', '3', '--tf32', '--poison-scratch')
} finally { $env:DDP_NO_P2P = $savedNoPeer }
Check 'reject_wrong_gradient' 1 @('--ranks', '2', '--tf32', '--inject-gradient-error')
Check 'reject_forward_gradient_injection' 2 @('--forward-only', '--inject-gradient-error')
Write-Output 'ALL LOCAL CORRECTNESS CHECKS PASSED (not a multi-GPU scaling result)'
