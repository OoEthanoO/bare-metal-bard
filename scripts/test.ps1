# Run the whole test suite, then fail if training reaches a kernel no test does.
#
#   scripts\test.bat              build the test targets, then run
#   scripts\test.bat --no-build   run what is already in bench\
#
# The suite itself is scripts\test_suite.txt, shared with scripts/test.sh.
#
# WHY THE COVERAGE GATE. Twice now a kernel path shipped with the tested path
# and the trained path quietly apart. The layernorm fast path was selected by
# `if (C == 384)` while the gradient check ran at C=128, so the kernel the model
# trained with was never checked. Then the GEMM's fused epilogues were verified
# in the fp32 implementation while every --tf32 run used the tensor-core one.
# Both were found by BMB_COVER -- by hand, once. When the compact tile landed
# afterwards it added three new training branches and nobody re-ran the diff.
#
# A check that has to be remembered is a check that gets skipped. This one
# runs every time: each suite line is run with BMB_COVER=1, and every dispatch
# branch a `train` line reaches must also be reached by some `test` line, or
# the run fails and names the branch.
#
# No param() block, for the reason measure.ps1 gives: PowerShell would bind a
# leading dash before the script saw it.
$noBuild = $args -contains '--no-build'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$suite = @()
foreach ($line in Get-Content (Join-Path $PSScriptRoot 'test_suite.txt')) {
    $t = $line.Trim()
    if ($t -eq '' -or $t.StartsWith('#')) { continue }
    $f = $t -split '\s+', 3
    if ($f.Count -lt 3 -or ($f[1] -ne 'test' -and $f[1] -ne 'train')) {
        Write-Host "test_suite.txt: cannot parse: $line"; exit 2
    }
    $parts = $f[2] -split '\s+', 2
    $rest = if ($parts.Count -gt 1) { $parts[1] } else { '' }
    $suite += [pscustomobject]@{ name = $f[0]; kind = $f[1]; exe = $parts[0]; args = $rest }
}

if (-not $noBuild) {
    $targets = ($suite | ForEach-Object { $_.exe } | Sort-Object -Unique) -join ' '
    Write-Host "[build] $targets"
    cmd /c "scripts\build.bat $targets" | Out-Null
    if ($LASTEXITCODE -ne 0) { Write-Host "[build] FAILED -- run scripts\build.bat to see why"; exit 1 }
}
if (($suite | Where-Object kind -eq 'train') -and -not (Test-Path 'data\input.txt')) {
    Write-Host "data\input.txt is missing; the train lines need it (scripts/get_data.sh)"; exit 2
}

$out = Join-Path $env:TEMP 'bmb_test'
if (Test-Path $out) { Remove-Item -Recurse -Force $out }
New-Item -ItemType Directory -Path $out | Out-Null

$failed = @()
$covered = @{}; $trained = @{}
foreach ($s in $suite) {
    $log = Join-Path $out "$($s.name).log"; $cov = Join-Path $out "$($s.name).cov"
    $sw = [Diagnostics.Stopwatch]::StartNew()
    cmd /c "call scripts\env.bat >nul && set BMB_COVER=1 && bench\$($s.exe).exe $($s.args) 1> `"$log`" 2> `"$cov`""
    $rc = $LASTEXITCODE
    $secs = $sw.Elapsed.TotalSeconds
    $branches = @(Get-Content $cov -ErrorAction SilentlyContinue | Where-Object { $_ -like 'COVER *' })
    $into = if ($s.kind -eq 'test') { $covered } else { $trained }
    foreach ($b in $branches) { $into[$b.Substring(6)] = $s.name }
    $status = if ($rc -eq 0) { 'ok' } else { "FAILED (exit $rc)" }
    if ($rc -ne 0) { $failed += $s.name }
    Write-Host ("  {0,-16} {1,-5} {2,6:N1}s  {3,3} branches  {4}" -f $s.name, $s.kind, $secs, $branches.Count, $status)
}

$uncovered = @($trained.Keys | Where-Object { -not $covered.ContainsKey($_) } | Sort-Object)
Write-Host ""
Write-Host ("coverage: training reaches {0} dispatch branches; the tests reach {1}" -f $trained.Count, $covered.Count)
if ($uncovered.Count) {
    Write-Host "REACHED BY TRAINING AND BY NO TEST:"
    foreach ($u in $uncovered) { Write-Host ("  {0}    (first seen in {1})" -f $u, $trained[$u]) }
}
if ($failed.Count) { Write-Host ("FAILED: " + ($failed -join ', ') + "   logs in $out") }
if ($failed.Count -or $uncovered.Count) { exit 1 }
Write-Host "all tests passed, and every branch training reaches is tested"
exit 0
