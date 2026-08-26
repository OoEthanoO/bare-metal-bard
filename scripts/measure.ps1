# Run a command and watch the SM clock the whole time it runs.
#
#   scripts\measure.bat <command> [args ...]
#
# WHY THIS EXISTS. `nvidia-smi -lgc` does not stay applied. Nsight Compute
# resets the application clock when it detaches, and the lock has also been
# observed to lapse on its own across a driver power event. Both times it
# happened, the result was a number that looked like a large speedup and was a
# clock ratio: 59.9 -> 38 ms is 1.58x, and so is 1900/1200.
#
# WHY IT NO LONGER CHECKS BEFORE AND AFTER. It used to read the clock on both
# sides of the command and demand both equal the target. That worked on the
# Ada laptop, whose locked clock was also its idle clock. It does not work
# here: this card idles at 0-600 MHz with the lock perfectly applied, and was
# also seen at 2032 MHz a second after a run finished. Both readings would have
# condemned a run that was pinned at 1192 MHz for every microsecond of the
# measurement.
#
# So the clock is sampled WHILE the command runs, and only samples taken with
# the GPU actually busy are judged. That is the population the timing came
# from, and it is the only one worth checking.
param(
    [Parameter(Mandatory = $true, ValueFromRemainingArguments = $true)]
    [string[]]$Command
)

$target = if ($env:BMB_CLOCK) { [int]$env:BMB_CLOCK } else { 1200 }
# nvidia-smi reports the nearest achievable step, not the requested number:
# asking for 1200 on this card gives 1185-1192. 2% covers that and still
# catches a lock that has lapsed, since the next boost bin up is 1.6x away.
$tol = 0.02
# Below this the GPU is between kernels or between benchmark cases, and its
# clock says nothing about the clock the kernels ran at.
$busyPct = 40

$exe = $Command[0]
$rest = if ($Command.Length -gt 1) { $Command[1..($Command.Length - 1)] } else { @() }

$proc = Start-Process -FilePath $exe -ArgumentList $rest -NoNewWindow -PassThru
$samples = [System.Collections.Generic.List[int]]::new()
while (-not $proc.HasExited) {
    Start-Sleep -Milliseconds 400
    $line = (nvidia-smi --query-gpu=clocks.sm,utilization.gpu --format=csv,noheader,nounits 2>$null)
    if ($line -match '^\s*(\d+)\s*,\s*(\d+)') {
        if ([int]$Matches[2] -ge $busyPct) { $samples.Add([int]$Matches[1]) }
    }
}
$proc.WaitForExit()
$rc = $proc.ExitCode

if ($samples.Count -eq 0) {
    Write-Host "[clock] no busy samples -- the command was too short to judge."
    Write-Host "[clock] Timings above are unverified."
    exit $rc
}

$lo = ($samples | Measure-Object -Minimum).Minimum
$hi = ($samples | Measure-Object -Maximum).Maximum
$off = $samples | Where-Object { [Math]::Abs($_ - $target) -gt $target * $tol }
if ($off.Count -eq 0) {
    Write-Host ("[clock] {0}-{1} MHz across {2} busy samples, target {3} -- timings above are comparable." -f $lo, $hi, $samples.Count, $target)
} else {
    Write-Host ("[clock] UNPINNED: {0} of {1} busy samples off target {2} (range {3}-{4} MHz)." -f $off.Count, $samples.Count, $target, $lo, $hi)
    Write-Host "[clock] The timings above are NOT a result -- re-pin and run again:"
    Write-Host ("[clock]   scripts\gpu_clocks.bat lock {0}" -f $target)
}
exit $rc
