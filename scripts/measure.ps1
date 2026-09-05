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
#
# The sampler runs in a background job and the command runs in the FOREGROUND,
# so its output reaches the terminal untouched. The other way round -- command
# in Start-Process, sampler in the loop -- swallows the output, which makes the
# clock check useless in the only way that matters: nobody uses a wrapper that
# hides the numbers it is vouching for.
# WHY THERE IS NO param() BLOCK. There was one -- $Command with
# ValueFromRemainingArguments -- and it ate any flag whose name is a prefix of
# a PowerShell common parameter before the script ever saw it:
#
#   scripts\measure.bat bench\sgemm.exe -s 4096 -k 10 -i 60
#   -> "ambiguous. Possible matches include: -InformationAction -InformationVariable."
#
# PowerShell binds parameters before ValueFromRemainingArguments collects the
# leftovers, so `-i` was resolved against the cmdlet common parameters and the
# run died. Anything starting -i, -e, -o, -v, -d, -w has the same problem, and
# -w is bench/sgemm's warmup flag.
#
# A script with no param() block does not bind anything: every argument lands
# in $args verbatim, dashes and all. The cost is checking arity by hand, which
# is two lines.
$Command = $args
if (-not $Command -or $Command.Count -eq 0) {
    Write-Error "usage: measure.bat <command> [args ...]"
    exit 2
}

$target = if ($env:BMB_CLOCK) { [int]$env:BMB_CLOCK } else { 1200 }
# nvidia-smi reports the nearest achievable step, not the requested number:
# asking for 1200 on this card gives 1185-1192. 2% covers that and still
# catches a lock that has lapsed, since the next boost bin up is 1.6x away.
$tol = 0.02
# Below this the GPU is between kernels or between benchmark cases, and its
# clock says nothing about the clock the kernels ran at.
$busyPct = 40

# Power is sampled beside the clock because the two failure modes below are
# told apart by it: a card sitting UNDER its own lock is power-capped, and
# saying so needs the watts.
#
# Samples above the card's physical ceiling are dropped. nvidia-smi's
# instantaneous clocks.sm is not trustworthy on an idle Blackwell laptop part
# -- five consecutive idle samples on this machine read 0, 2002, 5587, 3630 and
# 1290 MHz, and 5587 is impossible on a card whose maximum is 3090. Filtering
# to busy samples already removes most of it; this removes the rest.
$sampler = Start-Job -ScriptBlock {
    param($busyPct)
    $out = @(); $pw = @()
    $ceil = 0
    $m = (nvidia-smi --query-gpu=clocks.max.sm --format=csv,noheader,nounits 2>$null)
    if ($m -match '^\s*(\d+)') { $ceil = [int]$Matches[1] }
    while ($true) {
        $line = (nvidia-smi --query-gpu=clocks.sm,utilization.gpu,power.draw --format=csv,noheader,nounits 2>$null)
        if ($line -match '^\s*(\d+)\s*,\s*(\d+)\s*,\s*([\d.]+)') {
            $c = [int]$Matches[1]
            if ([int]$Matches[2] -ge $busyPct -and ($ceil -eq 0 -or $c -le $ceil)) {
                $out += $c; $pw += [double]$Matches[3]
            }
        }
        Write-Output $out.Count  # keeps the job's output stream alive
        Set-Content -Path "$env:TEMP\bmb_clock_samples.txt" -Value ($out -join ',')
        Set-Content -Path "$env:TEMP\bmb_power_samples.txt" -Value ($pw -join ',')
        Start-Sleep -Milliseconds 400
    }
} -ArgumentList $busyPct

try {
    $exe = $Command[0]
    $rest = if ($Command.Length -gt 1) { $Command[1..($Command.Length - 1)] } else { @() }
    & $exe @rest
    $rc = $LASTEXITCODE
} finally {
    Stop-Job $sampler -ErrorAction SilentlyContinue
    Remove-Job $sampler -Force -ErrorAction SilentlyContinue
}

$raw = if (Test-Path "$env:TEMP\bmb_clock_samples.txt") { Get-Content "$env:TEMP\bmb_clock_samples.txt" } else { "" }
$samples = @($raw -split ',' | Where-Object { $_ -match '^\d+$' } | ForEach-Object { [int]$_ })

if ($samples.Count -eq 0) {
    Write-Host "[clock] no busy samples -- the command was too short to judge."
    Write-Host "[clock] Timings above are unverified."
    exit $rc
}

$lo = ($samples | Measure-Object -Minimum).Minimum
$hi = ($samples | Measure-Object -Maximum).Maximum
$rawp = if (Test-Path "$env:TEMP\bmb_power_samples.txt") { Get-Content "$env:TEMP\bmb_power_samples.txt" } else { "" }
$power = @($rawp -split ',' | Where-Object { $_ -match '^[\d.]+$' } | ForEach-Object { [double]$_ })
$pmax = if ($power.Count) { ($power | Measure-Object -Maximum).Maximum } else { 0 }

# THREE OUTCOMES, NOT TWO. The old test asked one question -- is every sample
# within 2% of the target -- and answered "NOT a result" to everything else.
# That is right for a lapsed lock and wrong for a card that is holding its lock
# and simply cannot reach it.
#
# The two are far apart and easy to separate. `nvidia-smi -lgc` sets a CEILING:
# if it lapses the card jumps to its boost bin, which on this part is 1.6x the
# pin -- nowhere near it. If instead the card is power-capped it sits just
# UNDER the pin, tightly. Measured here: the GEMM sweeps hold 1185-1192, and
# the training step -- more memory traffic per unit of arithmetic, so more
# watts per clock -- runs 1162-1192 with nvidia-smi reporting SW Power Cap
# active. 1162 is 3.2% low, which the old 2% test called "NOT a result".
#
# What actually matters for a ratio is that the clock is the SAME across the
# runs being compared, so a stable-but-low clock is usable as long as it is
# reported, and a clock that MOVES during one run is not usable even if every
# sample is in range.
$spread = if ($lo -gt 0) { ($hi - $lo) / $lo } else { 1.0 }
$lapsed = $hi -gt $target * 1.15

if ($lapsed) {
    Write-Host ("[clock] UNPINNED: {0}-{1} MHz across {2} busy samples, target {3}." -f $lo, $hi, $samples.Count, $target)
    Write-Host "[clock] The timings above are NOT a result -- re-pin and run again:"
    Write-Host ("[clock]   scripts\gpu_clocks.bat lock {0}" -f $target)
} elseif ($spread -gt 0.03) {
    Write-Host ("[clock] MOVED during the run: {0}-{1} MHz ({2:N1}% spread) across {3} busy samples." -f $lo, $hi, ($spread * 100), $samples.Count)
    Write-Host "[clock] The timings above are not internally comparable -- the clock changed under them."
} elseif ($lo -lt $target * (1 - $tol)) {
    Write-Host ("[clock] HELD BELOW THE PIN: {0}-{1} MHz across {2} busy samples, target {3}{4}." -f $lo, $hi, $samples.Count, $target, $(if ($pmax) { ", peak $pmax W" } else { "" }))
    Write-Host ("[clock] The lock is applied; the card cannot reach it under this load. Timings are")
    Write-Host ("[clock] comparable ONLY to other runs at {0}-{1} MHz -- quote the clock with them." -f $lo, $hi)
} else {
    Write-Host ("[clock] {0}-{1} MHz across {2} busy samples, target {3} -- timings above are comparable." -f $lo, $hi, $samples.Count, $target)
}
exit $rc
