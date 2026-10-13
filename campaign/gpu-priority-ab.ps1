# campaign/gpu-priority-ab.ps1 — DOES THE ENGINE'S GPU SCHEDULING BAND SET THE WINDOW COST?
#
# THE OBSERVATION THIS TESTS (F-154): three states were measured today with the SAME engine
# binary and config, and the difference tracks the BOOT, not the recipe:
#     13.0 ms/window, 141-184 t/s   early afternoon   "healthy"
#     28-30 ms/window, 60-66 t/s    profiled boot     stable across three probes
#     38.7 -> 75.2 ms/window        standing boot     grew within two minutes
# and the profiled boot disabled four fusions, so it should have been the SLOWER one.  On the
# failed boot the card was 99% busy, at full SM clock, no throttle flags, gen5 x16, at 158-200 W
# of a 575 W budget, with the memory controllers only 5-17% busy: the GPU was running the
# engine's kernels, they just took 3-6x longer.
#
# That is the signature of WDDM scheduling: Windows maps a process's CPU priority class onto its
# GPU scheduling priority band, and a lower band is preempted at packet boundaries.  A decode
# window is ~1000 small kernels, so packet-level preemption is paid ~1000 times per window.
#
# The arm is a one-line, instantly reversible, no-code change: set the ENGINE process's
# PriorityClass and re-probe IN THE SAME BOOT.  Interleaved control/variant/control so drift
# cannot be mistaken for the effect.
#
# Usage: pwsh -File campaign\gpu-priority-ab.ps1 -Port 8081 -MaxTok 300
param(
  [int]$Port = 8081,
  [int]$MaxTok = 300,
  [string]$LogPath = 'E:\LLM\Engine\strata-custom-2\strata-iq3_s.log',
  [string[]]$Arms = @('Normal', 'High', 'RealTime', 'Normal'),
  # the arms that also carry a PAIR probe: M1 alone cannot gate an adoption (six-metric ultimatum)
  [string[]]$ConcArms = @(),
  # the real workload class: the sustained 2000-token default-reasoning generation the bar is stated on
  [switch]$RealWorkload
)

$base = "http://127.0.0.1:$Port/v1/chat/completions"
# pwsh -File passes "a,b,c" as ONE string: split it so the documented -Arms a,b,c usage works
$Arms = @($Arms | ForEach-Object { $_ -split ',' } | Where-Object { $_ })
$ConcArms = @($ConcArms | ForEach-Object { $_ -split ',' } | Where-Object { $_ })
$prompt = if ($RealWorkload) {
  'Write a comprehensive technical analysis of modern GPU memory hierarchies and their impact on LLM inference performance. Cover HBM bandwidth, L2 cache behavior, PCIe transfer overhead, and pinned memory tradeoffs, with concrete numbers and examples throughout.'
} else {
  'Explain the tradeoffs of expert caching in a mixture-of-experts inference engine, with concrete numbers.'
}
$concPrompt = 'Explain in detail how mixture-of-experts routing works in modern LLMs, including load balancing, capacity factors, and the tradeoffs versus dense models.'

function ConcRun([int]$n, [int]$maxTok) {
  $jobs = 1..$n | ForEach-Object {
    Start-Job -ScriptBlock {
      param($u, $p, $mt)
      $b = @{ messages = @(@{ role = 'user'; content = $p }); max_tokens = $mt; temperature = 0.7 } | ConvertTo-Json -Depth 4
      $t0 = Get-Date
      $r = Invoke-RestMethod -Uri $u -Method Post -ContentType 'application/json' -Body $b -TimeoutSec 900
      [pscustomobject]@{ tok = $r.usage.completion_tokens; sec = ((Get-Date) - $t0).TotalSeconds }
    } -ArgumentList $base, $concPrompt, $maxTok
  }
  $res = $jobs | Wait-Job | Receive-Job; $jobs | Remove-Job -Force
  $tok = ($res | Measure-Object tok -Sum).Sum
  $sec = ($res | Measure-Object sec -Maximum).Maximum
  return [math]::Round($tok / [math]::Max($sec, 0.001), 1)
}

function Probe([int]$maxTok) {
  $b = @{ messages = @(@{ role = 'user'; content = $prompt }); max_tokens = $maxTok; temperature = 0.7 } | ConvertTo-Json -Depth 4
  $lg = (Get-Item $LogPath).Length
  $t0 = Get-Date
  $r = Invoke-RestMethod -Uri $base -Method Post -ContentType 'application/json' -Body $b -TimeoutSec 900
  $dt = ((Get-Date) - $t0).TotalSeconds
  Start-Sleep -Seconds 2
  $fs = [System.IO.File]::Open($LogPath, 'Open', 'Read', 'ReadWrite'); $fs.Seek($lg, 'Begin') | Out-Null
  $sr = New-Object System.IO.StreamReader($fs); $new = $sr.ReadToEnd(); $sr.Close(); $fs.Close()
  $line = ($new -split "`r?`n") | Where-Object { $_ -match 'decode timing' } | Select-Object -First 1
  $o = [pscustomobject]@{ tps = [math]::Round($r.usage.completion_tokens / [math]::Max($dt, 0.001), 1); tok = $r.usage.completion_tokens
                          T = $null; win = $null; wait = $null; cpu = $null; draft = $null }
  if ($line -match 'avg T ([\d.]+), [\d.]+ tokens/window, ([\d.]+) ms/window.*GPU-reach wait ([\d.]+) \+ per-layer host ([\d.]+) \[plan [\d.]+ actq [\d.]+ jobs [\d.]+ CPU ([\d.]+)\].*draft ([\d.]+)') {
    $o.T = [double]$Matches[1]; $o.win = [double]$Matches[2]; $o.wait = [double]$Matches[3]; $o.cpu = [double]$Matches[5]; $o.draft = [double]$Matches[6]
  }
  return $o
}

$eng = Get-Process strata -ErrorAction Stop | Select-Object -First 1
Write-Output "=== GPU PRIORITY A/B | $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') | engine pid $($eng.Id), priority was $($eng.PriorityClass)"
Write-Output "    workload: $(if ($RealWorkload) { "REAL ($MaxTok tok)" } else { "short ($MaxTok tok)" }) | pair arms: $(if ($ConcArms) { $ConcArms -join ',' } else { 'none' })"
Write-Output "    arm            t/s    T     ms/win   GPUwait  CPU   draft   pair_agg"
$rows = @()
foreach ($arm in $Arms) {
  $eng = Get-Process -Id $eng.Id
  $eng.PriorityClass = $arm
  Start-Sleep -Milliseconds 500
  $p = Probe $MaxTok
  $pair = $null
  if ($ConcArms -contains $arm) { $pair = ConcRun 2 600 }
  $rows += [pscustomobject]@{ arm = $arm; p = $p; pair = $pair }
  Write-Output ("    {0,-12} {1,7:N1} {2,6:N2} {3,8:N2} {4,8:N2} {5,5:N2} {6,6:N2}   {7,8}" -f `
    $arm, $p.tps, $p.T, $p.win, $p.wait, $p.cpu, $p.draft, $(if ($pair) { "{0:N1}" -f $pair } else { '-' }))
}
(Get-Process -Id $eng.Id).PriorityClass = 'Normal'
Write-Output "    (engine priority restored to Normal)"
$ok = $rows | Where-Object { $_.p.win }
if ($ok.Count -ge 2) {
  $best = ($ok | Sort-Object { $_.p.win } | Select-Object -First 1)
  $worst = ($ok | Sort-Object { $_.p.win } -Descending | Select-Object -First 1)
  Write-Output ("RESULT: best arm {0} at {1:N2} ms/window ({2:N1} t/s) vs worst {3} at {4:N2} ms/window ({5:N1} t/s) = {6:N2}x" -f `
    $best.arm, $best.p.win, $best.p.tps, $worst.arm, $worst.p.win, $worst.p.tps, ($worst.p.win / [math]::Max($best.p.win, 0.001)))
}
Write-Output "=== END"
