# campaign/probe-lane.ps1 — THE STANDARD PROBE (Proof Law instrument).
#
# Measures, on a live lane, in one run:
#   M1  single-stream sustained decode t/s        (the headline metric)
#   M2  pair and 6-conc aggregate t/s             (the concurrent class)
#   M3  MTP draft acceptance, per request         (read from the engine log)
#   TREND ms/window per request in order          (the degradation signal)
# Plus the engine's own hit rate and the lane state, and the raw log lines a hostile
# auditor can open.
#
# Why ms/window: t/s hides WHERE the time goes. window_ms = decode_ms / (generated - accepted)
# is the engine's own accounting of one verify window; it is comparable across boots and it
# exposes progressive decay inside a single boot, which t/s alone masks.
#
# Usage: pwsh -File campaign\probe-lane.ps1 -Port 8081 -Label v2
#        pwsh -File campaign\probe-lane.ps1 -Port 8081 -Label v1 -LogPath E:\LLM\Engine\strata-custom\strata-iq3_s.log
param(
  [int]$Port = 8081,
  [string]$Label = 'v2',
  [string]$LogPath = 'E:\LLM\Engine\strata-custom-2\strata-iq3_s.log',
  [int]$ShortN = 6,
  [int]$LongTok = 2000,
  [int]$FreshSections = 90,
  [switch]$SkipConc
)

$base = "http://127.0.0.1:$Port/v1/chat/completions"
$real = 'Write a comprehensive technical analysis of modern GPU memory hierarchies and their impact on LLM inference performance. Cover HBM bandwidth, L2 cache behavior, PCIe transfer overhead, and pinned memory tradeoffs, with concrete numbers and examples throughout.'
$conc = 'Explain in detail how mixture-of-experts routing works in modern LLMs, including load balancing, capacity factors, and the tradeoffs versus dense models.'
# The operator's real class: a short agentic request with a real answer to produce.
$short = 'Summarize the tradeoffs between static expert residency and dynamic expert swapping in a memory-constrained mixture-of-experts inference engine. Be concrete.'

function Probe([string]$prompt, [int]$maxTok, [double]$temp = 0.7) {
  $b = @{ messages = @(@{ role = 'user'; content = $prompt }); max_tokens = $maxTok; temperature = $temp } | ConvertTo-Json -Depth 4
  $t0 = Get-Date
  try {
    $r = Invoke-RestMethod -Uri $base -Method Post -ContentType 'application/json' -Body $b -TimeoutSec 1200
    $dt = ((Get-Date) - $t0).TotalSeconds
    return [pscustomobject]@{ ok = $true; tok = $r.usage.completion_tokens; sec = [math]::Round($dt, 2); tps = [math]::Round($r.usage.completion_tokens / [math]::Max($dt, 0.001), 1) }
  } catch {
    $dt = ((Get-Date) - $t0).TotalSeconds
    return [pscustomobject]@{ ok = $false; tok = 0; sec = [math]::Round($dt, 2); tps = 0 }
  }
}

function ConcRun([int]$n, [int]$maxTok, [string]$tag) {
  $jobs = 1..$n | ForEach-Object {
    Start-Job -ScriptBlock {
      param($u, $p, $mt)
      $b = @{ messages = @(@{ role = 'user'; content = $p }); max_tokens = $mt; temperature = 0.7 } | ConvertTo-Json -Depth 4
      $t0 = Get-Date
      $r = Invoke-RestMethod -Uri $u -Method Post -ContentType 'application/json' -Body $b -TimeoutSec 1200
      [pscustomobject]@{ tok = $r.usage.completion_tokens; sec = ((Get-Date) - $t0).TotalSeconds }
    } -ArgumentList $base, $conc, $maxTok
  }
  $res = $jobs | Wait-Job | Receive-Job
  $jobs | Remove-Job -Force
  $aggTok = ($res | Measure-Object -Property tok -Sum).Sum
  $aggSec = ($res | Measure-Object -Property sec -Maximum).Maximum
  return [pscustomobject]@{ tag = $tag; n = $n; tok = $aggTok; sec = [math]::Round($aggSec, 2); agg = [math]::Round($aggTok / [math]::Max($aggSec, 0.001), 1) }
}

# ---- lane state ------------------------------------------------------------
$logFile = Get-Item $LogPath
$offset0 = $logFile.Length
$boot = (Select-String -Path $LogPath -Pattern 'engine started' | Select-Object -Last 1)
$bootTime = if ($boot) { ($boot.Line -split '\s+')[1..2] -join ' ' } else { '?' }
$used = [int]((nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) | Select-Object -First 1)
$free = 32607 - $used
Write-Output "=== PROBE $Label | $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') | log=$LogPath"
Write-Output "    boot: $bootTime | GPU used ${used} MiB, free ${free} MiB | ps1 pid $PID"

# ---- warmup ---------------------------------------------------------------
$w = Probe 'Say OK.' 8 0
Write-Output "    warmup: $($w.tok) tok in $($w.sec)s"

# ---- phase A: the operator's short class, before any long prompt ----------
$shortsA = @()
for ($i = 1; $i -le $ShortN; $i++) {
  $s = Probe $short 220
  $shortsA += $s
  Write-Output "    A.short[$i]: $($s.tok) tok in $($s.sec)s = $($s.tps) t/s"
}

# ---- the fresh long prompt (the agent-turn shape) -------------------------
# Unique text (no repeated sentences, so neither the suffix drafter nor the prefix cache
# can shortcut it): the closest thing to a fresh agent turn with a large context.
$sb = New-Object System.Text.StringBuilder
for ($i = 1; $i -le $FreshSections; $i++) {
  [void]$sb.AppendLine("Section $i of the incident review. At step $i the operator observed a distinct change in behaviour and recorded it against the runbook entry number $i. The measurement taken at that point was catalogued with identifier $i and compared against the baseline that had been established before the change was introduced.")
}
$longPrompt = $sb.ToString()
$fresh = Probe $longPrompt 80
Write-Output "    fresh-long-prompt: $($fresh.tok) tok out in $($fresh.sec)s (see log for prompt tokens)"

# ---- phase B: the same short class, after the long fresh prompt ----------
$shortsB = @()
for ($i = 1; $i -le $ShortN; $i++) {
  $s = Probe $short 220
  $shortsB += $s
  Write-Output "    B.short[$i]: $($s.tok) tok in $($s.sec)s = $($s.tps) t/s"
}
$ma = ($shortsA | Where-Object { $_.tok -gt 20 } | Measure-Object tps -Average).Average
$mb = ($shortsB | Where-Object { $_.tok -gt 20 } | Measure-Object tps -Average).Average
if ($ma -and $mb) {
  Write-Output ("SHORT-CLASS A/B   : before long prompt {0:N1} t/s -> after {1:N1} t/s ({2:N0}%)" -f $ma, $mb, (100 * ($mb - $ma) / $ma))
}

# ---- M1: sustained single stream ------------------------------------------
$m1 = Probe $real $LongTok
Write-Output "M1 single-stream : $($m1.tps) t/s ($($m1.tok) tok in $($m1.sec)s)"

# ---- M2: pair and 6-conc --------------------------------------------------
if (-not $SkipConc) {
  $p = ConcRun 2 900 'pair'
  Write-Output "M2 pair          : $($p.agg) t/s aggregate ($($p.tok) tok in $($p.sec)s)"
  $c = ConcRun 6 600 '6-conc'
  Write-Output "M2 6-conc        : $($c.agg) t/s aggregate ($($c.tok) tok in $($c.sec)s)"
}

Start-Sleep -Seconds 2

# ---- parse the engine's own accounting for this probe's requests ----------
$fs = [System.IO.File]::Open($LogPath, 'Open', 'Read', 'ReadWrite')
$fs.Seek($offset0, 'Begin') | Out-Null
$sr = New-Object System.IO.StreamReader($fs)
$new = $sr.ReadToEnd(); $sr.Close(); $fs.Close()
$lines = $new -split "`r?`n"

$rows = @(); $pending = $null
foreach ($l in $lines) {
  if ($l -match 'prompt (\d+) tokens.*?(\d+) generated in (\d+) ms.*?drafts accepted (\d+) of (\d+)') {
    if ($pending) { $rows += $pending }
    $g = [int]$Matches[2]; $ms = [int]$Matches[3]; $acc = [int]$Matches[4]; $tot = [int]$Matches[5]
    $win = $g - $acc
    $pending = [pscustomobject]@{
      prompt = [int]$Matches[1]; gen = $g; ms = $ms; acc = $acc; of = $tot; win = $win
      mswin = if ($win -gt 0) { [math]::Round($ms / $win, 1) } else { $null }
      accept = [math]::Round(100 * $acc / [math]::Max($tot, 1), 1); hit = $null
    }
  } elseif ($pending -and $l -match 'decode expert cache hit rate: ([\d.]+)%') {
    $pending.hit = [double]$Matches[1]; $rows += $pending; $pending = $null
  }
}
if ($pending) { $rows += $pending }

Write-Output "--- engine accounting, in order (ms/window is the degradation signal) ---"
$k = 0
foreach ($r in $rows) {
  $k++
  Write-Output ("    [{0,2}] prompt {1,4} gen {2,5} -> {3,7} ms/window | accept {4,5}% ({5}/{6}) | hit {7}%" -f $k, $r.prompt, $r.gen, $r.mswin, $r.accept, $r.acc, $r.of, $r.hit)
}
if ($rows.Count -ge 4) {
  $half = [int]($rows.Count / 2)
  $early = ($rows[0..($half - 1)] | Where-Object mswin | Measure-Object mswin -Average).Average
  $late = ($rows[$half..($rows.Count - 1)] | Where-Object mswin | Measure-Object mswin -Average).Average
  Write-Output ("TREND ms/window: early {0:N1} -> late {1:N1} ({2})" -f $early, $late, $(if ($late -gt 1.3 * $early) { 'DEGRADING' } else { 'stable' }))
}
$accs = $rows | Where-Object { $_.of -ge 8 }
if ($accs) { Write-Output ("M3b rate         : {0:N1}% mean over {1} requests (diagnostic from 2026-10-12, no longer the defended quantity)" -f (($accs | Measure-Object accept -Average).Average), $accs.Count) }

# M3 (amended 2026-10-12 by operator decision): ACCEPTED DRAFT TOKENS PER SECOND is the defended
# quantity, read from the engine's own decode-timing line. Definition: (tokens/window - 1) / (ms/window / 1000).
# Reported beside it, never instead of it: the banked tokens per window (the field's acceptance length,
# vLLM's mean_acceptance_length) and the rate above.
$dt = @()
foreach ($l in $lines) {
  if ($l -match 'avg T ([\d.]+), ([\d.]+) tokens/window, ([\d.]+) ms/window') {
    $tw = [double]$Matches[2]; $mw = [double]$Matches[3]
    if ($mw -gt 0) { $dt += [pscustomobject]@{ tw = $tw; mw = $mw; aps = 1000.0 * ($tw - 1.0) / $mw } }
  }
}
if ($dt) {
  Write-Output ("M3 accepted/s    : {0:N1} accepted draft tokens per second (mean over {1} windows; {2:N2} banked tokens/window at {3:N2} ms/window)" -f `
    (($dt | Measure-Object aps -Average).Average), $dt.Count, (($dt | Measure-Object tw -Average).Average), (($dt | Measure-Object mw -Average).Average))
}
$pos = @($lines | Where-Object { $_ -match 'spec depth: accepted/tested' })
foreach ($p in $pos) { Write-Output "M3d per-position : $p" }
if (-not $pos) { Write-Output "M3d per-position : not printed (arm STRATA_SPEC_DEPTH=1 to publish accepted/tested by depth)" }
Write-Output "=== PROBE END $Label"
