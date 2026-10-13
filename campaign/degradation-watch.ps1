# campaign/degradation-watch.ps1 — THE DEGRADATION INSTRUMENT (F-119's law, made a curve).
#
# One boot, one fixed warm request, repeated. Each sample records the engine's own accounting:
#   t/s, ms/window, tokens/window, acceptance, hit rate, and the state counters that could grow.
# The point is to turn "10.3 t/s aged vs 101.6 fresh" into a timeline with the suspect counters
# aligned to it, so the bisect has a target instead of a hunch.
#
# Counters sampled every request (all read from the engine log, the source of truth):
#   exch    - "N exchanged with the VRAM tier" (adaptive tier churn, cumulative)
#   ramBlob - "since the start RAM N blobs" (CPU/RAM expert path, cumulative)
#   parkMB  - "bytes=" of the conversation cache (parked conversations)
#   evict   - "evictions=" of the conversation cache
#   kvLook  - "N block reads" of the KV streaming counter for this request
#   rescue  - typical-acceptance first-rejections rescued
#
# Usage: pwsh -File campaign\degradation-watch.ps1 -Samples 20 -IntervalSec 45
param(
  [int]$Port = 8081,
  [int]$Samples = 20,
  [int]$IntervalSec = 45,
  [int]$MaxTok = 200,
  [string]$Label = 'watch',
  [string]$LogPath = 'E:\LLM\Engine\strata-custom-2\strata-iq3_s.log'
)

$base = "http://127.0.0.1:$Port/v1/chat/completions"
$prompt = 'Summarize the tradeoffs between static expert residency and dynamic expert swapping in a memory-constrained mixture-of-experts inference engine. Be concrete.'

function Read-New([long]$from) {
  $fs = [System.IO.File]::Open($LogPath, 'Open', 'Read', 'ReadWrite')
  try {
    if ($from -gt $fs.Length) { $from = 0 }
    $fs.Seek($from, 'Begin') | Out-Null
    $sr = New-Object System.IO.StreamReader($fs)
    $txt = $sr.ReadToEnd(); $sr.Close()
    return @{ text = $txt; len = $fs.Length }
  } finally { $fs.Close() }
}

$bootLine = (Select-String -Path $LogPath -Pattern 'engine started' | Select-Object -Last 1)
$bootTime = [datetime]::ParseExact((($bootLine.Line -split '\s+')[1..2] -join ' '), 'yyyy-MM-dd HH:mm:ss', $null)
Write-Output "=== DEGRADATION WATCH $Label | start $(Get-Date -Format 'HH:mm:ss') | boot $($bootTime.ToString('HH:mm:ss')) | log=$LogPath"
Write-Output "    sample | age_min |  t/s | ms/win | tok/win | draft/win | accept% | hit% |   exch | ramBlob | parkMB | evict | kvLook | rescue"

$rows = @()
$off = (Get-Item $LogPath).Length
for ($i = 1; $i -le $Samples; $i++) {
  $body = @{ messages = @(@{ role = 'user'; content = $prompt }); max_tokens = $MaxTok; temperature = 0.7 } | ConvertTo-Json -Depth 4
  $t0 = Get-Date
  $tps = 0
  try {
    $r = Invoke-RestMethod -Uri $base -Method Post -ContentType 'application/json' -Body $body -TimeoutSec 900
    $wall = ((Get-Date) - $t0).TotalSeconds
    $tps = [math]::Round($r.usage.completion_tokens / [math]::Max($wall, 0.001), 1)
  } catch {
    Write-Output ("    {0,6} | ENGINE NOT ANSWERING: {1}" -f $i, $_.Exception.Message)
    break
  }
  Start-Sleep -Milliseconds 1200
  $chunk = Read-New $off
  $off = $chunk.len
  $txt = $chunk.text

  $gen = $null; $ms = $null; $acc = $null; $of = $null
  if ($txt -match '(?s)prompt (\d+) tokens.*?(\d+) generated in (\d+) ms.*?drafts accepted (\d+) of (\d+)') {
    $gen = [int]$Matches[2]; $ms = [int]$Matches[3]; $acc = [int]$Matches[4]; $of = [int]$Matches[5]
  }
  $hit = if ($txt -match 'decode expert cache hit rate: ([\d.]+)%') { [double]$Matches[1] } else { $null }
  $exch = if ($txt -match 'of experts in RAM, (\d+) exchanged') { [int]$Matches[1] } else { $null }
  $ramBlob = if ($txt -match 'since the start RAM (\d+) blobs') { [int]$Matches[1] } else { $null }
  $parkMB = if ($txt -match 'parked=\d+ bytes=(\d+)') { [math]::Round([int]$Matches[1] / 1MB, 0) } else { $null }
  $evict = if ($txt -match 'evictions=(\d+)') { [int]$Matches[1] } else { $null }
  $kvLook = if ($txt -match 'KV streaming: [\d.]+% of (\d+) block reads') { [int]$Matches[1] } else { $null }
  $rescue = if ($txt -match 'typical acceptance: (\d+) of (\d+) first-rejections') { "$($Matches[1])/$($Matches[2])" } else { $null }

  $win = if ($gen -and $acc) { $gen - $acc } else { $null }
  $mswin = if ($win -and $win -gt 0) { [math]::Round($ms / $win, 1) } else { $null }
  $tokwin = if ($win -and $win -gt 0) { [math]::Round($gen / $win, 2) } else { $null }
  $draftwin = if ($win -and $win -gt 0 -and $of) { [math]::Round($of / $win, 2) } else { $null }
  $accept = if ($of -and $of -gt 0) { [math]::Round(100 * $acc / $of, 1) } else { $null }
  $age = [math]::Round(((Get-Date) - $bootTime).TotalMinutes, 1)

  $row = [pscustomobject]@{ i = $i; age = $age; tps = $tps; mswin = $mswin; tokwin = $tokwin; draftwin = $draftwin
    accept = $accept; hit = $hit; exch = $exch; ramBlob = $ramBlob; parkMB = $parkMB; evict = $evict; kvLook = $kvLook; rescue = $rescue }
  $rows += $row
  Write-Output ("    {0,6} | {1,7} | {2,4} | {3,6} | {4,7} | {5,9} | {6,7} | {7,4} | {8,6} | {9,7} | {10,6} | {11,5} | {12,6} | {13}" -f `
      $i, $age, $tps, $mswin, $tokwin, $draftwin, $accept, $hit, $exch, $ramBlob, $parkMB, $evict, $kvLook, $rescue)
  if ($i -lt $Samples) { Start-Sleep -Seconds $IntervalSec }
}

# --- verdict -----------------------------------------------------------------
$good = $rows | Where-Object { $_.mswin }
if ($good.Count -ge 6) {
  $third = [int]($good.Count / 3)
  $first = ($good[0..($third - 1)] | Measure-Object mswin -Average).Average
  $last = ($good[($good.Count - $third)..($good.Count - 1)] | Measure-Object mswin -Average).Average
  $ratio = $last / [math]::Max($first, 0.001)
  $verdict = if ($ratio -gt 1.3) { 'DEGRADING' } elseif ($ratio -lt 0.77) { 'RECOVERING' } else { 'stable' }
  Write-Output ("VERDICT $verdict : ms/window first third {0:N1} -> last third {1:N1} (x{2:N2}) over {3} samples, {4:N1} min" -f $first, $last, $ratio, $good.Count, ($good[-1].age - $good[0].age))
  Write-Output ("    t/s    first {0:N1} -> last {1:N1}" -f ($good[0..($third - 1)] | Measure-Object tps -Average).Average, ($good[($good.Count - $third)..($good.Count - 1)] | Measure-Object tps -Average).Average)
  foreach ($c in 'exch', 'parkMB', 'evict', 'kvLook') {
    $a = ($good | Where-Object { $_.$c -ne $null } | Select-Object -First 1).$c
    $b = ($good | Where-Object { $_.$c -ne $null } | Select-Object -Last 1).$c
    Write-Output ("    {0,-7} first {1} -> last {2}" -f $c, $a, $b)
  }
} else {
  Write-Output "VERDICT inconclusive: only $($good.Count) samples with a parseable window"
}
Write-Output "=== WATCH END $Label"
