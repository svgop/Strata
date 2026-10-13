# campaign/pcie-split-sweep.ps1 — THE FROZEN SPLIT, SWEPT IN ONE WINDOW.
#
# F-154/F-155 established that the window cost is dominated by the MISS path: the CPU term is
# (missed experts per layer-window) x n_layers x ~85 us, and it is what makes a high-T window
# cost 34.82 ms where a healthy one costs 15.  The engine's own self-tuning of that path
# (PcieAuto) is dormant on this lane twice over - generate.cpp:2829 never runs the link probe
# because the adopted config passes --pcie-frac 0.35 explicitly, and generate.cpp:8353 also
# requires o.batch == 0 while this lane runs the batch path.  So the split has sat at a value
# chosen in an earlier era, on the dominant term, unmeasured.
#
# THE DESIGN THAT MAKES THIS DECISIVE: the engine accepts `pcie_frac` as a PER-REQUEST tuning
# key (generate.cpp:10277, forwarded by serve/server.py:1005 from sampling.strata_tune, the
# mechanism tools/calibrate.py uses).  With temperature 0 the model emits the SAME tokens in
# every arm, so every arm routes to the SAME experts and reads the SAME miss set: any
# difference between arms is the SPLIT ALONE, not content and not drift.  Arms are run in
# forward order and then reversed, so a monotone drift cannot be mistaken for an arm effect.
#
# Self-verifying: with pcie_frac 0.0 the per-layer-window "PCIe" count must fall to ~0 and the
# CPU miss count must carry the whole set; with 1.0 the two swap.  If the columns do not move,
# the tune did not reach the engine and the run is void (recorded, not argued).
#
# Usage: pwsh -File campaign\pcie-split-sweep.ps1 -Port 8081 -MaxTok 1200 -Passes 2
param(
  [int]$Port = 8081,
  [string]$Label = 'pcie-split',
  [string]$LogPath = 'E:\LLM\Engine\strata-custom-2\strata-iq3_s.log',
  [int]$MaxTok = 1200,
  [int]$Passes = 2,
  [string[]]$Arms = @('0.0', '0.2', '0.35', '0.55', '0.75', '1.0'),
  # the engine's per-request tuning keys (generate.cpp:10264-10279, serve/server.py:1005):
  #   pcie_frac  - the share of missed experts the GPU fetches over PCIe instead of the CPU pool
  #   spec_min_p - the draft-probability floor; 1.0 blocks every draft (T becomes exactly 1) and lower
  #                values lengthen the verify window. WITH temperature 0 THE EMITTED TOKENS ARE THE
  #                SAME IN EVERY ARM, so a spec_min_p sweep is a WINDOW-SIZE sweep at identical routing.
  [string]$Key = 'pcie_frac',
  [double]$Standing = 0.35,
  # 0 holds the routing identical across arms (the cleanest mechanism probe); 0.7 is the production
  # regime the bar metric is measured in, where the outputs differ per arm and only interleaving controls it
  # the arms that also carry a PAIR probe: M1 alone cannot gate an adoption (six-metric ultimatum)
  [string[]]$ConcArms = @(),
  [double]$Temp = 0,
  [string]$Prompt = 'Write a comprehensive technical analysis of modern GPU memory hierarchies and their impact on LLM inference performance. Cover HBM bandwidth, L2 cache behavior, PCIe transfer overhead, and pinned memory tradeoffs, with concrete numbers and examples throughout.'
)

$base = "http://127.0.0.1:$Port/v1/chat/completions"
# pwsh -File passes "a,b,c" as ONE string: split and convert so the documented -Arms a,b,c usage works
$Arms = @($Arms | ForEach-Object { $_ -split ',' } | Where-Object { $_ } | ForEach-Object { [double]$_ })
$ConcArms = @($ConcArms | ForEach-Object { $_ -split ',' } | Where-Object { $_ } | ForEach-Object { [double]$_ })
$concPrompt = 'Explain in detail how mixture-of-experts routing works in modern LLMs, including load balancing, capacity factors, and the tradeoffs versus dense models.'

# M2 in the same window: the pair aggregate. The batch path runs DRAFTLESS on this build
# ("slot drafts accepted 0 of 0"), so a drafting knob should leave this number alone - which is
# exactly what makes it a clean no-regression check for spec_min_p.
function ConcRun([double]$frac, [int]$n = 2, [int]$maxTok = 600) {
  $jobs = 1..$n | ForEach-Object {
    Start-Job -ScriptBlock {
      param($u, $p, $mt, $k, $f)
      $tune = @{}; $tune[$k] = $f
      $b = @{ messages = @(@{ role = 'user'; content = $p }); max_tokens = $mt; temperature = 0.7
              sampling = @{ strata_tune = $tune }; strata_tune = $tune } | ConvertTo-Json -Depth 6
      $t0 = Get-Date
      $r = Invoke-RestMethod -Uri $u -Method Post -ContentType 'application/json' -Body $b -TimeoutSec 900
      [pscustomobject]@{ tok = $r.usage.completion_tokens; sec = ((Get-Date) - $t0).TotalSeconds }
    } -ArgumentList $base, $concPrompt, $maxTok, $Key, $frac
  }
  $res = $jobs | Wait-Job | Receive-Job; $jobs | Remove-Job -Force
  $tok = ($res | Measure-Object tok -Sum).Sum
  $sec = ($res | Measure-Object sec -Maximum).Maximum
  return [math]::Round($tok / [math]::Max($sec, 0.001), 1)
}

function Probe([double]$frac) {
  $tune = @{}; $tune[$Key] = $frac
  $body = @{
    messages    = @(@{ role = 'user'; content = $Prompt })
    max_tokens  = $MaxTok
    temperature = $Temp
    sampling    = @{ strata_tune = $tune }
    strata_tune = $tune
  } | ConvertTo-Json -Depth 6
  $lg = (Get-Item $LogPath).Length
  $t0 = Get-Date
  try { $r = Invoke-RestMethod -Uri $base -Method Post -ContentType 'application/json' -Body $body -TimeoutSec 900 }
  catch { return [pscustomobject]@{ frac = $frac; ok = $false; err = $_.Exception.Message } }
  $dt = ((Get-Date) - $t0).TotalSeconds
  Start-Sleep -Milliseconds 1500
  $fs = [System.IO.File]::Open($LogPath, 'Open', 'Read', 'ReadWrite'); $fs.Seek($lg, 'Begin') | Out-Null
  $sr = New-Object System.IO.StreamReader($fs); $new = $sr.ReadToEnd(); $sr.Close(); $fs.Close()
  $lines = $new -split "`r?`n"
  $l = $lines | Where-Object { $_ -match 'decode timing' } | Select-Object -First 1
  $h = $lines | Where-Object { $_ -match 'expert cache hit rate' } | Select-Object -First 1
  $o = [pscustomobject]@{ frac = $frac; ok = $true; tps = [math]::Round($r.usage.completion_tokens / [math]::Max($dt, 0.001), 1)
                          tok = $r.usage.completion_tokens; T = $null; win = $null; wait = $null; CPU = $null
                          miss = $null; pcie = $null; vram = $null; hit = $null; acc = $null; drafts = $null; wins = $null }
  if ($l -match 'avg T ([\d.]+), [\d.]+ tokens/window, ([\d.]+) ms/window.*GPU-reach wait ([\d.]+) \+ per-layer host [\d.]+ \[plan [\d.]+ actq [\d.]+ jobs [\d.]+ CPU ([\d.]+)\].*CPU experts ([\d.]+) \([\d.]+ entries\), VRAM hits ([\d.]+), PCIe ([\d.]+)') {
    $o.T = [double]$Matches[1]; $o.win = [double]$Matches[2]; $o.wait = [double]$Matches[3]
    $o.CPU = [double]$Matches[4]; $o.miss = [double]$Matches[5]; $o.vram = [double]$Matches[6]; $o.pcie = [double]$Matches[7]
  }
  $d = @($lines | Where-Object { $_ -match 'drafts accepted (\d+) of (\d+)' }) | Select-Object -First 1
  if ($d -and $d -match 'drafts accepted (\d+) of (\d+)') {
    $o.drafts = [int]$Matches[2]
    if ($o.drafts -gt 0) { $o.acc = [math]::Round(100.0 * [int]$Matches[1] / [int]$Matches[2], 1) }
  }
  if ($h -match 'hit rate: ([\d.]+)%') { $o.hit = [double]$Matches[1] }
  return $o
}

$boot = (Select-String -Path $LogPath -Pattern 'engine started' | Select-Object -Last 1).Line
Write-Output "=== TUNE SWEEP $Label ($Key) | $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') | boot $($boot.Substring(10,19))"
Write-Output "    ${MaxTok} greedy tokens per arm (identical routing across arms) | $Passes pass(es) forward+reverse | arms $($Arms -join ', ')"
Write-Output "    pass  frac     t/s     T     ms/win  GPUwait   CPU   misses/lw  PCIe/lw  VRAMhits/lw  hit%   acc%  drafts"
$all = @()
for ($p = 1; $p -le $Passes; $p++) {
  $order = if ($p % 2 -eq 1) { $Arms } else { @($Arms | Sort-Object -Descending) }
  foreach ($a in $order) {
    $r = Probe $a
    if (-not $r.ok) { Write-Output ("    {0,4}  {1,-6} FAILED: {2}" -f $p, $a, $r.err); continue }
    $pair = $null
    if ($ConcArms -contains $a) { $pair = ConcRun $a }
    $r | Add-Member -NotePropertyName pair -NotePropertyValue $pair -Force
    $all += $r
    Write-Output ("    {0,4}  {1,-6} {2,6:N1} {3,6:N2} {4,8:N2} {5,8:N2} {6,6:N2} {7,10:N2} {8,8:N2} {9,11:N2} {10,6:N1} {11,6:N1} {12,6} {13,7}" -f `
      $p, $a, $r.tps, $r.T, $r.win, $r.wait, $r.CPU, $r.miss, $r.pcie, $r.vram, $r.hit, $r.acc, $r.drafts, $(if ($pair) { "{0:N1}" -f $pair } else { '-' }))
  }
}
$good = $all | Where-Object { $_.win }
if ($good.Count -ge 2) {
  Write-Output "--- per arm (median of its passes; the split's effect is on CPU/misses, the verdict is on t/s) ---"
  $good | Group-Object frac | Sort-Object { [double]$_.Name } | ForEach-Object {
    $g = $_.Group
    $med = { param($k) $s = @($g | Sort-Object $k); $s[[int]($s.Count / 2)].$k }
    Write-Output ("    frac {0,-5} t/s {1,6:N1} | ms/win {2,6:N2} | CPU {3,5:N2} | misses {4,5:N2} | PCIe {5,5:N2} | hit {6,5:N1}%" -f `
      $_.Name, (& $med 'tps'), (& $med 'win'), (& $med 'CPU'), (& $med 'miss'), (& $med 'pcie'), (& $med 'hit'))
  }
  $best = $good | Sort-Object tps -Descending | Select-Object -First 1
  $stand = $good | Where-Object { $_.frac -eq $Standing } | Sort-Object tps -Descending | Select-Object -First 1
  Write-Output ("BEST ARM: $Key {0} at {1:N1} t/s" -f $best.frac, $best.tps)
  if ($stand) { Write-Output ("  vs the standing $Standing at {0:N1} t/s  =>  {1:N1}%" -f $stand.tps, (100.0 * ($best.tps - $stand.tps) / $stand.tps)) }
}
Write-Output "=== END"
