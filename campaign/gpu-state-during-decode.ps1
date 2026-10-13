# campaign/gpu-state-during-decode.ps1 — WHAT IS HOLDING THE CARD DOWN?
#
# F-141..F-147 blamed the desktop compositor (dwm) for the degraded state (GPU-reach wait
# 8.4-10.4 ms healthy vs 20-32 ms degraded, a 2.3x swing).  That correlation is now broken:
# on the boot of 18:58:32 the engine read 30.84 ms GPU-reach wait at 46.3 t/s while dwm held
# 1.0% of the GPU.  So the degraded state has a different cause and this instrument finds it.
#
# It samples, DURING a live decode of the real workload class, four independent families:
#   1. the card's own state        clocks.sm / clocks.mem / power / temp / throttle reasons
#   2. the link state              pcie.link.gen.current / pcie.link.width.current
#   3. who else holds the GPU      per-process \GPU Engine(*)\Utilization Percentage, ranked
#   4. the engine's own accounting the decode-timing line for THAT request (ms/window, wait, CPU)
# (1+2+3) sampled while the engine is busy, so the engine's own share is in the same ranking.
#
# Usage: pwsh -File campaign\gpu-state-during-decode.ps1 -Port 8081 -MaxTok 400 -Samples 8
param(
  [int]$Port = 8081,
  [int]$MaxTok = 400,
  [int]$Samples = 8,
  [string]$LogPath = 'E:\LLM\Engine\strata-custom-2\strata-iq3_s.log',
  [string]$Prompt = 'Explain in detail the tradeoffs of expert caching in a mixture-of-experts inference engine, with concrete numbers.'
)

$base = "http://127.0.0.1:$Port/v1/chat/completions"
$logFile = Get-Item $LogPath
$off0 = $logFile.Length

$sb = @{ messages = @(@{ role = 'user'; content = $Prompt }); max_tokens = $MaxTok; temperature = 0.7 } | ConvertTo-Json -Depth 4
$job = Start-Job -ScriptBlock {
  param($u, $b)
  $t0 = Get-Date
  $r = Invoke-RestMethod -Uri $u -Method Post -ContentType 'application/json' -Body $b -TimeoutSec 900
  [pscustomobject]@{ tok = $r.usage.completion_tokens; sec = ((Get-Date) - $t0).TotalSeconds }
} -ArgumentList $base, $sb

Start-Sleep -Milliseconds 1500   # let the prompt finish; sample the decode proper

$gpu = @()
$util = @{}
for ($i = 1; $i -le $Samples; $i++) {
  $q = (nvidia-smi --query-gpu=clocks.sm,clocks.mem,power.draw,power.limit,utilization.gpu,utilization.memory,temperature.gpu,pcie.link.gen.current,pcie.link.width.current,clocks_throttle_reasons.active --format=csv,noheader,nounits) -join ''
  $gpu += $q.Trim()
  $c = Get-Counter '\GPU Engine(*)\Utilization Percentage' -MaxSamples 1 -SampleInterval 1 -ErrorAction SilentlyContinue
  foreach ($s in $c.CounterSamples) {
    if ($s.CookedValue -le 0.5) { continue }
    if ($util.ContainsKey($s.InstanceName)) { $util[$s.InstanceName] += $s.CookedValue } else { $util[$s.InstanceName] = $s.CookedValue }
  }
}

$res = $job | Wait-Job | Receive-Job
$job | Remove-Job -Force

Write-Output "=== GPU STATE DURING DECODE | $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') | port $Port"
Write-Output ("    request: {0} tok in {1:N2}s = {2:N1} t/s" -f $res.tok, $res.sec, ($res.tok / [math]::Max($res.sec, 0.001)))
Write-Output "--- the card's own state ($Samples samples during the decode) ---"
Write-Output "    sm_clk  mem_clk  power  limit  gpu%  mem%  temp  lgen  lwidth  throttle"
foreach ($g in $gpu) { Write-Output "    $g" }

Write-Output "--- who holds the GPU (summed % over $Samples samples, engine ranked with everyone else) ---"
$rows = $util.GetEnumerator() | ForEach-Object {
  $inst = $_.Key
  $pid_ = if ($inst -match '^pid_(\d+)_') { [int]$Matches[1] } else { 0 }
  $eng = if ($inst -match 'engtype_(\w+)$') { $Matches[1] } else { '?' }
  [pscustomobject]@{ pid = $pid_; proc = (Get-Process -Id $pid_ -ErrorAction SilentlyContinue).ProcessName; eng = $eng; share = [math]::Round($_.Value, 1) }
}
$byProc = $rows | Group-Object pid, proc | ForEach-Object {
  [pscustomobject]@{ pid = $_.Group[0].pid; proc = $_.Group[0].proc; share = [math]::Round(($_.Group | Measure-Object share -Sum).Sum, 1) }
}
$byProc | Sort-Object share -Descending | Select-Object -First 10 | Format-Table -AutoSize

Write-Output "--- the engine's own accounting for this request ---"
$fs = [System.IO.File]::Open($LogPath, 'Open', 'Read', 'ReadWrite'); $fs.Seek($off0, 'Begin') | Out-Null
$sr = New-Object System.IO.StreamReader($fs); $new = $sr.ReadToEnd(); $sr.Close(); $fs.Close()
($new -split "`r?`n") | Where-Object { $_ -match 'decode timing|hit rate|drafts accepted|adapt overlap|PCIe share' } | ForEach-Object { "    $_" }
Write-Output "=== END"
