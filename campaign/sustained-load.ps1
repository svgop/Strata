# campaign/sustained-load.ps1 — a real-workload-shaped load generator.
# Sequential requests with a short gap, the shape of an agent working through turns.
# The engine (booted with STRATA_SPLIT_TIMING/DECODE_TIMING) logs the per-window and
# per-request breakdown for every one of them, so the degradation curve comes out with
# its causes attached.
# Usage: pwsh -File campaign\sustained-load.ps1 -Minutes 20 -MaxTok 200
param(
  [int]$Port = 8081,
  [int]$Minutes = 20,
  [int]$MaxTok = 200,
  [int]$GapMs = 250,
  [string]$Label = 'load'
)
$base = "http://127.0.0.1:$Port/v1/chat/completions"
$prompts = @(
  'Explain how expert caching and adaptive swapping interact in a mixture-of-experts inference engine running on a single GPU.',
  'Describe the tradeoffs between speculative decoding depth and verify-window cost for a memory-bound MoE engine.',
  'A user reports that generation slows down after thirty minutes of use but a restart fixes it. List the mechanisms that could cause that and how to distinguish them.',
  'Compare PCIe transfer paths, pinned host memory and CPU expert compute for serving mixture-of-experts weights that do not fit in VRAM.',
  'Summarize how a KV cache interacts with a paged attention index when a long conversation is parked and later resumed.'
)
$deadline = (Get-Date).AddMinutes($Minutes)
$n = 0; $tot = 0; $t0 = Get-Date
Write-Output "=== SUSTAINED LOAD $Label | $(Get-Date -Format 'HH:mm:ss') for $Minutes min | max_tokens $MaxTok"
while ((Get-Date) -lt $deadline) {
  $n++
  $p = $prompts[($n - 1) % $prompts.Count]
  $body = @{ messages = @(@{ role = 'user'; content = $p }); max_tokens = $MaxTok; temperature = 0.7 } | ConvertTo-Json -Depth 4
  $s = Get-Date
  try {
    $r = Invoke-RestMethod -Uri $base -Method Post -ContentType 'application/json' -Body $body -TimeoutSec 900
    $dt = ((Get-Date) - $s).TotalSeconds
    $tok = $r.usage.completion_tokens
    $tot += $tok
    Write-Output ("  [{0,4}] {1,5} tok in {2,7:N2}s = {3,7:N1} t/s   (elapsed {4:N1} min)" -f $n, $tok, $dt, ($tok / [math]::Max($dt, 0.001)), ((Get-Date) - $t0).TotalMinutes)
  } catch {
    Write-Output ("  [{0,4}] ENGINE NOT ANSWERING: {1}" -f $n, $_.Exception.Message)
    break
  }
  Start-Sleep -Milliseconds $GapMs
}
$mins = ((Get-Date) - $t0).TotalMinutes
Write-Output ("=== LOAD END ${Label}: {0} requests, {1} tokens in {2:N1} min ({3:N1} tok/min)" -f $n, $tot, $mins, ($tot / [math]::Max($mins, 0.001)))
