# campaign/draft-yield-sweep.ps1 — the draft-yield instrument.
#
# Four request classes, same window, one after another. For each request it reads the engine's own
# decode-timing line and reports where the window went:
#   T, tokens/window, ms/window, GPU-reach wait, per-layer host, and the CPU-expert term inside it.
# The point: throughput = tokens/window / ms/window, so a deeper window only pays if the CPU term
# does not eat the gain. This measures both sides at once.
#
# Usage: pwsh -File campaign\draft-yield-sweep.ps1 -Label "batch4"
param([int]$Port = 8081, [string]$Label = 'sweep', [string]$LogPath = 'E:\LLM\Engine\strata-custom-2\strata-iq3_s.log')
$base = "http://127.0.0.1:$Port/v1/chat/completions"
$cases = @(
  @{ n = 'repetitive-list'; p = 'List the integers from 1 to 150, one per line, each in the exact format: n, n squared, n cubed. Do not add commentary.'; mt = 700 },
  @{ n = 'code';            p = 'Write a complete Python class implementing an LRU cache with get and put methods, type hints, docstrings and a small example at the end.'; mt = 500 },
  @{ n = 'prose';           p = 'Write a comprehensive technical analysis of modern GPU memory hierarchies and their impact on LLM inference performance, with concrete numbers.'; mt = 500 },
  @{ n = 'literal-repeat';  p = 'Repeat exactly this sentence forty times, one per line, with no extra text: the quick brown fox jumps over the lazy dog.'; mt = 500 }
)
$mark = (Get-Content $LogPath | Measure-Object -Line).Lines
Write-Output "=== DRAFT-YIELD SWEEP $Label | $(Get-Date -Format 'HH:mm:ss') | boot $((Select-String -Path $LogPath -Pattern 'engine started' | Select-Object -Last 1).Line.Substring(14,19))"
foreach ($c in $cases) {
  $b = @{ messages = @(@{ role = 'user'; content = $c.p }); max_tokens = $c.mt; temperature = 0.7 } | ConvertTo-Json -Depth 4
  $t0 = Get-Date
  try {
    $r = Invoke-RestMethod -Uri $base -Method Post -ContentType 'application/json' -Body $b -TimeoutSec 900
    $dt = ((Get-Date) - $t0).TotalSeconds
    Write-Output ("  {0,-16} {1,4} tok in {2,6:N2}s = {3,6:N1} t/s (wall)" -f $c.n, $r.usage.completion_tokens, $dt, ($r.usage.completion_tokens / $dt))
  } catch { Write-Output ("  {0,-16} FAILED: {1}" -f $c.n, $_.Exception.Message) }
}
Start-Sleep -Seconds 2
$lines = Get-Content $LogPath
$i = $mark
$k = 0
while ($i -lt $lines.Count) {
  $l = $lines[$i]
  if ($l -match 'strata decode timing: (\d+) windows, avg T ([\d.]+), ([\d.]+) tokens/window, ([\d.]+) ms/window = verify ([\d.]+) \(GPU-reach wait ([\d.]+) \+ per-layer host ([\d.]+) \[plan ([\d.]+) actq ([\d.]+) jobs ([\d.]+) CPU ([\d.]+)\]') {
    $k++
    $name = if ($k -le $cases.Count) { $cases[$k - 1].n } else { "extra$k" }
    Write-Output ("  {0,-16} T {1,5} | tok/win {2,5} | {3,6} ms/win = GPU wait {4,6} + host {5,6} (CPU {6,6}) | => {7,6:N1} t/s" -f `
        $name, $Matches[2], $Matches[3], $Matches[4], $Matches[6], $Matches[7], $Matches[11], (1000 * [double]$Matches[3] / [double]$Matches[4]))
  }
  $i++
}
Write-Output "=== SWEEP END $Label ($k requests with a decode-timing line)"
