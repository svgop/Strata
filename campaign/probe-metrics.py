#!/usr/bin/env python3
# campaign/probe-metrics.py — T27-B: the probe-side stability harness.
# Scrapes the server's existing vLLM-named Prometheus histograms (TTFT/ITL/E2E), computes p50/p99,
# records the GPU clock/temp/power of the leg, and hashes a deterministic response as a per-boot
# drift alarm. Zero server/engine change; CPU-side only. Usage:
#   .venv/bin/python campaign/probe-metrics.py --port 8081 --label t26-share75-r1
#   (append a JSONL record to campaign/results/metrics.jsonl and print one line)
import argparse, hashlib, json, re, subprocess, time, urllib.request
from datetime import datetime

BUCKET_RE = re.compile(r'^vllm:(time_to_first_token_seconds|inter_token_latency_seconds|e2e_request_latency_seconds)_bucket\{[^}]*le="([^"]+)"\}\s+([0-9.eE+-]+)$', re.M)
COUNT_RE = re.compile(r'^vllm:(time_to_first_token_seconds|inter_token_latency_seconds|e2e_request_latency_seconds)_(count|sum)(?:\{[^}]*\})?\s+([0-9.eE+-]+)$', re.M)
GAUGE_RE = re.compile(r'^vllm:(num_requests_running|num_requests_waiting|kv_cache_usage_perc)(?:\{[^}]*\})?\s+([0-9.eE+-]+)$', re.M)
SPEC_RE = re.compile(r'^vllm:(spec_decode_num_draft_tokens_total|spec_decode_num_accepted_tokens_total)(?:\{[^}]*\})?\s+([0-9.eE+-]+)$', re.M)

GOLDEN_PROMPT = 'List the first eight prime numbers, comma-separated, nothing else.'
GOLDEN_FILE = '/home/svg/LLM/strata-custom-2/campaign/results/golden-response.sha256'


def scrape(port, key):
    req = urllib.request.Request(f'http://127.0.0.1:{port}/metrics', headers={'Accept': 'text/plain'})
    if key:
        req.add_header('Authorization', 'Bearer ' + key)
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read().decode('utf-8', 'replace')


def quantile(buckets, q):
    """Prometheus-style histogram quantile from cumulative bucket counts."""
    if not buckets:
        return None
    total = buckets[-1][1]
    if total <= 0:
        return None
    rank = q * total
    prev_le, prev_c = 0.0, 0
    for le, c in buckets:
        if c >= rank:
            if c == prev_c:
                return prev_le if le == float('inf') else le
            if le == float('inf'):
                # the quantile sits past the last finite bucket: report that bound, not inf
                return prev_le
            frac = (rank - prev_c) / (c - prev_c)
            return prev_le + frac * (le - prev_le)
        prev_le, prev_c = le, c
    return buckets[-1][0]


def parse(text):
    hist = {}
    for name, le, c in BUCKET_RE.findall(text):
        le_f = float('inf') if le == '+Inf' else float(le)
        hist.setdefault(f'vllm:{name}', []).append((le_f, float(c)))
    out = {}
    for name, buckets in hist.items():
        buckets.sort()
        out[name] = {q: quantile(buckets, q) for q in (0.5, 0.95, 0.99)}
    counts = {f'vllm:{n}_{k}': float(v) for n, k, v in COUNT_RE.findall(text)}
    gauges = {f'vllm:{n}': float(v) for n, v in GAUGE_RE.findall(text)}
    spec = {f'vllm:{n}': float(v) for n, v in SPEC_RE.findall(text)}
    return out, counts, gauges, spec


def gpu():
    try:
        s = subprocess.run(['nvidia-smi', '--query-gpu=clocks.sm,temperature.gpu,power.draw,utilization.gpu,memory.used',
                            '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=10).stdout.strip()
        clock, temp, power, util, mem = [x.strip() for x in s.split(',')]
        return {'sm_clock_mhz': float(clock), 'temp_c': float(temp), 'power_w': float(power),
                'util_pct': float(util), 'mem_used_mib': float(mem)}
    except Exception:
        return {}


def golden(port, key):
    body = json.dumps({'messages': [{'role': 'user', 'content': GOLDEN_PROMPT}],
                       'max_tokens': 32, 'temperature': 0}).encode()
    req = urllib.request.Request(f'http://127.0.0.1:{port}/v1/chat/completions', data=body,
                                 headers={'Content-Type': 'application/json'})
    if key:
        req.add_header('Authorization', 'Bearer ' + key)
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            d = json.load(r)
        txt = d['choices'][0]['message'].get('content') or ''
        h = hashlib.sha256(txt.encode()).hexdigest()[:16]
        return {'hash': h, 'chars': len(txt), 'sec': round(time.time() - t0, 3)}
    except Exception as e:
        return {'hash': f'ERROR:{type(e).__name__}', 'chars': 0, 'sec': round(time.time() - t0, 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=8081)
    ap.add_argument('--label', default='leg')
    ap.add_argument('--key', default='3f0940df8efb9c363e4189956ae907d3e4acac221aab39f44804fe0b0df1b8')
    ap.add_argument('--out', default='/home/svg/LLM/strata-custom-2/campaign/results/metrics.jsonl')
    ap.add_argument('--no-golden', action='store_true')
    a = ap.parse_args()

    rec = {'ts': datetime.now().isoformat(timespec='seconds'), 'label': a.label, 'gpu': gpu()}
    try:
        hist, counts, gauges, spec = parse(scrape(a.port, a.key))
        rec['hist'] = hist
        rec['counts'] = counts
        rec['gauges'] = gauges
        rec['spec'] = spec
        if spec:
            draft = spec.get('vllm:spec_decode_num_draft_tokens_total', 0.0)
            acc = spec.get('vllm:spec_decode_num_accepted_tokens_total', 0.0)
            rec['acceptance_pct'] = round(100.0 * acc / draft, 2) if draft else None
    except Exception as e:
        rec['scrape_error'] = f'{type(e).__name__}: {e}'

    if not a.no_golden:
        g = golden(a.port, a.key)
        rec['golden'] = g
        try:
            prev = open(GOLDEN_FILE).read().strip()
        except OSError:
            prev = ''
        if prev and prev != g['hash']:
            rec['drift'] = f'{prev} -> {g["hash"]}'
        elif not prev and g['chars']:
            open(GOLDEN_FILE, 'w').write(g['hash'] + '\n')

    with open(a.out, 'a') as f:
        f.write(json.dumps(rec) + '\n')

    h = rec.get('hist', {})
    def p(name, q):
        v = h.get(name, {}).get(q)
        return f'{v*1000:.0f}ms' if isinstance(v, float) else 'n/a'
    print(f"[metrics {a.label}] TTFT p50/p99 {p('vllm:time_to_first_token_seconds',0.5)}/{p('vllm:time_to_first_token_seconds',0.99)}"
          f"  ITL p99 {p('vllm:inter_token_latency_seconds',0.99)}"
          f"  accept {rec.get('acceptance_pct')}%  clock {rec['gpu'].get('sm_clock_mhz')}MHz {rec['gpu'].get('temp_c')}C {rec['gpu'].get('power_w')}W"
          f"  golden {rec.get('golden',{}).get('hash')}" + (f"  DRIFT {rec['drift']}" if rec.get('drift') else ''))


if __name__ == '__main__':
    main()
