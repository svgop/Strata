"""Compile the four TUs that warned in the F-459 build from the HEAD tree (no F-459 lines) with their exact
build-fresh commands; list the warnings.  Outputs and dep files go to /tmp/f459-base/out."""
import re, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
R = "/home/svg/LLM/strata-custom-2"
B = R + "/build-fresh"
T = "/tmp/f459-base"
objs = ["CMakeFiles/strata_engine.dir/src/core/expert_source.cpp.o",
        "CMakeFiles/strata_engine.dir/src/core/verify.cpp.o",
        "CMakeFiles/strata_engine.dir/src/core/remote_expert_opt.cu.o",
        "CMakeFiles/strata_mmq.dir/src/prefill/moe_fused_iq.cu.o"]
def run(o):
    cmd = subprocess.run(["ninja", "-t", "commands", o], cwd=B, capture_output=True, text=True).stdout.strip().splitlines()[-1]
    src = R + "/" + o.split(".dir/")[1][:-2]          # .../src/core/expert_source.cpp
    name = o.split(".dir/")[1].replace("/", "_")
    cmd = cmd.replace("-I" + R + "/include", "-I" + T + "/include")
    cmd = cmd.replace("-o " + o, "-o " + T + "/out/" + name + ".o")
    cmd = cmd.replace("-MT " + o, "-MT " + T + "/out/" + name + ".o").replace("-MF " + o + ".d", "-MF " + T + "/out/" + name + ".d")
    assert cmd.count(src) == 1, (src, cmd.count(src))
    cmd = cmd.replace(src, T + "/" + src[len(R) + 1:])
    p = subprocess.run(cmd, cwd=B, shell=True, capture_output=True, text=True)
    warn = [l for l in (p.stderr + p.stdout).splitlines() if re.search(r"warning", l) and "Remark" not in l]
    return o, p.returncode, warn
with ThreadPoolExecutor(4) as ex:
    res = list(ex.map(run, objs))
n = 0
for o, rc, warn in res:
    print("%s  exit %d  warnings %d" % (o, rc, len(warn)))
    for w in warn:
        n += 1
        print("    " + w.replace(T, "$HEAD"))
print("BASELINE WARNINGS TOTAL %d" % n)
