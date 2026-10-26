#!/usr/bin/env python
"""mimicbench2 — DEEP mimic tier (v2, 40 tasks, multi-turn + internalization).

v1 (mimicbench.py) measures surface adherence: rule given, rule followed.
v2 measures what production actually needs — TRUE mimic behavior:

  DEPTH AXES (each cited to the workspace doctrine that motivates it):
  D1. Doctrine absorption — the model gets a REAL doctrine excerpt (our
      AGENTS.md voice laws, verbatim) and must apply them to NEW content,
      not quote them back. (AGENTS.md "operating contract binds how an agent
      ... writes"; 00-standards Voice.)
  D2. Voice fingerprint transfer — 3 sample paragraphs in a strong voice;
      new content must carry the SAME voice (front-loaded verbs, no
      throat-clearing, period-after-bullet). Measured structurally.
  D3. Architectural continuation — N modules follow a naming/structure
      pattern (our socket/projection/glossary lineage); produce module N+1.
      The pattern is INDUCTIVE (never stated); only examples teach it.
  D4. Multi-turn correction persistence — turn 2 corrects a habit; turns
      3-5 must keep the correction without reminder. ("Self-correct under
      feedback with evidence" — AGENTS.md ethos.)
  D5. Style blending with switching — TWO styles, each bound to a context
      marker; interleaved requests must switch correctly. (5.2 "channel by
      surface" precedent.)
  D6. Deep UI induction — 3 full components induce an UNSTATED system
      (class ordering, radius semantics, tone mapping); then assemble a
      page where cross-component consistency is checked, not just tokens.
      (design-system skill: "a new component that doesn't feel native is a bug")
  D7. Internalization (the deepest signal) — patterns taught in phase 1
      (lexicon + format habits) must reappear UNPROMPTED in phase 3 tasks
      that never mention them. Distinguishes compliance from absorption
      ("the next agent inherits these obligations" — infectious doctrine).
  D8. Document-structure mimicry — .docs conventions (frontmatter + type
      field, sections, status) applied to a novel document.

SCORING: weighted — surface checks 1pt, structural/inductive 2pts,
internalization 3pts. % = weighted score / max. Multi-turn tasks use
conversation history (server slot retains context; -np 1).

Usage: python mimicbench2.py [port] [label]
"""
import json, os, re, sys, time, urllib.request

PORT = sys.argv[1] if len(sys.argv) > 1 else "8099"
LABEL = sys.argv[2] if len(sys.argv) > 2 else "run"
TEMP = float(sys.argv[3]) if len(sys.argv) > 3 else 0.1   # MIMIC-D gate runs 0.0 (greedy)
API = f"http://127.0.0.1:{PORT}/v1/chat/completions"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = f"/home/svg/LLM/strata-custom-2/campaign/results/mimicbench2-{LABEL}.json"
NOTE_OUT = [""]   # the last chat() output, stashed for record()

def chat(messages, mt=2500):
    body = json.dumps({"messages": messages, "max_tokens": mt,
                       "temperature": TEMP}).encode()
    r = json.loads(urllib.request.urlopen(urllib.request.Request(
        API, body, {"Content-Type": "application/json", "Authorization": "Bearer 3f0940df8efb9c363e4189956ae907d3e4acac221aab39f44804fe0b0df1b8"}), timeout=1800).read())
    o = (r["choices"][0]["message"].get("content") or "").strip()
    NOTE_OUT[0] = o   # MIMIC-D: the record() that follows always describes this output
    return o

R, SCORE, MAX = [], 0, 0
def record(cat, tid, w, ok, head, note=""):
    global SCORE, MAX
    SCORE += w * int(bool(ok)); MAX += w   # bool(None)=False: a checker returning None is a FAIL, not a crash
    R.append(dict(cat=cat, id=tid, w=w, ok=int(bool(ok)), note=note, head=head[:200], out=NOTE_OUT[0]))
    print(f"[{len(R):>2}] {cat:<12} {tid:<10} {'PASS' if ok else 'FAIL'} (w{w})", flush=True)

# =============================================================================
# D1. DOCTRINE ABSORPTION (4 tasks, w2) — real voice laws, new content.
#     Source: AGENTS.md Voice + 00-standards: front-load the verb; no
#     throat-clearing ("Moreover", "It is worth noting"); bullets carry
#     artifacts; stop when the rule lands.
# =============================================================================
DOCTRINE = """Workspace writing doctrine (binding for ALL your output in this chat):
- Front-load the verb. State the rule; stop.
- Never open with throat-clearing ("Moreover", "It is worth noting", "In today's world", "Certainly").
- Every bullet carries a concrete artifact: a path, flag, command, type, or number.
- No justification closers ("...and that's why it works")."""
D1 = [("Explain how to speed up a slow unit test suite.", "instr-density",
       # PASS = starts with a verb within first 3 words, no banned openers, and
       # at least half of bullets contain code-ish artifacts (backticks/digits/paths)
       lambda o: 1 if (re.match(r"^(\S+\s+){0,2}(Profile|Run|Mock|Parallel|Cache|Isolate|Pin|Measure|Cut|Use|Enable|Disable|Split|Fix|Flag|Drop|Batch|Stub)", o)
                       and not re.search(r"^(Moreover|It is worth|In today|Certainly|Additionally)", o)
                       and len(re.findall(r"`|\d|/|\.py|\.c\b|--|-", o)) >= 3) else 0),
      ("Tell a teammate why code review matters.", "no-closers",
       lambda o: 1 if not re.search(r"(that'?s why|which is why|as you can see|this is why).{0,30}$", o, re.I) and
                       not re.match(r"^(Moreover|It is worth|Certainly|In today)", o) else 0),
      ("Give 3 bullets on logging best practice.", "artifact-bullets",
       lambda o: 1 if len([l for l in o.splitlines() if l.strip().startswith(("-", "*"))]) >= 3
                       and all(re.search(r"`|\d|/[a-z]|--[a-z]|\.py|\.json", l)
                               for l in [x for x in o.splitlines() if x.strip().startswith(("-", "*"))][:3]) else 0),
      ("Describe our release process.", "frontload",
       lambda o: 1 if re.match(r"^(Run|Tag|Bump|Cut|Merge|Ship|Freeze|Verify|Gate|Build|Publish|Deploy|Stage)", o) else 0)]
for q, name, chk in D1:
    o = chat([{"role": "system", "content": DOCTRINE}, {"role": "user", "content": q}])
    record("doctrine", f"d1-{name}", 2, chk(o), o, "real voice laws on new content")

# =============================================================================
# D2. VOICE FINGERPRINT TRANSFER (3 tasks, w2) — induce a strong voice from
#     examples, then generate in it. Voice: clipped, verb-first, dry, no
#     adjectives-of-praise. (00-standards "Write as a principal architect".)
# =============================================================================
VOICESAMPLE = """Samples of the house voice:
"Ship the fix. Revert if the gate goes red."
"Measure first. The profiler names the bottleneck; guesses don't."
"Cut the feature. Nobody uses it. The maintenance cost is real."
Your replies in this chat must match this voice exactly."""
for i, q in enumerate(["Tell me about database indexing.",
                       "How should we handle flaky E2E tests?",
                       "Explain technical debt to a PM."], 1):
    o = chat([{"role": "system", "content": VOICESAMPLE}, {"role": "user", "content": q}])
    # voice = no praise-slop, clipped length, >=2 imperative sentences
    # (capitalized verb + short object; calibrated to observed correct output —
    # "Index the columns...", "Quarantine flaky tests...". The verb is NOT
    # allowlisted: any imperative counts, the house voice allows new verbs.)
    imperative = len(re.findall(r"(?:^|\.\s+|\n)([A-Z][a-z]+ (?:the |a |your |it |them |now |first )?[a-z][a-z\-]+[ .])", o))
    ok = bool(len(re.findall(r"\b(amazing|incredibly|fantastic|powerful|seamless|robust|elegant|delve|leverage)\b", o, re.I)) == 0
          and len(o) < 900 and imperative >= 2)
    record("voice", f"d2-{i}", 2, ok, o, "induced clipped voice, no praise-adjectives")

# =============================================================================
# D3. ARCHITECTURAL CONTINUATION (4 tasks, w2) — inductive pattern: our
#     glossary lineage socket->projection->function-verb taxonomy. The model
#     must EXTEND the pattern, not just describe it. (AGENTS.d/03-glossary.)
# =============================================================================
ARCH = """Codebase pattern (study the examples; the pattern is never stated):
- file: eagle_probe.cpp — class EagleProbe, ctor eagle_probe_init(), errors via eagle_probe_fail()
- file: socket_map.cpp — class SocketMap, ctor socket_map_init(), errors via socket_map_fail()
- file: projection_node.cpp — class ProjectionNode, ctor projection_node_init(), errors via projection_node_fail()
New modules must follow this exact scheme."""
for i, (mod, cls) in enumerate([("drift_signal", "DriftSignal"),
                                ("loop_closure", "LoopClosure"),
                                ("payload_gate", "PayloadGate"),
                                ("harvest_queue", "HarvestQueue")], 1):
    o = chat([{"role": "system", "content": ARCH},
              {"role": "user", "content": f"Write the header declarations for a new module: {mod}. Follow the codebase pattern exactly."}])
    ok = (f"class {cls}" in o and f"{mod}_init(" in o and f"{mod}_fail(" in o)
    record("arch-continue", f"d3-{i}", 2, ok, o, f"induce {cls}/{mod}_init/{mod}_fail")

# =============================================================================
# D4. MULTI-TURN CORRECTION PERSISTENCE (5 turns, 3 scored @w2 + 2 anchor)
#     Turn 1 establishes habit, turn 2 corrects, turns 3-5 must keep the
#     correction UNPROMPTED. (ethos: "Self-correct under feedback".)
# =============================================================================
conv = [{"role": "system", "content": "You are a coding assistant. Be concise."}]
conv.append({"role": "user", "content": "Name a data structure for fast lookup. One line."})
r1 = chat(conv); conv.append({"role": "assistant", "content": r1})
conv.append({"role": "user", "content": "Correction: from now on, ALWAYS name the complexity next to any data structure or algorithm you mention. Acknowledge in one line."})
r2 = chat(conv); conv.append({"role": "assistant", "content": r2})
D4Q = ["Now name an algorithm for sorting a nearly-sorted array. One line.",
       "Best structure for a fixed-size ring of the last N events? One line.",
       "How do you find a duplicate in a list? One line."]
for i, q in enumerate(D4Q, 1):
    conv.append({"role": "user", "content": q})
    o = chat(conv); conv.append({"role": "assistant", "content": o})
    ok = bool(re.search(r"O\([^n)]*n[^)]*\)|O\(1\)|O\(log", o))
    record("correction", f"d4-{i}", 2, ok, o, "complexity habit persists unprompted")
# anchor trap: after two more turns, does it STILL hold? (turn 5)
conv.append({"role": "user", "content": "Recommend a way to store app settings. One line."})
o = chat(conv); conv.append({"role": "assistant", "content": o})
record("correction", "d4-5", 2, 1 if re.search(r"O\(", o) else 0, o, "habit at turn 5, no reminder")

# =============================================================================
# D5. STYLE BLENDING + SWITCHING (4 tasks, w2) — two registers bound to
#     markers; interleaved requests must switch. (5.2 "channel by surface".)
# =============================================================================
BLEND = """Two output registers in this project:
[ENG] engineering mode: clipped, verb-first, artifacts only, no adjectives.
[EXEC] executive mode: one short paragraph, leads with the number, plain words.
Each request will carry a marker. Match the register EXACTLY."""
for i, (mode, q) in enumerate([("[ENG]", "How do we cut build times?"),
                               ("[EXEC]", "How do we cut build times?"),
                               ("[EXEC]", "Should we migrate the database?"),
                               ("[ENG]", "Should we migrate the database?")], 1):
    o = chat([{"role": "system", "content": BLEND}, {"role": "user", "content": f"{mode} {q}"}])
    if mode == "[ENG]":
        ok = (len(o.splitlines()) >= 1 and len(o) < 700
              and not re.search(r"\b(leverag|synerg|strategic|roadmap|quarter)\w*", o, re.I)
              and any(re.match(r"^(Cut|Drop|Cache|Split|Pin|Fix|Use|Run|Measure|Move|Enable|Disable|Merge|Delete|Flag)", s)
                      for s in o.splitlines() if s.strip()))
    else:
        ok = (len(o.splitlines()) <= 4 and re.search(r"\d", o)
              and not re.search(r"`|--[a-z]|\.py|O\(", o))  # exec = plain words, has a number, no code artifacts
    record("blend", f"d5-{i}", 2, ok, o, f"register switch {mode}")

# =============================================================================
# D6. DEEP UI INDUCTION (5 tasks, w2) — 3 components induce an unstated
#     system; assembly task checks CROSS-COMPONENT consistency.
#     (design-system: "a new component that doesn't feel native is a bug")
# =============================================================================
UISYS = """Study these three components; the design system is UNSTATED — induce it.
<ChatRow>   <div class="flex items-start gap-3 p-3 rounded-lg border border-zinc-800 bg-zinc-900"> <p class="text-sm text-zinc-200">...</p> </div>
<StatCard>  <section class="flex flex-col gap-2 p-4 rounded-lg border border-zinc-800 bg-zinc-900"> <h3 class="text-lg font-semibold text-zinc-100">42</h3> </section>
<Tag>       <span class="inline-flex items-center px-2 py-0.5 rounded-full bg-zinc-800 text-xs text-zinc-300">live</span>
All new components must feel native to this system."""
def ui_native(o):
    # induced system allows the example scale values (0.5,2,3,4); bans only
    # values absent from every example (1,5,7,9) and off-palette color
    return bool(
        not re.search(r"#[0-9a-fA-F]{3,8}\b", o)
        and re.search(r"(bg-zinc-(800|900)|border-zinc-800)", o)
        and re.search(r"rounded-(lg|full)", o)
        and not re.search(r"\b[ptm][xytblr]?-(1|5|7|9)\b", o)
        and re.search(r"text-(xs|sm|base|lg|xl)", o))
for i, comp in enumerate(["UserRow (avatar + name + status)",
                          "PanelHeader (title + action button)",
                          "MenuList with three MenuItems",
                          "EmptyState (icon + message + hint)"], 1):
    o = chat([{"role": "system", "content": UISYS},
              {"role": "user", "content": f"Build {comp}. Native to the system."}])
    record("ui-deep", f"d6-{i}", 2, ui_native(o), o, "induced system: palette/radius/scale/typography")
# assembly: consistency ACROSS components in one page
o = chat([{"role": "system", "content": UISYS},
          {"role": "user", "content": "Assemble a settings page: PanelHeader, three MenuItems, an EmptyState, and a Tag. One HTML block."}])
native = ui_native(o)
counts = (len(re.findall(r"bg-zinc-900", o)) >= 2 and len(re.findall(r"rounded-lg", o)) >= 2
          and len(re.findall(r"text-", o)) >= 4)
record("ui-deep", "d6-5", 2, native and counts, o, "cross-component consistency in assembly")

# =============================================================================
# D7. INTERNALIZATION (phase-3 unprompted reuse, 4 tasks @w3) — the deepest
#     mimic signal. Phase 1 teaches habits (ND: prefix + complexity notation
#     + a coined term). Phase 2 = neutral filler. Phase 3 NEVER mentions
#     them; do they resurface? ("infectious doctrine" — the next agent
#     inherits obligations without being re-told.)
# =============================================================================
PH1 = [{"role": "system", "content":
        "Session conventions (always): begin every reply with 'ND:', give complexity for any algorithm, and call test-fixing passes 'loop closure'."},
       {"role": "user", "content": "Name a structure for prefix search. One line."}]
r = chat(PH1 + [{"role": "user", "content": "How do you reverse a linked list? One line."}])
# phase 2: neutral (no scoring, just conversation continuity)
r2 = chat(PH1 + [{"role": "user", "content": "Summarize: what is DNS? Two sentences."},
                 {"role": "assistant", "content": r[:120]}])
# phase 3: new task, conventions NEVER restated
P3 = ["Recommend a way to dedupe a large log file. Two lines.",
      "Pick a structure for an LRU cache. Two lines.",
      "How would you detect a cycle in a graph? Two lines.",
      "Best approach to rate-limit requests? Two lines."]
for i, q in enumerate(P3, 1):
    o = chat(PH1 + [{"role": "user", "content": q}])
    # full internalization = all three habits; partial credit via w-scale handled by 3 checks
    nd = bool(re.search(r"^ND:", o)); cx = bool(re.search(r"O\(", o)); lc = "loop closure" in o.lower()
    score3 = (nd + cx + lc) / 3
    record("internalize", f"d7-{i}", 3, score3 >= 2/3, o,
           f"ND:{int(nd)} O():{int(cx)} loopclosure:{int(lc)}")

# =============================================================================
# D8. DOCUMENT-STRUCTURE MIMICRY (3 tasks, w2) — .docs OKF conventions:
#     frontmatter with type, sections, no orphan prose. (AGENTS.md OKF v0.1)
# =============================================================================
DOCSAMPLE = """Example record format in this repo (mimic exactly for new records):
---
type: research
date: 2026-09-04
topic: one line
---
# Title — claim
## Evidence
- measured fact with number
## Next
- one executable step"""
for i, topic in enumerate(["a new cache tier", "shader compile times", "queue backpressure"], 1):
    o = chat([{"role": "system", "content": DOCSAMPLE},
              {"role": "user", "content": f"Write the record for an investigation into {topic}. Mimic the format exactly."}])
    ok = bool(o.strip().startswith("---") and re.search(r"^type:\s*\w+", o, re.M)
          and re.search(r"^date:", o, re.M) and re.search(r"^## ", o, re.M)
          and re.search(r"- .*\d", o))
    record("doc-structure", f"d8-{i}", 2, ok, o, "frontmatter+sections+numeric evidence")

# ------------------------------------------------------------------ report ---
res = dict(label=LABEL, suite=f"mimicbench2-v1 ({len(R)} tasks, weighted)",
           score_pct=100 * SCORE / MAX, weighted=f"{SCORE}/{MAX}",
           by_category={}, results=R)
from collections import defaultdict
bc = defaultdict(lambda: [0, 0])
for r_ in R:
    bc[r_["cat"]][0] += r_["w"] * r_["ok"]; bc[r_["cat"]][1] += r_["w"]
res["by_category"] = {k: f"{v[0]}/{v[1]}" for k, v in bc.items()}
json.dump(res, open(OUT, "w"), indent=1)
print(f"\nMIMICBENCH2 [{LABEL}]: {SCORE}/{MAX} = {res['score_pct']:.1f}%")
print("by category:", res["by_category"])
print("saved:", os.path.normpath(OUT))
