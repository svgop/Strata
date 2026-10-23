#!/usr/bin/env python3
"""campaign/tpl-user-render-battery.py — offline parity battery for the user's candidate
chat template (Downloads/chat_template (4).jinja) against the standing F-279 one.

Renders both through serve/frontend.py's ChatTemplate (the exact production Jinja
environment: ImmutableSandboxedEnvironment, trim_blocks, lstrip_blocks, loopcontrols,
raise_exception global).  No engine, no port: pure render/raise parity plus the
structural diffs that matter for the live gates.  Exit 0 = battery ran (verdicts are
printed, not asserted — the adopt decision is the live gates').
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "serve"))
from frontend import ChatTemplate, TemplateRequestError  # noqa: E402

USER = Path("/home/svg/Downloads/chat_template (4).jinja")
STANDING = REPO / "serve/chat_template.jinja"


def render(tpl, **kw):
    """(ok, text-or-error).  add_generation_prompt defaults True as in production."""
    try:
        return True, tpl.render(kw.pop("messages"), tools=kw.pop("tools", None), **kw)
    except TemplateRequestError as exc:
        return False, str(exc)
    except Exception as exc:                                   # noqa: BLE001 — parity report wants everything
        return False, f"ENGINE-ERROR {type(exc).__name__}: {exc}"


user, standing = ChatTemplate(USER), ChatTemplate(STANDING)

print(f"caps user     : {user.caps}")
print(f"caps standing : {standing.caps}")
caps_verdict = "PARITY" if user.caps == standing.caps else "DIFF"
print(f"caps verdict  : {caps_verdict}\n")

U = {"role": "user", "content": "hello"}
SYS = {"role": "system", "content": "You are a helpful assistant."}
TOOL_PLAIN = [{"name": "get_weather", "description": "Weather lookup",
               "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}]
TOOL_WEB = [{"name": "web_search", "description": "Search the web",
             "parameters": {"type": "object", "properties": {"q": {"type": "string"}}}}]

cases = [
    # (label, kwargs)
    ("simple user, defaults", dict(messages=[U])),
    ("effort low", dict(messages=[U], reasoning_effort="low")),
    ("effort medium", dict(messages=[U], reasoning_effort="medium")),
    ("effort high", dict(messages=[U], reasoning_effort="high")),
    ("effort max", dict(messages=[U], reasoning_effort="max")),
    ("effort xhigh (never a render input; parity class only)",
     dict(messages=[U], reasoning_effort="xhigh")),
    ("effort banana (invalid)", dict(messages=[U], reasoning_effort="banana")),
    ("effort none (API accepts; render behavior)", dict(messages=[U], reasoning_effort="none")),
    ("thinking off", dict(messages=[U], enable_thinking=False)),
    ("thinking off + system", dict(messages=[SYS, U], enable_thinking=False)),
    ("system + user", dict(messages=[SYS, U])),
    ("developer role merged", dict(messages=[{"role": "developer", "content": "dev text"}, U])),
    ("tools plain (no retrieval keyword)", dict(messages=[U], tools=TOOL_PLAIN)),
    ("tools web (web_search: retrieval branch)", dict(messages=[U], tools=TOOL_WEB)),
    ("tool call round trip", dict(
        messages=[U, {"role": "assistant", "content": "",
                      "tool_calls": [{"function": {"name": "get_weather",
                                                   "arguments": {"city": "Paris"}}}]},
                 {"role": "tool", "content": "18C"}], tools=TOOL_PLAIN)),
    ("multi-turn with reasoning (default preserve)",
     dict(messages=[U, {"role": "assistant", "content": "hi", "reasoning_content": "greet->reply"}, U])),
    ("multi-turn with reasoning (preserve_thinking=true)",
     dict(messages=[U, {"role": "assistant", "content": "hi", "reasoning_content": "greet->reply"}, U],
          preserve_thinking=True)),
    ("inline think block stripped", dict(
        messages=[U, {"role": "assistant", "content": "<think>scratch</think>answer"}, U])),
    ("vision part", dict(messages=[{"role": "user", "content": [
        {"type": "text", "text": "what is this"},
        {"type": "image_url", "image_url": {"url": "x"}}]}], add_vision_id=True)),
    ("system mid-conversation (must raise)", dict(messages=[U, SYS, U])),
    ("empty messages (must raise)", dict(messages=[])),
]

for label, kw in cases:
    uo, ut = render(user, **kw)
    so, st = render(standing, **kw)
    if not uo and not so:
        r = f"both RAISE ok ({'user: ' + ut[:60]!r} | standing: {st[:60]!r})"
    elif uo and so:
        if ut == st:
            r = "IDENTICAL render"
        else:
            r = (f"both render, DIFFER (user {len(ut)} ch / standing {len(st)} ch)")
    else:
        r = f"ASYMMETRY — user {'OK' if uo else 'RAISE ' + ut[:80]!r} vs standing {'OK' if so else 'RAISE ' + st[:80]!r}"
    print(f"{label:58s} {r}")

# structural sanity on the mainline case: what each template actually emits up front
uo, ut = render(user, messages=[U])
so, st = render(standing, messages=[U])
print("\n--- user template, simple-user head (first 400 ch) ---")
print(ut[:400].replace("\n", "\\n"))
print("--- standing template, simple-user head (first 400 ch) ---")
print(st[:400].replace("\n", "\\n"))

# system-block token-ish overhead of the always-on protocol vs the one sentence
def sys_block(t):
    head = t.split("<|im_end|>")[0]
    return head

ub, sb = sys_block(ut), sys_block(st)
print(f"\nsystem block chars : user {len(ub)} vs standing {len(sb)} "
      f"(delta {len(ub) - len(sb):+d} ch ≈ {(len(ub) - len(sb)) / 4:+.0f} tok)")
uo, utl = render(user, messages=[U], reasoning_effort="low")
print(f"effort-low system block chars : user {len(sys_block(utl))}")
print("battery done")
