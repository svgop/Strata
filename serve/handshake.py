"""serve/handshake.py - Layer 2's control plane: the versioned engine<->server handshake.

The engine's boot output was a bag of `INFO key=value` lines, regexed by every consumer that
met them.  The handshake replaces that for identity and armed geometry: at boot the engine
emits one `IDENT <json>` line beside the INFO lines, and THIS module is its single schema
owner on the serve side - parse, validate, refuse.  An engine built before the handshake (or
one whose IDENT is not valid JSON) degrades to the INFO line protocol through the same
verdict function: one gate, two wire formats, no third parser.  The INFO `cvec=` line stays
on the wire for older servers during the migration (strangler-fig, never a cutover).

v1 schema (what the engines emit, src/program/generate.cpp and src/dsv41/ds4_model.cu):
    IDENT {"v":1, "engine":"strata"|"ds4", "version":"<build>", "ctx":<int>, "layers":<int>,
           "cvec":{"mode":"project"|"add", "first":<int>, "last":<int>, "steered":<int>},
           "caps":[...]}
`layers` is model identity (top level); `cvec` is present only when a vector armed (absence
= none armed).  `steered` is the engine's MEASURED truth - how many layers actually steer -
which the gate checks against the model's own `layers`, instead of the asked range the
engine would parrot.
"""

import json
import re

HANDSHAKE_VERSION = 1

# the legacy wire format: `INFO cvec=<mode>:<first>-<last>` (+ an optional ":singleN" suffix)
# and `INFO cvec_steered=<steered>/<layers>`, as printed by every engine before the handshake
_INFO_CVEC = re.compile(r"(\w+):(\d+)-(\d+)")


class _Garbled(Exception):
    """The legacy protocol reported something it cannot have armed; the message is the verdict."""

    def __init__(self, why: str):
        super().__init__(why)
        self.why = why


def parse_ident(line: str) -> dict | None:
    """`IDENT {json}` -> the parsed object, or None when the line is not valid JSON (the
    caller degrades to the INFO line protocol with one warning - a boot is never lost to a
    malformed handshake)."""
    try:
        obj = json.loads(line[len("IDENT "):])
    except (ValueError, TypeError):
        return None
    return obj if isinstance(obj, dict) else None


def _ident_geometry(ident: dict) -> tuple | None:
    """The IDENT's armed geometry as the tuple the gate reasons over, or None when no vector
    armed.  `layers` is model identity and lives at the IDENT's top level; the cvec object
    carries what the vector does (mode/first/last/steered).  A present-but-malformed report
    raises _Garbled: a v1 engine with a broken emitter is a refusal, not a silent pass."""
    cv = ident.get("cvec")
    if cv is None:
        return None
    if not isinstance(cv, dict):
        raise _Garbled(f"the engine's IDENT handshake reports a malformed control vector "
                       f"(cvec is {type(cv).__name__}, not an object); refusing to serve on an unreadable arm")
    try:
        mode, first, last, steered, layers = cv["mode"], cv["first"], cv["last"], cv["steered"], ident["layers"]
    except KeyError as e:
        raise _Garbled(f"the engine's IDENT handshake reports a malformed control vector "
                       f"(missing {e}); refusing to serve on an unreadable arm")
    if not isinstance(mode, str) or not mode or not all(isinstance(x, int) for x in (first, last, steered, layers)):
        raise _Garbled(f"the engine's IDENT handshake reports a malformed control vector "
                       f"(cvec={cv!r}); refusing to serve on an unreadable arm")
    return mode, first, last, steered, layers


def _info_geometry(info: dict) -> tuple | None:
    """The INFO line protocol's cvec, _ident_geometry's mirror: None = nothing armed, a
    5-tuple = parsed geometry (steered/layers None when the engine predates the truth key)."""
    reported = info.get("cvec")
    if reported in (None, "0"):
        return None
    m = _INFO_CVEC.fullmatch(str(reported).split(":single")[0])
    if m is None:
        raise _Garbled(f"the engine reported a control vector shape it cannot have armed: {reported!r}")
    mode, first, last = m.group(1), int(m.group(2)), int(m.group(3))
    steered = layers = None
    st = info.get("cvec_steered")
    if isinstance(st, str) and st.count("/") == 1:
        try:
            steered, layers = (int(x) for x in st.split("/"))
        except ValueError:
            raise _Garbled(f"the engine reported cvec_steered={st!r}, which is not a count")
    return mode, first, last, steered, layers


def _asked_geometry(args: list):
    """What the config asked for: (asks, want_mode, want_first, want_last).  The mode
    defaults to project; the range is only known when the flag carries a parsable one."""
    asks = "--control-vector-scaled" in args
    want_mode = "project"
    if "--cvec-mode" in args and args.index("--cvec-mode") + 1 < len(args):
        want_mode = args[args.index("--cvec-mode") + 1]
    want_first = want_last = None
    if "--control-vector-layer-range" in args and args.index("--control-vector-layer-range") + 2 < len(args):
        try:
            want_first = int(args[args.index("--control-vector-layer-range") + 1])
            want_last = int(args[args.index("--control-vector-layer-range") + 2])
        except ValueError:
            want_first = want_last = None          # unparsable: the range stays unchecked, the mode still holds
    return asks, want_mode, want_first, want_last


def armed_geometry_problem(engine, args: list) -> str | None:
    """The Layer-2 gate: the geometry the engine ARMED must equal the geometry the config
    ASKED for, checked on the engine's measured truth.  An engine that silently armed its
    own default range - or armed nothing where steering was asked - is the inert-vector
    class (the 4..24 war story): a refused boot with the reason, never a silent no-op.
    Prefers the engine's IDENT handshake; an engine without one (or a future version)
    degrades to the INFO line protocol.  -> None when consistent, else the why."""
    ident = getattr(engine, "ident", None)
    if isinstance(ident, dict) and ident.get("v") == HANDSHAKE_VERSION:
        structured = True
    else:
        if isinstance(ident, dict) and "v" in ident:
            print(f"[strata] WARNING: engine handshake version {ident.get('v')!r}, this server speaks "
                  f"{HANDSHAKE_VERSION} - using the INFO line protocol", flush=True)
        structured = False
    asks, want_mode, want_first, want_last = _asked_geometry(args)
    if not asks:
        # warn about an unexpected arm, never refuse - steering was never part of this config's contract.
        # (Checked before parsing, exactly as the INFO-protocol gate did: garbage from an engine whose
        # config asked nothing is a warning, not a dead server.)
        shape = None
        if structured:
            try:
                geom = _ident_geometry(ident)
                shape = None if geom is None else f"{geom[0]}:{geom[1]}-{geom[2]}"
            except _Garbled:
                shape = repr(ident.get("cvec"))    # an arm the schema cannot read: say what arrived
        else:
            reported = (getattr(engine, "info", {}) or {}).get("cvec")
            shape = f"{reported}" if reported and reported not in ("0", 0) else None
        if shape:
            print(f"[strata] WARNING: the engine armed a control vector the config did not ask for ({shape})",
                  flush=True)
        return None
    try:
        geom = _ident_geometry(ident) if structured else _info_geometry(getattr(engine, "info", {}) or {})
    except _Garbled as e:
        return e.why
    if geom is None:
        return ("the config asks for a control vector (--control-vector-scaled) but the engine armed none - "
                "the vector file failed to load, or the engine ignored the flag")
    mode, first, last, steered, layers = geom
    if mode != want_mode:
        return (f"the engine armed {mode}:{first}-{last} but the config asked for mode {want_mode} - "
                "steering is NOT what was requested; refusing to serve")
    if want_first is not None and (first != want_first or last != want_last):
        return (f"the engine armed {mode}:{first}-{last} but the config asked for "
                f"{want_mode}:{want_first}-{want_last} - steering is NOT what was requested; refusing to serve")
    if layers is None:
        return None                                  # an old engine without the truth key: the shape was checked
    # the engine's MEASURED truth: S of N layers actually steered.  A range running past the model's
    # layers steers nothing there (the engine reports the asked range unclamped); zero steered with a
    # vector asked is the inert load.  Both are refused on the engine's own numbers, not its parroting.
    if last >= layers:
        return (f"the control-vector range {mode}:{first}-{last} runs past the model's {layers} layers - "
                f"layers {layers}..{last} steer nothing; refusing to serve a misconfigured range")
    if steered is None or steered <= 0:
        return ("the control vector loaded but steers 0 layers (cvec_steered=0) - it is inert; "
                "refusing to serve steering that does nothing")
    return None
