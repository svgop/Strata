#!/usr/bin/env python3
"""F-459 - which functions of a translation unit changed machine code between two builds.

Usage: unset-path-diff.py <baseline.o> <candidate.o>

Disassembles both objects (objdump -dr -C) and compares every function's instruction list after removing
what a rebuild moves regardless of the source:
  - instruction addresses; a branch inside the function is resolved to the index of its target;
  - callee and data addresses: the relocation entry names the symbol (its addend is the PC bias), and a
    section symbol loses its offset;
  - the assembler's local labels (.LCn): each is resolved to its section and its bytes.
Member displacements (0x218(%rax)) stay: they are the struct offsets a change could move.

Classes: identical | different | only in one build.  A different function is listed with its differing
instructions (+ candidate, - baseline), at most HUNKS hunks with context.
"""
import difflib
import re
import subprocess
import sys

HUNKS = 6
HEAD = re.compile(r"^([0-9a-f]+) <(.+)>:$")
INSN = re.compile(r"^\s*([0-9a-f]+):\s+(\S+)\s*(.*?)\s*$")
RELOC = re.compile(r"^\s*(?:[0-9a-f]+:\s*)?(R_X86_64_\w+)\s+(.+?)\s*$")     # `offset: R_X86_64_..  symbol+addend`
ADDEND = re.compile(r"[+-]0x[0-9a-f]+$")
SECOFF = re.compile(r"(\.\w[\w.]*?)[+-]0x[0-9a-f]+")                      # a section symbol with an offset
ANNOT = re.compile(r"(0x[0-9a-f]+|[0-9a-f]+)\s+<(.*)>$")                  # objdump's `address <symbol>` annotation
OFFSET = re.compile(r"\+0x[0-9a-f]+$")
RIP = re.compile(r"0x[0-9a-f]+\(%rip\)")
LOCAL = re.compile(r"^([0-9a-f]+)\s+l\s+(\S+)\s+([0-9a-f]+)\s+(\.LC\d+)$")
SECHEAD = re.compile(r"^Contents of section (\S+):$")


def sh(args):
    return subprocess.run(args, capture_output=True, text=True, check=True).stdout


def constants(path):
    """The assembler's local labels -> `section:bytes` (a string runs to its NUL, a .cstN entry is N bytes)."""
    secs, cur = {}, None
    for line in sh(["objdump", "-s", path]).splitlines():
        m = SECHEAD.match(line)
        if m:
            cur = secs.setdefault(m.group(1), {})
            continue
        r = line.split("  ", 1)[0].split() if cur is not None else []   # the ASCII column follows a double space
        if r and re.fullmatch(r"[0-9a-f]+", r[0]) and all(len(g) % 2 == 0 and re.fullmatch(r"[0-9a-f]+", g) for g in r[1:]):
            cur[int(r[0], 16)] = bytes.fromhex("".join(r[1:]))
    out = {}
    for line in sh(["objdump", "-t", path]).splitlines():
        m = LOCAL.match(line)
        if not m:
            continue
        value, sec, name = int(m.group(1), 16), m.group(2), m.group(4)
        rows = secs.get(sec)
        if not rows:
            continue
        start = min(rows)
        blob = b"".join(rows[o] for o in sorted(rows))
        at = value - start
        if at < 0 or at >= len(blob):
            continue
        cst = re.match(r"\.rodata\.cst(\d+)$", sec)
        n = int(cst.group(1)) if cst else (blob.index(b"\0", at) - at if sec.startswith(".rodata.str") else 8)
        out[name] = "%s:%s" % (sec, blob[at:at + n].hex())
    return out


def functions(path):
    """name -> the function's instruction list, each `mnemonic operands` with layout removed."""
    lab = constants(path)
    out = sh(["objdump", "-dr", "-C", "--no-show-raw-insn", path])
    funcs, name, rows = {}, None, []

    def resolve(text):
        return re.sub(r"\.LC\d+", lambda m: "@" + lab.get(m.group(0), m.group(0)), text)

    def close(rows):
        index = {addr: k for k, (addr, _, _, _) in enumerate(rows)}   # this function's addresses -> instruction index
        body = []
        for _, mnem, ops, reloc in rows:
            if reloc:                                   # the symbol is the relocation's, never the printed address's
                ops = "REL " + reloc
            else:
                m = ANNOT.search(ops)                   # the annotation ends the operand
                if m:
                    addr, sym = m.group(1), OFFSET.sub("", m.group(2))
                    ops = ops[:m.start()] + ("L%d" % index[addr] if addr in index else "<%s>" % sym)
                ops = RIP.sub("RIP(%rip)", ops)
            body.append(resolve(mnem + " " + ops))
        return body

    for line in out.splitlines():
        h = HEAD.match(line)
        if h:
            if name is not None:
                funcs[name] = close([tuple(x) for x in rows])
            name, rows = h.group(2), []
            continue
        if name is None:
            continue
        r = RELOC.match(line)                           # belongs to the instruction above; tested first (it has an offset too)
        if r and rows:
            rows[-1][3] = r.group(1) + " " + SECOFF.sub(r"\1", ADDEND.sub("", r.group(2)))
            continue
        i = INSN.match(line)
        if i:
            rows.append([i.group(1), i.group(2), i.group(3), ""])
    if name is not None:
        funcs[name] = close([tuple(x) for x in rows])
    return funcs


def main(argv):
    if len(argv) != 3:
        sys.stderr.write(__doc__)
        return 2
    a, b = functions(argv[1]), functions(argv[2])
    names = sorted(set(a) | set(b))
    same = [n for n in names if n in a and n in b and a[n] == b[n]]
    differ = [n for n in names if n in a and n in b and a[n] != b[n]]
    only_a = [n for n in names if n in a and n not in b]
    only_b = [n for n in names if n in b and n not in a]
    print("baseline: %s" % argv[1])
    print("candidate: %s" % argv[2])
    print("functions: %d  | identical %d  | different %d  | baseline only %d  | candidate only %d"
          % (len(names), len(same), len(differ), len(only_a), len(only_b)))
    for n in differ:
        al, bl = a[n], b[n]
        sm = difflib.SequenceMatcher(None, al, bl, autojunk=False)
        ops = sm.get_opcodes()
        added = sum(j2 - j1 for tag, i1, i2, j1, j2 in ops if tag != "equal")
        removed = sum(i2 - i1 for tag, i1, i2, j1, j2 in ops if tag != "equal")
        print("\nDIFFERENT  %s   instructions %d -> %d   (+%d / -%d)" % (n, len(al), len(bl), added, removed))
        for tag, i1, i2, j1, j2 in [o for o in ops if o[0] != "equal"][:HUNKS]:
            for k in range(max(0, i1 - 2), i1):
                print("      =  %s" % al[k])
            for k in range(i1, i2):
                print("      -  %s" % al[k])
            for k in range(j1, j2):
                print("      +  %s" % bl[k])
            for k in range(i2, min(len(al), i2 + 2)):
                print("      =  %s" % al[k])
    for tag, lst in (("BASELINE ONLY", only_a), ("CANDIDATE ONLY", only_b)):
        if lst:
            print("\n%s (%d):" % (tag, len(lst)))
            for n in lst:
                print("    " + n)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
