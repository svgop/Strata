#!/usr/bin/env python3
"""Mechanical return-path audit (the F-441/F-447 law): for each named function, the last
statement before the depth-0 closing brace must be a `return`.  Non-void extractions that fall
off their end are UB - this audit exists because that mistake was made twice.

Usage: return_audit.py <file> <function> [<function>...]
Exit 1 if any listed function's tail is not a return."""
import sys

def audit(path, names):
    lines = open(path).read().split('\n')
    bad = []
    for name in names:
        start = None
        for i, l in enumerate(lines):
            if name + '(' in l and not l.strip().startswith('//'):
                # definition-like: the line (or a following line) opens the body brace
                depth, j = 0, i
                while '{' not in lines[j] and j < i + 12:
                    j += 1
                start, j = j, j
                break
        if start is None:
            bad.append(f"{name}: NOT FOUND")
            continue
        depth, last = 0, ''
        for j in range(start, len(lines)):
            for ch in lines[j]:
                if ch == '{': depth += 1
                elif ch == '}': depth -= 1
            if depth == 0 and j > start:
                if 'return' not in last:
                    bad.append(f"{name}: TAIL-NOT-RETURN ({last[:60]})")
                break
            if lines[j].strip():
                last = lines[j].strip()
    for b in bad: print(b)
    return 1 if bad else 0

if __name__ == '__main__':
    sys.exit(audit(sys.argv[1], sys.argv[2:]))
