#!/usr/bin/env python3
"""Lay out the engine's machine code around the JCC erratum.

On Intel's Skylake-derived cores (Skylake through Cascade Lake, Coffee Lake,
Comet Lake) the microcode fix for the jump conditional code erratum keeps a
32-byte chunk of code out of the decoded uop cache when a branch in it
crosses or ends on a 32-byte boundary; a macro-fused compare+branch pair
counts as one branch. Such chunks go through the legacy decoders every time
they run. GNU as can pad around this (-mbranches-within-32B-boundaries);
NASM cannot, so this does it on NASM's preprocessed output:

  1. nasm -E: one flat source file, every macro and include expanded;
  2. assemble it with a listing, which gives every line's offset and length;
  3. walk the code in order and push each branch that crosses or ends on a
     boundary to the next one, with redundant DS segment prefixes (0x3E,
     ignored in 64-bit mode) on the plain integer instructions just before
     it, or a multi-byte NOP where they can't absorb it all;
  4. reassemble and repeat until nothing is left (a moved jump can change
     between its short and near forms).

Prefixes are architecturally no-ops: the program is the same instructions,
only placed differently. Usage:

  jcc-pad.py NASM OUTPUT.o [nasm args...] SOURCE.asm

writes the object and a one-line summary on stderr.
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile

BRANCH = re.compile(r"^(j[a-z]+|call|ret)$")
FUSIBLE = {"cmp", "test", "add", "sub", "and", "inc", "dec"}
# Plain integer instructions that take a redundant DS prefix. Anything else
# (vector, VEX/EVEX, string, branch, x87, system) is left alone.
PREFIXABLE = {
    "mov", "movzx", "movsx", "movsxd", "lea", "add", "sub", "and", "or", "xor",
    "cmp", "test", "inc", "dec", "neg", "not", "shl", "shr", "sar", "rol",
    "ror", "imul", "adc", "sbb", "bt", "bts", "btr", "cmovz", "cmove", "cmovnz",
    "cmovne", "cmova", "cmovae", "cmovb", "cmovbe", "cmovg", "cmovge", "cmovl",
    "cmovle", "cmovc", "cmovnc", "cmovs", "cmovns",
}
MAX_PREFIXES = 3            # added per instruction: more slow some decoders
MAX_LEN = 15
NOPS = {1: "90", 2: "6690", 3: "0F1F00", 4: "0F1F4000", 5: "0F1F440000",
        6: "660F1F440000", 7: "0F1F8000000000", 8: "0F1F840000000000"}
ALIGN = re.compile(r"^\s*times \(\(\((\d+)\) - \(\(\$-\$\$\) %")


def parse_listing(path):
    """listing line number -> (offset, length) for lines that emit bytes."""
    out = {}
    for rec in open(path, errors="replace"):
        m = re.match(r"^\s*(\d+) ([0-9A-F]{8}) (\S+)", rec)
        if not m:
            continue
        line, off = int(m.group(1)), int(m.group(2), 16)
        data = m.group(3).rstrip("-")
        if data.startswith("<"):                    # <rep N> of a times line
            continue
        n = len(re.sub(r"[^0-9A-F]", "", data.split("<")[0])) // 2
        if line in out:
            o, l = out[line]
            out[line] = (o, max(l, off + n - o))
        else:
            out[line] = (off, n)
    return out


def mnemonic(text):
    t = text.split(";")[0].strip()
    t = re.sub(r"^[.\w$@?]+:\s*", "", t)            # a label on the same line
    return t.split(None, 1)[0].lower() if t else ""


def text_entries(lines, listing):
    """The .text lines that emit bytes or align: (line index, offset, length,
    mnemonic, alignment or 0)."""
    section = None
    text = []
    for i, line in enumerate(lines):
        m = re.match(r"^\s*\[?section\s+([.\w]+)", line, re.I)
        if m:
            section = m.group(1).lower()
            continue
        if section != ".text" or i not in listing:
            continue
        off, n = listing[i]
        m = ALIGN.match(line)
        if n or m:
            text.append((i, off, n, mnemonic(line), int(m.group(1)) if m else 0))
    return text


def affected(start, end):
    return start // 32 != (end - 1) // 32 or end % 32 == 0


def plan_pass(text, prefixes, extra, nops):
    """Walk the code in order, tracking how far the padding placed so far
    moves what follows (an alignment directive re-aligns it), and push every
    branch that crosses or ends on a boundary at its moved position to the
    next boundary. Returns how many branches were padded."""
    shift = 0
    padded = 0
    moved = []
    for k, (i, off, n, mn, align) in enumerate(text):
        at = off + shift
        moved.append(at)
        if align:
            shift += (-at) % align - n
            continue
        if not BRANCH.match(mn):
            continue
        start, first = at, k
        if mn.startswith("j") and mn != "jmp" and k > 0:
            pi, poff, pn, pmn, pal = text[k - 1]
            if pmn in FUSIBLE and poff + pn == off:
                start, first = moved[k - 1], k - 1
        if not affected(start, at + n):
            continue
        need = left = 32 - start % 32
        # prefixes on the straight-line instructions before it, nearest first
        j = first - 1
        while left and j >= 0 and first - j <= 8:
            pi, poff, pn, pmn, pal = text[j]
            if pal or text[j + 1][1] != poff + pn or pmn not in PREFIXABLE:
                break
            have = prefixes.get(pi, 0)
            room = min(MAX_PREFIXES - have, MAX_LEN - pn, left)
            if room > 0:
                prefixes[pi] = have + room
                extra[pi] = "3E" * room + extra.get(pi, "")
                left -= room
            j -= 1
        # a NOP right before the branch (or its compare) for the rest
        fi = text[first][0]
        while left:
            m = min(left, 8)
            nops[fi] = nops.get(fi, "") + NOPS[m]
            left -= m
        shift += need
        padded += 1
    return padded


def main():
    nasm, output = sys.argv[1], sys.argv[2]
    args, source = sys.argv[3:-1], sys.argv[-1]
    work = tempfile.mkdtemp(prefix="jcc-pad.", dir=os.path.dirname(os.path.abspath(output)))
    try:
        flat = os.path.join(work, "flat.asm")
        obj = os.path.join(work, "flat.o")
        lst = os.path.join(work, "flat.lst")
        subprocess.run([nasm, *args, "-E", "-o", flat, source], check=True)
        lines = [l for l in open(flat) if not l.startswith("%line")]
        # the flat source needs no include paths
        nargs, skip = [], False
        for a in args:
            if skip:
                skip = False
            elif a == "-I":
                skip = True
            elif not a.startswith("-I"):
                nargs.append(a)
        # line index -> prefix bytes, prefix count, NOP bytes before it
        extra, prefixes, nops = {}, {}, {}
        passes = total = 0
        while True:
            passes += 1
            out, index = [], []     # output line number -> source line index

            def db(hexs):
                return "    db " + ", ".join(
                    "0x" + hexs[j:j + 2] for j in range(0, len(hexs), 2)) + "\n"
            for i, l in enumerate(lines):
                if i in nops:
                    out.append(db(nops[i]))
                    index.append(None)              # a NOP of its own
                if i in extra:
                    out.append(db(extra[i]))
                    index.append(-1 - i)            # line i's prefixes
                out.append(l)
                index.append(i)
            open(flat, "w").writelines(out)
            subprocess.run([nasm, *nargs, "-w-all", "-l", lst, "-o", obj, flat], check=True)
            listing = parse_listing(lst)
            # each source line's offset and length, its prefixes included
            merged = {}
            for n, i in enumerate(index):
                if i is None or n + 1 not in listing:
                    continue
                off, ln = listing[n + 1]
                if i < 0:
                    merged[-1 - i] = (off, -ln)     # prefixes: the line follows
                elif i in merged and merged[i][1] <= 0:
                    merged[i] = (merged[i][0], ln - merged[i][1])
                else:
                    merged[i] = (off, ln)
            text = text_entries(lines, merged)
            padded = plan_pass(text, prefixes, extra, nops)
            if not padded or passes >= 20:
                break
            total += padded
        shutil.move(obj, output)
        nop_bytes = sum(len(v) // 2 for v in nops.values())
        print(f"jcc-pad: {passes} passes, {total} branches moved with "
              f"{sum(prefixes.values())} prefixes and {nop_bytes} NOP bytes; "
              f"{padded} still affected", file=sys.stderr)
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
