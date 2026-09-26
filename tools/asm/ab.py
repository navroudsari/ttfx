#!/usr/bin/env python3
"""A/B the asm engine of two ttfx binaries on the speed.py workload.

Runs alternate between A and B (so drift on a shared machine hits both),
pinned to one core, and each effect reports the fastest of N runs, as CPU
time (user + system) by default or wall time with --wall, plus page faults.
The last line is the geometric mean of A/B: above 1 means B is faster.

Usage: ab.py A B [--runs N] [--core N] [--wall] [effect ...]
"""
import argparse
import math
import os
import subprocess
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CLOCKED = {"matrix", "thunderstorm"}

p = argparse.ArgumentParser()
p.add_argument("a")
p.add_argument("b")
p.add_argument("--runs", type=int, default=5)
p.add_argument("--core", default=os.environ.get("SPEED_CORE", "2"))
p.add_argument("--wall", action="store_true")
p.add_argument("effects", nargs="*")
o = p.parse_intermixed_args()

effects = o.effects or sorted(
    f[:-4] for f in os.listdir(os.path.join(ROOT, "asm/effects"))
    if f.endswith(".asm") and f != "registry.asm")
text = os.path.join(ROOT, "target/speed-input.txt")
line = ("The quick brown fox jumps over the lazy dog 0123456789 " * 4)[:190]
os.makedirs(os.path.dirname(text), exist_ok=True)
with open(text, "w") as f:
    f.write("\n".join(line for _ in range(46)))
WORKLOAD = ["--seed", "1", "--frame-rate", "0", "--canvas-width", "200",
            "--canvas-height", "50", "--ignore-terminal-dimensions"]


def run(binary, effect):
    args = ["taskset", "-c", o.core, binary]
    if effect in CLOCKED:
        args.append("--virtual-clock")
    start = time.perf_counter()
    with open(text) as stdin:
        child = subprocess.Popen(args + WORKLOAD + [effect], stdin=stdin,
                                 stdout=subprocess.DEVNULL,
                                 env=dict(os.environ, TTFX_ASM="force"))
        _, status, usage = os.wait4(child.pid, 0)
    wall = time.perf_counter() - start
    if os.waitstatus_to_exitcode(status) != 0:
        raise SystemExit(f"{binary} {effect} failed")
    ms = (wall if o.wall else usage.ru_utime + usage.ru_stime) * 1000
    return ms, usage.ru_minflt


logs = []
print(f"{'effect':16} {'A ms':>8} {'B ms':>8} {'A/B':>6} {'faults A':>9} {'faults B':>9}")
for effect in effects:
    ra, rb = [], []
    for _ in range(o.runs):
        ra.append(run(o.a, effect))
        rb.append(run(o.b, effect))
    a, b = min(r[0] for r in ra), min(r[0] for r in rb)
    logs.append(math.log(a / b))
    print(f"{effect:16} {a:8.1f} {b:8.1f} {a / b:6.3f} {ra[0][1]:9d} {rb[0][1]:9d}", flush=True)
print(f"geomean A/B {math.exp(sum(logs) / len(logs)):.4f} over {len(logs)} effects")
