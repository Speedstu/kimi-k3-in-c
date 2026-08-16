from pathlib import Path

p = Path('README.md')
s = p.read_text(encoding='utf-8')

old = '''- **measured thread tuning** via `--threads N` and `benchmarks/thread-sweep.sh` instead of
  assuming that all logical CPUs are fastest;'''
new = '''- **real-hardware CPU + I/O autotuning** via `benchmarks/autotune.py`: it measures
  `--threads N` together with `K3_ASYNC_IO_THREADS=M` on the actual CPU/NVMe path and
  refuses to recommend anything if the exact token stream changes;'''
if old not in s:
    raise SystemExit('README tuning bullet not found')
s = s.replace(old, new, 1)

old = '''# 2. Find the best compute thread count on THIS machine/workload.
K3_SWEEP_REPEATS=3 benchmarks/thread-sweep.sh ~/k3model \\
  --trunk ~/k3trunk-lossless --preset laptop --incremental \\
  --ids 1008,10484,318,15383,387 --gen 4

# 3. Re-run with the recommended --threads N.
./bin/k3 ~/k3model --trunk ~/k3trunk-lossless --preset laptop \\
  --incremental --threads N \\
  --ids 1008,10484,318,15383,387 --gen 8'''
new = '''# 2. Tune BOTH compute and async expert-I/O threads on THIS machine/storage path.
#    Use a short deterministic request so the sweep is affordable.
python3 benchmarks/autotune.py ~/k3model --repeats 2 -- \\
  --trunk ~/k3trunk-lossless --preset laptop --incremental \\
  --ids 1008,10484,318,15383,387 --gen 2 --temperature 0

# 3. Re-run with the recommended --threads N and K3_ASYNC_IO_THREADS=M.
K3_ASYNC_IO_THREADS=M ./bin/k3 ~/k3model --trunk ~/k3trunk-lossless --preset laptop \\
  --incremental --threads N \\
  --ids 1008,10484,318,15383,387 --gen 8'''
if old not in s:
    raise SystemExit('README recommended setup block not found')
s = s.replace(old, new, 1)

old = '''K3_NOASYNC_PREFETCH=1   # compare against the old synchronous caller behavior
K3_ASYNC_IO_THREADS=4   # background expert-read OpenMP team (default 4)'''
new = '''K3_NOASYNC_PREFETCH=1   # compare against the old synchronous caller behavior
K3_ASYNC_IO_THREADS=4   # background expert-read team; autotune this on real hardware'''
if old not in s:
    raise SystemExit('README async controls block not found')
s = s.replace(old, new, 1)

old = '''[`ASYNC_EXPERT_PREFETCH.md`](docs/ASYNC_EXPERT_PREFETCH.md) ·
[`THREAD_TUNING.md`](docs/THREAD_TUNING.md)'''
new = '''[`ASYNC_EXPERT_PREFETCH.md`](docs/ASYNC_EXPERT_PREFETCH.md) ·
[`AUTOTUNE.md`](docs/AUTOTUNE.md) ·
[`THREAD_TUNING.md`](docs/THREAD_TUNING.md)'''
if old not in s:
    raise SystemExit('README docs links not found')
s = s.replace(old, new, 1)

p.write_text(s, encoding='utf-8')
