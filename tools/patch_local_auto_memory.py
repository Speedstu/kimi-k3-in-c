#!/usr/bin/env python3
"""Stage machine-aware memory defaults for the local resident K3 service.

The model math is untouched. This only replaces the historical 3 GB trunk / 1 GB cache
startup default with a host-memory-aware split that remains trunk-first and reserves RAM
for the exact model head/state/prefill/KV working set.
"""
from pathlib import Path

p = Path("local/k3_local.py")
s = p.read_text()

s = s.replace("import argparse\nimport atexit\n", "import argparse\nimport atexit\nimport ctypes\n", 1)
s = s.replace('    preset: str = "laptop"', '    preset: str = "auto"', 1)
s = s.replace('    sp.add_argument("--preset", default="laptop")', '    sp.add_argument("--preset", default="auto")', 1)

anchor = '''class ResidentCBackend:\n    """One warm C process: weights/index/trunk/cache and the active KV/KDA state stay live."""\n\n'''
helper = r'''# Exact resident-memory policy.  This is deliberately kept in Python because the local
# bridge already owns process startup and can query the host before the huge C worker is
# created.  The values below are storage/residency budgets, never model approximations.
#
# Why trunk-first? Released-K3 measurements show every low-memory token sweeps ~108.81 GB
# of dense trunk but only ~25.83 GB of routed experts; expert LRU retention is negligible
# below tens of GB. The old local-service default (3/1) therefore left most of a 32 GB
# laptop unused for the one cache where every extra resident layer saves guaranteed I/O.
_K3_AUTO_FULL_TRUNK_GB = 111.0
_K3_AUTO_CACHE_FLOOR_GB = 0.5
_K3_AUTO_TRUNK_FLOOR_GB = 2.5
_K3_EXACT_HEAD_GB = 4.70
_K3_RECURRENT_AND_MISC_GB = 0.75
_K3_KV_BYTES_PER_POSITION = 2_370_000.0
_K3_AUTO_KV_HOT_POSITIONS = 2048


def _available_physical_memory_bytes() -> int:
    """Best-effort currently available physical memory, cross-platform, no dependency."""
    if os.name == "nt":
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]
        state = MEMORYSTATUSEX()
        state.dwLength = ctypes.sizeof(state)
        try:
            ok = ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(state))
        except (AttributeError, OSError):
            ok = 0
        return int(state.ullAvailPhys) if ok else 0

    try:
        with open("/proc/meminfo", "r", encoding="ascii") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass

    # Portable fallback.  On platforms where AVPHYS_PAGES is unavailable, return zero
    # rather than guessing from total RAM and risking swap/reclaim on a 1.56 TB workload.
    try:
        pages = int(os.sysconf("SC_AVPHYS_PAGES"))
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        if pages > 0 and page_size > 0:
            return pages * page_size
    except (AttributeError, OSError, ValueError):
        pass
    return 0


def _auto_worker_budgets(
    available_gb: float,
    *,
    prefill_mb: float = 256.0,
    worker_context: int = 1024,
) -> tuple[float, float]:
    """Return exact trunk/cache budgets without driving the machine into reclaim.

    `available_gb` is memory available BEFORE worker startup.  Reserve the model head,
    recurrent/misc state, configured transient prefill, a bounded hot KV working set and
    a host safety margin.  Everything left goes to the exact trunk until it is fully
    resident; only then does extra memory feed the expert cache.

    The full worker can reserve a 1M-position KV virtually, so reserving all configured
    positions here would incorrectly require terabytes at startup.  Physical KV pages are
    demand-committed.  We reserve up to 2048 hot positions (~4.85 GB) and leave an
    additional 8%/1.5 GB host margin; long-context requests remain governed by the
    worker's demand paging and request/context checks.
    """
    if not (available_gb > 0.0):
        raise ValueError("available_gb must be > 0")
    if not (prefill_mb > 0.0):
        raise ValueError("prefill_mb must be > 0")
    if worker_context < 2:
        raise ValueError("worker_context must be >= 2")

    hot_positions = min(int(worker_context), _K3_AUTO_KV_HOT_POSITIONS)
    kv_hot_gb = hot_positions * _K3_KV_BYTES_PER_POSITION / 1e9
    prefill_gb = prefill_mb * (1024.0 * 1024.0) / 1e9
    host_margin = max(1.5, 0.08 * available_gb)
    fixed = _K3_EXACT_HEAD_GB + _K3_RECURRENT_AND_MISC_GB + prefill_gb + kv_hot_gb
    allocator = available_gb - fixed - host_margin

    # On a heavily loaded or genuinely tiny machine, do not pretend an aggressive auto
    # plan is safe. The historical 3/1 preset remains available explicitly as `laptop`.
    minimum = _K3_AUTO_TRUNK_FLOOR_GB + _K3_AUTO_CACHE_FLOOR_GB
    if allocator < minimum:
        raise RuntimeError(
            f"auto memory has only {allocator:.1f} GB left for trunk/cache after exact "
            f"fixed costs and safety margin; need at least {minimum:.1f} GB. "
            "Close memory-heavy apps or pass --preset laptop explicitly."
        )

    if allocator >= _K3_AUTO_FULL_TRUNK_GB + _K3_AUTO_CACHE_FLOOR_GB:
        trunk = _K3_AUTO_FULL_TRUNK_GB
        cache = allocator - trunk
    else:
        cache = _K3_AUTO_CACHE_FLOOR_GB
        trunk = allocator - cache

    # Stable command lines/logs, while leaving sub-100 MB precision irrelevant to layer
    # residency and O_DIRECT slots.
    return round(trunk, 2), round(cache, 2)


class ResidentCBackend:
    """One warm C process: weights/index/trunk/cache and the active KV/KDA state stay live."""

'''
if anchor not in s:
    raise SystemExit("ResidentCBackend anchor not found")
s = s.replace(anchor, helper, 1)

old = '''    def _budgets(self) -> tuple[float, float]:\n        defaults = self._PRESET_BUDGETS.get(self.cfg.preset)\n        if defaults is None:\n            raise ValueError(\n                f"resident worker cannot resolve preset {self.cfg.preset!r}; "\n                "use a named fixed preset or explicit --trunk-gb/--cache-gb"\n            )\n        trunk = self.cfg.trunk_gb if self.cfg.trunk_gb is not None else defaults[0]\n        cache = self.cfg.cache_gb if self.cfg.cache_gb is not None else defaults[1]\n        return float(trunk), float(cache)\n'''
new = '''    def _budgets(self) -> tuple[float, float]:\n        # Explicit values always win, independently, just as they do in the one-shot CLI.\n        if self.cfg.preset == "auto":\n            available = _available_physical_memory_bytes()\n            if available <= 0:\n                raise RuntimeError(\n                    "resident --preset auto could not read available physical memory; "\n                    "pass a named preset or explicit --trunk-gb/--cache-gb"\n                )\n            auto_trunk, auto_cache = _auto_worker_budgets(\n                available / 1e9,\n                prefill_mb=self.cfg.prefill_mb,\n                worker_context=self.cfg.worker_context,\n            )\n            trunk = self.cfg.trunk_gb if self.cfg.trunk_gb is not None else auto_trunk\n            cache = self.cfg.cache_gb if self.cfg.cache_gb is not None else auto_cache\n            return float(trunk), float(cache)\n\n        defaults = self._PRESET_BUDGETS.get(self.cfg.preset)\n        if defaults is None:\n            raise ValueError(\n                f"resident worker cannot resolve preset {self.cfg.preset!r}; "\n                "use auto, a named fixed preset, or explicit --trunk-gb/--cache-gb"\n            )\n        trunk = self.cfg.trunk_gb if self.cfg.trunk_gb is not None else defaults[0]\n        cache = self.cfg.cache_gb if self.cfg.cache_gb is not None else defaults[1]\n        return float(trunk), float(cache)\n'''
if old not in s:
    raise SystemExit("_budgets anchor not found")
s = s.replace(old, new, 1)

old = '''    def _command(self) -> list[str]:\n        trunk_gb, cache_gb = self._budgets()\n        cmd = ['''
new = '''    def _command(self) -> list[str]:\n        trunk_gb, cache_gb = self._budgets()\n        if self.cfg.preset == "auto":\n            print(\n                f"resident auto memory: trunk {trunk_gb:.2f} GB / "\n                f"expert cache {cache_gb:.2f} GB "\n                f"(available before worker {_available_physical_memory_bytes()/1e9:.1f} GB)"\n            )\n        cmd = ['''
if old not in s:
    raise SystemExit("_command anchor not found")
s = s.replace(old, new, 1)

p.write_text(s)

# Add focused deterministic tests to the existing bridge suite.
p = Path("tests/python/test_local_bridge.py")
t = p.read_text()
t = t.replace(
    "from local.k3_local import LocalK3, _is_loopback_host, parse_xtml",
    "from local.k3_local import (\n    BackendConfig, LocalK3, ResidentCBackend, _auto_worker_budgets,\n    _is_loopback_host, parse_xtml,\n)",
    1,
)
insert_anchor = '''    def test_loopback_guard(self):\n'''
tests = '''    def test_auto_worker_budget_uses_32gb_for_exact_trunk_not_expert_lru(self):\n        trunk, cache = _auto_worker_budgets(32.0, prefill_mb=256.0, worker_context=1024)\n        self.assertGreater(trunk, 18.0)\n        self.assertLess(trunk, 24.0)\n        self.assertEqual(cache, 0.5)\n\n    def test_auto_worker_budget_reserves_hot_kv_for_large_virtual_context(self):\n        small_ctx = _auto_worker_budgets(32.0, prefill_mb=256.0, worker_context=1024)\n        huge_ctx = _auto_worker_budgets(32.0, prefill_mb=256.0, worker_context=1048576)\n        self.assertLess(huge_ctx[0], small_ctx[0])\n        self.assertEqual(huge_ctx[1], 0.5)\n        # Virtual 1M context must not reserve 1M physical KV rows at startup.\n        self.assertGreater(huge_ctx[0], 14.0)\n\n    def test_auto_worker_budget_fills_trunk_before_expert_cache(self):\n        trunk, cache = _auto_worker_budgets(192.0, prefill_mb=256.0, worker_context=1024)\n        self.assertEqual(trunk, 111.0)\n        self.assertGreater(cache, 0.5)\n\n    def test_auto_worker_budget_fails_closed_when_host_is_too_busy(self):\n        with self.assertRaisesRegex(RuntimeError, "Close memory-heavy apps"):\n            _auto_worker_budgets(8.0, prefill_mb=256.0, worker_context=1024)\n\n    def test_default_backend_config_is_machine_aware(self):\n        cfg = BackendConfig(\n            model_dir=__import__('pathlib').Path('/model'),\n            trunk_dir=__import__('pathlib').Path('/trunk'),\n            binary=__import__('pathlib').Path('/bin/k3'),\n        )\n        self.assertEqual(cfg.preset, "auto")\n\n'''
if insert_anchor not in t:
    raise SystemExit("test insertion anchor not found")
t = t.replace(insert_anchor, tests + insert_anchor, 1)
p.write_text(t)
print("staged local resident auto-memory defaults")
