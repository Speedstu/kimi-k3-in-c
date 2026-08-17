# Full-checkpoint score gate

Hosted CI validates runtime exactness and interface invariants. Actual K3 Max score parity requires the released full checkpoint plus the benchmark suites themselves, so `.github/workflows/k3-full-benchmark-gate.yml` deliberately runs only on a self-hosted runner labelled `k3-full-checkpoint`.

That runner provides `K3_MODEL_DIR`, `K3_TRUNK_DIR`, and a `k3-max-benchmark` command that writes measured results using the keys in `k3_max_reference.json`. The workflow then invokes `k3_score_gate.py`; missing or lower scores fail by default.
