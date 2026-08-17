# No fake parity

A green hosted-CI exactness gate means the runtime preserves the reference computation on its validated fixtures. It is not a substitute for executing the released full checkpoint on the external benchmark suites. Full score parity is reported only after the self-hosted full-checkpoint gate produces measured results and `k3_score_gate.py` accepts every requested benchmark.
