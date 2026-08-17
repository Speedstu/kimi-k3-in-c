# Verified-improvement training

K3-Compact cannot beat K3 Max by minimizing teacher KL alone. Pure imitation converges
toward the teacher. The specialist stage therefore separates **retention** from
**improvement**.

## Retention channel

`compact/distill.py` retains general capability with three signals:

1. token-distribution KL from K3 Max;
2. normalized hidden-state alignment at selected layers;
3. full router-distribution distillation after summing the 896 teacher probabilities into
   the 48 behavior clusters selected by `expert_cluster.py`.

The router target is not top-k-truncated. All teacher routing probability mass is mapped to
the student expert space.

## Improvement channel

Training tasks generate two or more candidate trajectories. An executable verifier selects
`chosen` versus `rejected`. K3 Max and K3-Compact both score the same pair. The loss requires

```
student_chosen_logp - student_rejected_logp
    >= teacher_chosen_logp - teacher_rejected_logp + margin
```

The default margin is 0.25 sequence-logprob units. A smooth softplus hinge provides the
gradient. This gives the student an explicit objective to become more decisive than its
teacher on independently verified successes.

Verifier examples by training domain:

- **code:** compilation, unit/integration tests, repository regression suites, static type
  checks, deterministic performance/correctness contracts;
- **agentic:** tool-call schema validity, file/task state checks, deterministic browser or
  terminal sandbox goals, multi-step completion assertions;
- **cyber specialist:** isolated CTF challenge flags, defensive patch regression tests,
  detection-rule fixtures, malware-analysis toy/lab artifacts, and explicitly authorized
  lab tasks only.

No real-world unauthorized target is a valid training verifier.

## Avoiding fake benchmark gains

Benchmark test items used by `compact/score_gate.py` must never enter this training stream.
Dataset producers must keep source IDs and split provenance so exact hashes can be denied at
training time. Public benchmark *training/dev* material may only be used when the benchmark
license and protocol permit it; held-out evaluation items remain denied.

The final release claim remains fail-closed: verified-training loss going down is **not**
evidence that K3-Compact beats K3 Max. Only the held-out head-to-head score gate can make
that claim.
