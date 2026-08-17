# K3-Compact Specialist

Research branch for a **new derived model**, not a byte-identical K3 checkpoint.
The target is deliberately harder than compression: fit in <=100 GB and beat a measured
K3 Max teacher on code, sandboxed cybersecurity/CTF + defensive tasks, and agentic work.
No win is reported until `score_gate.py` passes on held-out evaluations.

## Architecture target

`k3_compact.json` currently chooses 48 routed experts/layer and top-4 routing while
preserving the 93-layer KDA/MLA skeleton, tokenizer, and released vision frontend.
Using the dimensions already measured by this repository, `plan.py` estimates:

- ~202.6B total parameters;
- ~68.9B active parameters/token;
- ~79.1 GB for 3-bit weights plus 16-bit scale per 128 weights;
- ~89.1 GB after explicit 8 GB unquantized + 2 GB packaging reserves.

Run:

```bash
python compact/plan.py
```

This is a storage/architecture estimate, **not a quality result**. The final checkpoint
must be measured after QAT.

## Why 48 experts instead of deleting 848 experts

Pruning by frequency alone would erase rare capabilities. The teacher-collection phase
must instead build a compact behaviour sketch for every expert on a large, benchmark-clean
training stream. `expert_cluster.py` performs usage-weighted spherical clustering of those
sketches and chooses an actual K3 expert as each cluster medoid. The medoid initializes the
student expert; the cluster is then learned by activation distillation. Packed MXFP4 bytes
are never averaged together.

Per routed layer the full-run pipeline is:

1. collect K3 Max router decisions plus expert-output random projections on training data;
2. cluster 896 behaviour sketches into 48 groups;
3. initialize 48 student experts from the selected teacher medoids;
4. freeze the trunk and train student experts/router against teacher hidden activations;
5. unfreeze and distill teacher logits + selected hidden states + router distribution;
6. run verifier-reward training on executable code tests, terminal tasks, isolated CTF labs,
   defensive analysis/patching/detection tasks and tool-use trajectories;
7. perform 3-bit quantization-aware training, then export the <=100 GB checkpoint;
8. evaluate teacher and student head-to-head with the same harness/sampling contract.

## Training mix

The initial curriculum is 40% code, 25% agentic, 20% isolated cyber/CTF + defense and
15% general-retention data. Held-out benchmark test items are excluded from training.
For executable tasks, verifier outcomes are preferred over style/judge rewards: unit tests,
compilers, repository tests, task completion checks, CTF flags inside isolated challenge
environments, patch regression tests, and defensive detection/analysis checks.

Cyber training is intentionally scoped to isolated CTF, defensive work and explicitly
authorized lab targets. This keeps the specialization reproducible without depending on
real-world unauthorized targets.

## "Better than K3 Max" contract

`k3_max_target_baselines.json` pins the public K3 bars used for code and agentic comparison.
The cyber baseline is intentionally null because no K3 Max Cybench value is present in the
pinned public K3 table. It must be measured with the same local harness before a cyber win
can pass.

`score_gate.py` is fail-closed. A release may carry the claim
`K3-Compact > K3 Max on code/cyber/agentic` only when:

- every configured code metric beats the teacher;
- every configured agentic metric beats the teacher;
- measured student Cybench beats measured teacher Cybench;
- checkpoint size is <=100 GB;
- general-retention composite is >=97% of the teacher.

Example:

```bash
python compact/score_gate.py results/k3_compact_head_to_head.json
```

Missing measurements are failures, not zeros or assumed wins.

## What is already real vs what still needs the full checkpoint

Already implemented and CI-testable on GitHub Hosted:

- deterministic parameter/storage accounting;
- 896 -> 48 behaviour-aware expert clustering with teacher medoids;
- strict head-to-head score gate;
- synthetic 896-expert clustering smoke and unit tests.

Requires the full K3 teacher + substantial training compute:

- teacher activation/router sketch collection;
- medoid weight extraction for all routed layers;
- activation/logit/router distillation;
- verifier-reward specialist training;
- 3-bit QAT/export;
- actual code/cyber/agentic head-to-head scores.

Until those are run, this repository must describe K3-Compact as a **research target**, not
as an already trained model or a proven K3 Max replacement.
