# Speedboost branch

This branch keeps **exact Kimi K3 authoritative**. Anything approximate is confined to a speculative draft whose proposals are verified by the exact bf16/MXFP4 model before a token is emitted.

## What changed

### Exact path: allocation-free KDA recurrence

The real KDA layer no longer allocates/frees its `u` temporary for every head and token. It reuses a dead `q` scratch slice after the scaled query has been copied into the per-head work row. Arithmetic and recurrence ordering are unchanged.

### Speculative verification: reuse lm-head rows

For short speculative batches (2 to 9 positions), verification walks vocabulary rows outermost and positions innermost. A row of the ~2.35 GB bf16 `lm_head` is therefore reused while hot instead of rereading the whole head once per proposed position. Each row dot product still uses the normal `k3_mmw` path, so logits are unchanged.

### Draft-only reduced expert routing

`--draft-topk K` controls the number of routed experts used by the proposal model. Default: `4`. The exact model still uses the checkpoint's top-16 routing and verifies every emitted token.

`--draft-cache-only` is optional. It makes the draft route only among already-resident experts, eliminating new expert reads for a draft step, but small expert caches can hurt proposal acceptance badly.

### Draft-only groupwise Q4 trunk

`tools/q4_trunk.py` derives a groupwise signed-int4 draft trunk from an existing packed bf16 trunk. The exact trunk is not modified.

Format for each 128-weight group:

```
[f32 scale][64 bytes containing 128 signed int4 values]
```

That is 68 / 128 = **0.53125 bytes per weight** for full groups, versus ~1 byte/weight for the I8R draft and 2 bytes/weight for bf16. For the large matrix portion of the K3 trunk this should put the draft around the ~30 GB class; the exact full-checkpoint size must be measured after conversion rather than inferred from the ratio.

Build a Q4 draft:

```bash
python3 tools/q4_trunk.py /path/to/bf16_packed_trunk /path/to/k3_q4_draft
```

Then start with:

```bash
./bin/k3 /path/to/model \
  --trunk /path/to/bf16_packed_trunk \
  --incremental \
  --draft-trunk /path/to/k3_q4_draft \
  --draft-trunk-gb 32 \
  --draft-topk 4 \
  --spec 4 \
  --ids 1,2,3
```

The best `--draft-topk` is workload- and storage-dependent. Sweep `2`, `4`, and `8`; compare end-to-end seconds/token and the hybrid draft acceptance summary. A lower acceptance rate can make an aggressively cheap draft slower overall even though exact output remains unchanged.

## Correctness boundary

The following remain exact and authoritative:

- original bf16 trunk;
- original MXFP4 routed experts;
- exact top-16 routing;
- exact greedy verification;
- emitted token stream.

Q4 and reduced top-k are proposal-only. They can affect speed and acceptance, not the exact model's chosen token.

## Validation

The branch was built with the normal Makefile and passed `make test` in GitHub Actions. The added `matmul_q4g` test independently checks group scale placement, signed nibble decode, nibble order, row stride, and numerical output. The full weightless oracle also continues to match teacher forcing, greedy decode, and incremental decode exactly.

A full-checkpoint benchmark is still required before claiming a concrete end-to-end Q4 speedup or acceptance rate on real Kimi K3 weights.
