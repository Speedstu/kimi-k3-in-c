# Lossless compressed trunk

This mode targets the machine class where K3 is limited by reading the bf16 trunk from SSD every generated token.

It is **not quantization**. The compressed file is only a storage representation. Before a layer is bound, the runtime reconstructs the original packed trunk run byte-for-byte, so the matmul kernels receive the same bf16/f32 bytes as the normal trunk.

## Why this codec

The dense trunk is dominated by bf16. In little-endian storage a bf16 value is:

```text
[low mantissa byte][high sign/exponent byte]
```

The low byte behaves almost like noise, while the high byte is strongly concentrated. The codec therefore keeps every low byte verbatim and encodes the high-byte plane with a tiny 15-entry dictionary:

- dictionary index `0..14`: one of the 15 most frequent high bytes in that block;
- nibble `15`: escape, followed by the literal high byte;
- two dictionary codes per byte;
- dictionary chosen independently for each block;
- raw fallback when encoding would not shrink a block.

Blocks are independent and 4096-byte aligned so the trunk reader can keep using `O_DIRECT`.

## Build a lossless trunk

First create the normal packed trunk as usual:

```bash
python3 tools/pack_trunk.py /path/to/k3-checkpoint /path/to/k3-trunk 93
```

Then derive the compressed storage representation:

```bash
python3 tools/lossless_trunk.py /path/to/k3-trunk /path/to/k3-trunk-lossless
```

Run K3 exactly as before, changing only the `--trunk` path:

```bash
./bin/k3 /path/to/k3-checkpoint \
  --trunk /path/to/k3-trunk-lossless \
  --preset auto \
  --incremental \
  --ids 1,2,3 \
  --gen 8
```

Old raw packed trunks remain supported. The loader selects the block path only when `trunk.json` contains the lossless block directory.

## RAM accounting

The runtime reconstructs a block into the normal ring slot before binding tensors. A separate compressed-block scratch area is needed for each active ring slot so asynchronous prefetch and foreground loading cannot race.

That scratch is **included in the trunk budget**. The current converter uses 128 MiB raw blocks; a typical ~0.75-ratio encoded block needs roughly 96 MiB of compressed scratch per ring slot.

## Correctness gates

The branch is gated at several levels:

- `test_trunk_codec`: byte-exact synthetic roundtrip, including escapes, a non-SIMD tail, malformed/truncated input rejection, and a scalar C99 build;
- normal `make test`: the full existing weightless kernel/model oracle;
- CMake/ctest: the codec test is registered in the second build system too;
- tiny-checkpoint end-to-end smoke test: raw packed trunk and lossless trunk produce **binary-identical dumped logits** and identical prompt/generated/full token IDs.

In the end-to-end tiny test on the GitHub runner, the compressed trunk stored **23.0% fewer bytes after O_DIRECT alignment**, and both runs generated:

```text
[92, 168, 13]
```

The logits files compared equal with `cmp`.

## Decoder microbenchmark

`benchmarks/bench_bf16_dict_codec.c` benchmarks only the reconstruction loop so storage and model compute do not hide codec cost.

On the GitHub Ubuntu runner used during development, a 256 MiB synthetic bf16 stream with a ~0.1% escape rate measured:

- encoded ratio: **0.7505**;
- byte-exact roundtrip: **PASS**;
- best reconstructed throughput: **17.59 GB/s**;
- mean reconstructed throughput over five runs: **17.20 GB/s**.

Example:

```bash
cc -O3 -mavx2 -Wall -Wextra benchmarks/bench_bf16_dict_codec.c -o /tmp/bench_codec
/tmp/bench_codec 256 5
```

These are **codec microbenchmark numbers, not full K3 throughput**.

## What has not been measured yet

The complete released ~109 GB streamed trunk was not available in the development runner. Therefore this branch does **not** claim yet that the real K3 trunk is exactly 25% smaller or that end-to-end decode is exactly 1.33x faster.

The full-checkpoint measurement still required is:

1. convert the real packed trunk and record its final stored size;
2. run the same prompt, memory budget, thread count, and SSD with raw vs lossless trunks;
3. replicate several runs because the project's published I/O timings have substantial noise;
4. confirm identical token IDs/logits while comparing physical trunk bytes read and seconds/token.

The intended regime is low/medium RAM where substantial trunk data is streamed every token. When the trunk is already resident, storage compression should not improve steady-state decode and only adds one-time load/decode work.
