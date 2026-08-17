/* k3_trunk.h - stream the resident trunk, so RAM becomes a dial instead of a floor.
 *
 * WHY STREAM THE TRUNK AT ALL
 *   The engine holds 110 GB of trunk plus 4.70 GB of embed/lm_head. That is the floor
 *   that forces a 128 GB machine. Quantising it down is the obvious idea and it is the
 *   wrong one: Kimi K3's technical report section 4.1.4 says the experts are MXFP4 with
 *   quantisation-aware training "while all non-expert components (attention projections,
 *   latent MoE projections, shared experts, and MoE routers) remain in higher
 *   precision". That list IS this trunk. Measured on 31 real attention tensors
 *   (docs/data/trunk-quantisation.txt), post-hoc int4 costs 17.4% mean relative WEIGHT
 *   reconstruction error against 0.96% for int8 -- an ~18x gap, consistent across every
 *   tensor sampled. That is weight error, not output quality, but it is enough to rule
 *   out 4-bit on a trunk that was never trained for it.
 *
 *   Streaming costs zero error. The bytes are the checkpoint's own bytes.
 *
 * WHY IT IS AFFORDABLE
 *   Read bandwidth decides this, and it varies by an order of magnitude between a
 *   network volume and local NVMe. Measure the target device with tools/devbw.py
 *   before drawing conclusions. On local NVMe at a few GB/s the whole trunk costs
 *   tens of seconds per token against compute of the same order, so the read is not
 *   automatically the bottleneck. What makes it hideable is that, unlike expert routing,
 *   the trunk access order is FIXED: layer 0, 1, ... 92, every single token, so the next
 *   read is always known in advance.
 *
 *   It IS hidden now. k3_trunk_prefetch hands layer L+1 to a reader thread while the
 *   main thread computes on layer L, which is what the fixed walk order makes safe: the
 *   next layer is always known, so there is nothing to predict. The same reader also
 *   prepares the layer's memory bind and BF16/Q4/I8 elementwise widening before
 *   publishing the slot, so both storage and bind preparation overlap compute.
 *
 *   The second ring slot it needs costs a full slot, 2.37 GB at the floor, so it is only
 *   taken when the trunk budget can pay for it. Below that the ring stays at one slot and
 *   reads are serial again; k3_trunk_open says so on stdout when that happens. Correctness
 *   does not depend on which path runs, and the emitted tokens are identical either way.
 *
 * WHY LRU WOULD BE THE WORST POSSIBLE POLICY HERE
 *   A cyclic sequential scan is the classic LRU pathology. With N < 93 slots, by the
 *   time the walk returns to layer 0 it is exactly the least recently used thing and has
 *   just been evicted, so the hit rate is ZERO no matter how much RAM is added. This
 *   cache therefore PINS a prefix of layers and streams the rest through a small ring:
 *   pin K layers and the hit rate is exactly K/93, deterministically, and every extra
 *   gigabyte buys its fair share. The expert cache keeps LRU because expert reuse is
 *   data-dependent, which is the opposite situation.
 *
 * LAYOUT
 *   tools/pack_trunk.py copies each layer's trunk, which is ONE contiguous run in its
 *   shard, into trunk.bin, and records offsets in trunk.json. So loading a layer is a
 *   single pread from local NVMe. The bytes are copied verbatim, so a tensor's position
 *   inside a slot is (its absolute shard offset - the run start).
 */
#ifndef K3_TRUNK_H
#define K3_TRUNK_H

#include "k3.h"
#include "k3_bind.h"
#include "k3_codec.h"

#define K3_TRUNK_ALIGN 4096   /* pack_trunk.py pads runs to this so O_DIRECT works */

typedef struct {
    char    *name;
    int64_t  off;          /* byte offset WITHIN the layer run */
    int64_t  nbytes;
    int      dtype;        /* K3Dtype */
} K3TrunkTensor;

typedef struct {
    int64_t file_off;       /* absolute offset in trunk.bin, 4096 aligned */
    int64_t stored_nbytes;  /* bytes read with O_DIRECT, includes tail padding */
    int64_t encoded_nbytes; /* payload bytes before O_DIRECT padding */
    int64_t raw_off;        /* destination offset in the reconstructed layer run */
    int64_t raw_nbytes;
    int     codec;          /* 0 raw, 1 dict15, 2 dict7 */
    unsigned char dict[K3_DICT15_SIZE];
} K3TrunkBlock;

typedef struct {
    int64_t  file_off;     /* offset in trunk.bin (raw trunks) / first block */
    int64_t  nbytes;       /* RECONSTRUCTED raw bytes; tensor offsets refer to these */
    int64_t  stored_nbytes;/* physical bytes for compressed blocks, informational */
    K3TrunkTensor *t;
    int      nt;
    K3TrunkBlock *blocks;  /* NULL for the original raw one-pread format */
    int      nblocks;
} K3TrunkLayer;

typedef struct {
    int          fd;
    int          direct;        /* 1 when the file was opened O_DIRECT */
    int          n_layers;
    K3TrunkLayer *lay;

    /* Backs every K3TrunkTensor.name, so it must outlive the whole struct. Owned here
     * and freed by k3_trunk_close; do not free the parser arena separately. */
    char          *json_arena;

    /* The config is owned by the caller and outlives the trunk in every engine path.
     * The reader needs it to prepare the next K3LayerBind off the compute thread. */
    const K3Cfg    *cfg;

    /* Pinned layers get exact-size allocations; only the streaming ring is uniform.
     * Uniform slots everywhere would size EVERY slot for layer 0, whose dense MLP makes
     * it 2.34 GB against 1.27 GB for a normal layer, wasting about half the budget.
     * A pinned layer's raw bytes never change, so its memory bind/widening is prepared
     * on first touch and then copied for every later token. */
    unsigned char **pin;        /* [npin] one exact allocation per pinned layer */
    K3LayerBind   *pin_prepared;/* [npin] pointers into pin[L] + its widen tail   */
    unsigned char *pin_prepared_valid; /* [npin] */
    unsigned char *arena;       /* shared reconstructed streaming arena        */
    unsigned char *codec_arena; /* [nslot] compressed-block O_DIRECT scratch     */
    int64_t      slot_bytes;    /* logical normal-layer slot + widen area       */
    int64_t      arena_bytes;   /* physical bytes allocated for arena            */
    int64_t      widen_bytes;   /* of slot_bytes, the fp32 expansion area       */
    int          split_first;   /* layer 0 uses whole arena, then it becomes 2 slots */
    int64_t      codec_slot_bytes; /* max stored block, counted inside budget    */
    int          nslot;
    int          npin;          /* layers 0..npin-1 are pinned                  */
    int         *layer_of;      /* [nslot] which layer occupies each ring slot  */
    int32_t     *slot_of;       /* [n_layers], -1 when not resident             */
    int          ring;          /* next ring slot to reuse                      */

    /* Each streaming slot owns a fully prepared memory bind. Tensor pointers inside it
     * refer only to that slot's raw/widen bytes. The reader constructs it before
     * publication; the compute thread copies it and rebases only self-pointers. */
    K3LayerBind  *prepared;      /* [nslot] */
    int          *prepared_layer;/* [nslot], -1 until preparation completed */

    /* One asynchronous reader owns one spare ring slot. The worker never publishes a
     * layer name before its read AND bind preparation succeed; bind waits for completion
     * before consuming it. */
    void         *io_state;

    /* stats */
    uint64_t     hits, misses;
    uint64_t     bytes_read;             /* physical bytes read from trunk.bin */
    uint64_t     raw_bytes_reconstructed; /* raw bytes produced for the binder   */
    double       load_seconds;             /* pread only                          */
    double       decode_seconds;           /* dict15 reconstruction only          */
    uint64_t     prepared_builds;          /* completed memory binds/widenings     */
    uint64_t     prepared_hits;            /* prepared binds copied to compute     */
    uint64_t     async_prepared_builds;    /* subset prepared on reader thread     */
    double       prepare_seconds;           /* bind/widen wall across both threads  */
} K3Trunk;

/* budget_bytes sizes the slot array. Layers 0..K-1 are pinned, where K is as large as
 * the budget allows minus a small streaming ring. Returns 0 on success. */
int  k3_trunk_open(K3Trunk *tr, const char *dir, const K3Cfg *c, int64_t budget_bytes);
void k3_trunk_close(K3Trunk *tr);

/* Make layer L resident and copy its prepared memory bind into b. The copied bind owns
 * no allocation; k3_bind_free() remains safe and only clears it. */
int  k3_trunk_bind(K3Trunk *tr, const K3Cfg *c, int L, K3LayerBind *b);

/* Start an asynchronous read+decode+bind preparation of layer L into its slot, if it is
 * not resident. Safe to call for a pinned/already loaded layer (it becomes a no-op). */
void k3_trunk_prefetch(K3Trunk *tr, int L);

void k3_trunk_report(const K3Trunk *tr, const char *label);

#endif /* K3_TRUNK_H */
