#!/usr/bin/env python3
from pathlib import Path


def one(path: str, old: str, new: str) -> None:
    p = Path(path)
    s = p.read_text()
    n = s.count(old)
    if n == 0:
        if new in s:
            print(path, "already patched")
            return
        raise SystemExit(f"{path}: expected pattern missing")
    if n != 1:
        raise SystemExit(f"{path}: ambiguous pattern ({n} matches)")
    p.write_text(s.replace(old, new, 1))
    print(path, "patched")


one(
    "src/io/k3_trunk.h",
    """    unsigned char *arena;       /* [nslot] uniform RECONSTRUCTED ring slots     */
    unsigned char *codec_arena; /* [nslot] compressed-block O_DIRECT scratch     */
    int64_t      slot_bytes;    /* raw run + the widen area                     */
    int64_t      widen_bytes;   /* of slot_bytes, the fp32 expansion area       */
""",
    """    unsigned char *arena;       /* shared reconstructed streaming arena        */
    unsigned char *codec_arena; /* [nslot] compressed-block O_DIRECT scratch     */
    int64_t      slot_bytes;    /* logical normal-layer slot + widen area       */
    int64_t      arena_bytes;   /* physical bytes allocated for arena            */
    int64_t      widen_bytes;   /* of slot_bytes, the fp32 expansion area       */
    int          split_first;   /* layer 0 uses whole arena, then it becomes 2 slots */
""",
)

one(
    "src/io/k3_trunk.c",
    """    tr->npin = npin;
    tr->nslot = RING;
    tr->slot_bytes = ring_slot;
    tr->codec_slot_bytes = codec_slot;
""",
    """    /* Low-RAM special case: on released K3 layer 0 is the only dense layer and
     * is roughly twice a normal MoE trunk layer. At the laptop floor that one oversized
     * layer forces a single ~2.37 GB ring slot, which disables all read/compute overlap.
     *
     * Reuse, rather than add, memory: layer 0 gets the whole arena synchronously. Once
     * its compute is finished, the exact same bytes are viewed as two smaller slots for
     * layers 1..N-1, enabling the existing one-layer-ahead reader without raising the
     * trunk budget. K3_NO_SPLIT_RING=1 restores the old one-slot planner for A/B tests.
     * The first layer must truly be oversized, otherwise this mode buys nothing. */
    int split_first = 0;
    int64_t arena_bytes = (int64_t)RING * ring_slot;
    if (!getenv("K3_NO_SPLIT_RING") && RING == 1 && npin == 0 && tr->n_layers > 1) {
        int64_t rest_big = 0;
        for (int i = 1; i < tr->n_layers; i++)
            if (tr->lay[i].nbytes > rest_big) rest_big = tr->lay[i].nbytes;
        int64_t rest_slot = (rest_big + K3_TRUNK_ALIGN - 1)
                            & ~(int64_t)(K3_TRUNK_ALIGN - 1);
        rest_slot += (int64_t)widen;
        rest_slot = (rest_slot + 4095) & ~(int64_t)4095;
        int64_t first_slot = (tr->lay[0].nbytes + K3_TRUNK_ALIGN - 1)
                             & ~(int64_t)(K3_TRUNK_ALIGN - 1);
        first_slot += (int64_t)widen;
        first_slot = (first_slot + 4095) & ~(int64_t)4095;
        const int64_t two_rest = 2 * rest_slot;
        const int64_t shared = first_slot > two_rest ? first_slot : two_rest;
        const int64_t split_need = shared + 2 * codec_slot;
        if (rest_big > 0 && first_slot > rest_slot && split_need <= budget_bytes) {
            split_first = 1;
            RING = 2;
            ring_slot = rest_slot;
            arena_bytes = shared;
        }
    }

    tr->npin = npin;
    tr->nslot = RING;
    tr->slot_bytes = ring_slot;
    tr->arena_bytes = arena_bytes;
    tr->split_first = split_first;
    tr->codec_slot_bytes = codec_slot;
""",
)

one(
    "src/io/k3_trunk.c",
    """    if (k3_alloc_direct((void **)&tr->arena, (size_t)RING * (size_t)ring_slot) != 0) {
        fprintf(stderr, "k3_trunk: cannot allocate the %.2f GB streaming ring\\n",
                (double)RING * ring_slot / 1e9);
""",
    """    if (k3_alloc_direct((void **)&tr->arena, (size_t)arena_bytes) != 0) {
        fprintf(stderr, "k3_trunk: cannot allocate the %.2f GB streaming arena\\n",
                (double)arena_bytes / 1e9);
""",
)

one(
    "src/io/k3_trunk.c",
    """    if (codec_slot > 0)
        printf("              lossless dict15 blocks: %.2f GB codec scratch per ring slot "
               "(counted in trunk budget)\\n", (double)codec_slot / 1e9);
""",
    """    if (split_first)
        printf("              split-first arena: layer 0 uses %.2f GB whole, then layers 1..%d "
               "reuse it as 2 x %.2f GB slots (no extra trunk RAM)\\n",
               (double)arena_bytes / 1e9, tr->n_layers - 1, (double)ring_slot / 1e9);
    if (codec_slot > 0)
        printf("              lossless dict15 blocks: %.2f GB codec scratch per ring slot "
               "(counted in trunk budget)\\n", (double)codec_slot / 1e9);
""",
)

one(
    "src/io/k3_trunk.c",
    """    if (L < tr->npin) {
        base = tr->pin[L];
""",
    """    if (tr->split_first && L == 0) {
        /* The previous token must have consumed its last prefetch before the whole arena
         * can be reused for layer 0. Drain defensively: a future caller that changes the
         * walk order must fail here rather than overwrite an in-flight read. */
        K3TrunkIO *io = (K3TrunkIO *)tr->io_state;
        if (io) {
            pthread_mutex_lock(&io->mu);
            while (io->busy && !io->stop)
                pthread_cond_wait(&io->cv, &io->mu);
            const int bad = io->stop || (io->done && io->result != 0);
            io->done = 0;                 /* any successful unused prefetch is discarded */
            pthread_mutex_unlock(&io->mu);
            if (bad) {
                fprintf(stderr, "k3_trunk: split-ring reader failed before layer 0\\n");
                return -1;
            }
        }
        for (int i = 0; i < tr->nslot; i++) {
            if (tr->layer_of[i] >= 0) tr->slot_of[tr->layer_of[i]] = -1;
            tr->layer_of[i] = -1;
        }
        tr->ring = 0;
        if (load_run(tr, 0, tr->arena, 0) != 0) return -1;
        tr->misses++;
        base = tr->arena;
    } else if (L < tr->npin) {
        base = tr->pin[L];
""",
)

one(
    "src/io/k3_trunk.c",
    """void k3_trunk_prefetch(K3Trunk *tr, int L)
{
    if (L < 0 || L >= tr->n_layers || L < tr->npin) return;
""",
    """void k3_trunk_prefetch(K3Trunk *tr, int L)
{
    if (L < 0 || L >= tr->n_layers || L < tr->npin) return;
    /* forward() asks for L+1 immediately after binding L. In split-first mode layer 0
     * still occupies the WHOLE shared arena at that point, so L1 deliberately loads
     * synchronously only after layer0 compute returns. From L1 onward two slots exist. */
    if (tr->split_first && L == 1) return;
""",
)

one(
    "tools/make_tiny_checkpoint.py",
    """def build(seed: int):
    torch.manual_seed(seed)
    cfg = tiny_config(moe_intermediate_size=64)
""",
    """def build(seed: int, dense_intermediate: int = 96):
    torch.manual_seed(seed)
    cfg = tiny_config(moe_intermediate_size=64, intermediate_size=dense_intermediate)
""",
)

one(
    "tools/make_tiny_checkpoint.py",
    """    ap.add_argument("--prompt-ids", default="3,7,11,5,9")
    a = ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    cfg, model = build(a.seed)
""",
    """    ap.add_argument("--prompt-ids", default="3,7,11,5,9")
    ap.add_argument("--dense-intermediate", type=int, default=96,
                    help="dense layer-0 intermediate width; 512 mimics K3's oversized first layer")
    a = ap.parse_args()

    if a.dense_intermediate <= 0:
        ap.error("--dense-intermediate must be positive")
    os.makedirs(a.out_dir, exist_ok=True)
    cfg, model = build(a.seed, a.dense_intermediate)
""",
)
