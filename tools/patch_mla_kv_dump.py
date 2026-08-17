#!/usr/bin/env python3
from pathlib import Path
p=Path('src/core/k3_ops.c')
s=p.read_text()
needle='''    /* ---- attention, per head, causal ---- */\n'''
insert=r'''    /* Diagnostic-only exact cache dump for the lossless-RAM probe. Disabled unless the
     * environment variable is present, and never intended for the production branch. */
    const char *kvdump = getenv("K3_DUMP_MLA_KV");
    if (kvdump && *kvdump && kvc) {
        FILE *df = fopen(kvdump, "ab");
        if (!df) { perror("K3_DUMP_MLA_KV"); abort(); }
        for (int t = 0; t < T; t++) {
            const uint64_t magic = UINT64_C(0x31444b564b334b); /* K3KVD1-ish */
            const uint64_t key = (uint64_t)(uintptr_t)kvc;
            const uint32_t pos = (uint32_t)(cached + t);
            const uint32_t nf = (uint32_t)(H * kvd);
            fwrite(&magic, sizeof(magic), 1, df);
            fwrite(&key, sizeof(key), 1, df);
            fwrite(&pos, sizeof(pos), 1, df);
            fwrite(&nf, sizeof(nf), 1, df);
            fwrite(K3_KV_AT(cached + t), sizeof(float), nf, df);
        }
        fclose(df);
    }

    /* ---- attention, per head, causal ---- */
'''
if needle not in s: raise SystemExit('anchor not found')
s=s.replace(needle,insert,1)
p.write_text(s)
