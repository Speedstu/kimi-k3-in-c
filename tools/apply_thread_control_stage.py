#!/usr/bin/env python3
from pathlib import Path
import sys

root = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
p = root / "src/cli/k3_run.c"
s = p.read_text(encoding="utf-8")


def one(old, new, label):
    global s
    n = s.count(old)
    if n != 1:
        raise RuntimeError(f"{label}: expected one match, found {n}")
    s = s.replace(old, new, 1)


one(
'''#include <sys/resource.h>

#include "k3.h"''',
'''#include <sys/resource.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "k3.h"''',
"omp include",
)

one(
'''"generation:\\n"
"  --gen N               tokens to generate (default 8)\\n"''',
'''"generation / compute:\\n"
"  --gen N               tokens to generate (default 8)\\n"
"  --threads N           OpenMP compute threads for this run. Exact output is unchanged;\\n"
"                        use benchmarks/thread-sweep.sh to MEASURE the best N\\n"''',
"usage threads",
)

one(
'''    int gen = 8, want_layers = -1;
    double cache_gb = 64.0, trunk_gb = 16.0;''',
'''    int gen = 8, want_layers = -1;
    int threads = 0;              /* 0 = OpenMP/runtime default */
    double cache_gb = 64.0, trunk_gb = 16.0;''',
"threads variable",
)

one(
'''        else if (!strcmp(argv[i], "--gen") && i + 1 < argc) gen = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--cache-gb")''',
'''        else if (!strcmp(argv[i], "--gen") && i + 1 < argc) gen = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc) threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--cache-gb")''',
"parse threads",
)

# Configure the main OpenMP thread before budgets/loading/compute. The async expert worker
# has a thread-local OpenMP setting of its own, so this does not overwrite its I/O team.
needle = '''    {
        int nsrc = (ids_s != NULL) + (prompt_text != NULL) + (prompt_file != NULL);
        if (nsrc == 0) {
            fprintf(stderr, "one of --ids, --prompt or --prompt-file is required\\n");
            return 2;
        }
        if (nsrc > 1) {
            /* Refuse rather than pick: silently preferring one source would make a
             * mistyped invocation run the WRONG prompt for tens of minutes. */
            fprintf(stderr, "--ids, --prompt and --prompt-file are mutually exclusive\\n");
            return 2;
        }
    }

    /* ---- auto budget ----'''
replacement = '''    {
        int nsrc = (ids_s != NULL) + (prompt_text != NULL) + (prompt_file != NULL);
        if (nsrc == 0) {
            fprintf(stderr, "one of --ids, --prompt or --prompt-file is required\\n");
            return 2;
        }
        if (nsrc > 1) {
            /* Refuse rather than pick: silently preferring one source would make a
             * mistyped invocation run the WRONG prompt for tens of minutes. */
            fprintf(stderr, "--ids, --prompt and --prompt-file are mutually exclusive\\n");
            return 2;
        }
    }

    if (threads < 0 || threads > 4096) {
        fprintf(stderr, "--threads must be in [1,4096] when supplied, got %d\\n", threads);
        return 2;
    }
#ifdef _OPENMP
    if (threads > 0) {
        /* Dynamic team resizing defeats a repeatable thread sweep: two runs with the
         * same --threads could otherwise get different team sizes under system load. */
        omp_set_dynamic(0);
        omp_set_num_threads(threads);
    }
    const int compute_threads = omp_get_max_threads();
#else
    if (threads > 0 && threads != 1) {
        fprintf(stderr, "--threads %d requested, but this binary was built without OpenMP\\n",
                threads);
        return 2;
    }
    const int compute_threads = 1;
#endif

    /* ---- auto budget ----'''
one(needle, replacement, "configure threads")

# Echo effective threads in the captured banner.
one(
'''    printf("Kimi K3, pure C, released checkpoint\\n");
    /* The directory, not a shard count:''',
'''    printf("Kimi K3, pure C, released checkpoint\\n");
    printf("  threads  : %d compute%s\\n", compute_threads,
           threads > 0 ? " (explicit --threads)" : " (OpenMP/runtime default)");
    /* The directory, not a shard count:''',
"thread banner",
)

# Make result JSON self-describing for sweeps/results archives.
one(
'''        fprintf(f, "],\\\"layers\\\":%d,\\\"seconds_per_token\\\":%.4f}\\n", NL, t_total / nout);''',
'''        fprintf(f, "],\\\"layers\\\":%d,\\\"threads\\\":%d,\\\"seconds_per_token\\\":%.4f}\\n",
                NL, compute_threads, t_total / nout);''',
"JSON threads",
)

p.write_text(s, encoding="utf-8", newline="\n")
print("staged exact --threads control")
