/* Windows-only A/B for the model reader.
 *
 * Legacy reproduces the correctness-first seek+read lock from the first Windows port.
 * Overlapped calls the current k3_pread implementation on a FILE_FLAG_OVERLAPPED handle.
 * Every run reads the same blocks from the same file and checks the same checksum.
 */
#if !defined(_WIN32)
#error windows-pread-bench is Windows-only
#endif

#include "k3_portable_io.h"

#include <fcntl.h>
#include <io.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>

static SRWLOCK legacy_lock = SRWLOCK_INIT;

static ssize_t legacy_pread(int fd, void *buf, size_t count, int64_t offset)
{
    AcquireSRWLockExclusive(&legacy_lock);
    const __int64 saved = _lseeki64(fd, 0, SEEK_CUR);
    if (saved < 0 || _lseeki64(fd, (__int64)offset, SEEK_SET) < 0) {
        ReleaseSRWLockExclusive(&legacy_lock);
        return (ssize_t)-1;
    }

    size_t done = 0;
    while (done < count) {
        const size_t remain = count - done;
        const unsigned take = (unsigned)(remain > (size_t)0x7ffff000u ?
                                         (size_t)0x7ffff000u : remain);
        const int got = _read(fd, (char *)buf + done, take);
        if (got <= 0) break;
        done += (size_t)got;
        if ((unsigned)got < take) break;
    }
    (void)_lseeki64(fd, saved, SEEK_SET);
    ReleaseSRWLockExclusive(&legacy_lock);
    return (ssize_t)done;
}

static double now_s(void)
{
    LARGE_INTEGER f, t;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&t);
    return (double)t.QuadPart / (double)f.QuadPart;
}

typedef struct {
    double mb_s;
    uint64_t checksum;
    int errors;
} Result;

static Result run_once(const char *path, int use_overlapped,
                       int threads, size_t chunk, int reads_per_thread)
{
    int fd;
    if (use_overlapped) {
        fd = k3_open_read(path, 0, NULL);
    } else {
        fd = _open(path, _O_RDONLY | _O_BINARY);
    }
    if (fd < 0) {
        fprintf(stderr, "cannot open %s\n", path);
        exit(2);
    }

    const int64_t bytes = k3_file_size(fd);
    const int64_t blocks = bytes / (int64_t)chunk;
    if (blocks < (int64_t)threads * reads_per_thread) {
        fprintf(stderr, "file too small: need at least %lld bytes\n",
                (long long)((int64_t)threads * reads_per_thread * (int64_t)chunk));
        exit(2);
    }

    uint64_t checksum = 0;
    int errors = 0;
    const double t0 = now_s();

#pragma omp parallel num_threads(threads) reduction(+:checksum,errors)
    {
        const int tid = omp_get_thread_num();
        unsigned char *buf = (unsigned char *)malloc(chunk);
        if (!buf) {
            errors += 1;
        } else {
            for (int r = 0; r < reads_per_thread; r++) {
                const int64_t block = ((int64_t)r * threads + tid) % blocks;
                const int64_t off = block * (int64_t)chunk;
                const ssize_t got = use_overlapped ?
                    k3_pread(fd, buf, chunk, off) : legacy_pread(fd, buf, chunk, off);
                if (got != (ssize_t)chunk) {
                    errors += 1;
                    continue;
                }
                checksum += (uint64_t)got;
                checksum += (uint64_t)buf[0] + (uint64_t)buf[chunk / 2] +
                            (uint64_t)buf[chunk - 1];
            }
            free(buf);
        }
    }

    const double dt = now_s() - t0;
    _close(fd);
    const double total = (double)threads * reads_per_thread * (double)chunk;
    Result out = { total / 1000000.0 / dt, checksum, errors };
    return out;
}

static int cmp_double(const void *a, const void *b)
{
    const double x = *(const double *)a, y = *(const double *)b;
    return x < y ? -1 : x > y ? 1 : 0;
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "usage: %s FILE [threads=8] [chunk_mb=8] [reads_per_thread=8]\n", argv[0]);
        return 2;
    }
    const char *path = argv[1];
    const int threads = argc > 2 ? atoi(argv[2]) : 8;
    const int chunk_mb = argc > 3 ? atoi(argv[3]) : 8;
    const int reads = argc > 4 ? atoi(argv[4]) : 8;
    if (threads <= 0 || chunk_mb <= 0 || reads <= 0) return 2;
    const size_t chunk = (size_t)chunk_mb * 1024u * 1024u;

    /* Warm both code paths before recording medians so file-cache order does not make the
     * first implementation look artificially slower. */
    Result w0 = run_once(path, 0, threads, chunk, reads);
    Result w1 = run_once(path, 1, threads, chunk, reads);
    if (w0.errors || w1.errors || w0.checksum != w1.checksum) {
        fprintf(stderr, "warmup parity failed: legacy=%llu/%d overlapped=%llu/%d\n",
                (unsigned long long)w0.checksum, w0.errors,
                (unsigned long long)w1.checksum, w1.errors);
        return 1;
    }

    enum { N = 5 };
    double legacy[N], over[N];
    uint64_t expect = w0.checksum;
    for (int i = 0; i < N; i++) {
        Result a, b;
        if (i & 1) {
            b = run_once(path, 1, threads, chunk, reads);
            a = run_once(path, 0, threads, chunk, reads);
        } else {
            a = run_once(path, 0, threads, chunk, reads);
            b = run_once(path, 1, threads, chunk, reads);
        }
        if (a.errors || b.errors || a.checksum != expect || b.checksum != expect) {
            fprintf(stderr, "read/checksum failure in round %d\n", i);
            return 1;
        }
        legacy[i] = a.mb_s;
        over[i] = b.mb_s;
        printf("round %d legacy %.1f MB/s  overlapped %.1f MB/s\n",
               i + 1, a.mb_s, b.mb_s);
    }

    qsort(legacy, N, sizeof legacy[0], cmp_double);
    qsort(over, N, sizeof over[0], cmp_double);
    const double lmed = legacy[N / 2], omed = over[N / 2];
    printf("median legacy     : %.1f MB/s\n", lmed);
    printf("median overlapped : %.1f MB/s\n", omed);
    printf("overlapped/legacy : %.3fx\n", omed / lmed);
    printf("checksum parity   : PASS (%llu)\n", (unsigned long long)expect);
    return 0;
}
