/* k3_portable_io.h - shims for platform-specific low-level I/O.
 *
 * The engine wants three properties from its readers:
 *
 *   - positioned reads (pread semantics), because trunk/expert I/O can overlap;
 *   - a best-effort way to bypass the page cache for one-shot streamed weights;
 *   - binary file descriptors, mandatory on Windows for safetensors/trunk bytes.
 *
 * Linux supplies these directly. Darwin spells the no-cache hint differently. Native
 * Windows has no pread() in the UCRT, so the compatibility path serializes seek+read
 * per descriptor while preserving the descriptor's original position. This is the
 * correctness baseline; a later Win32 backend may replace it with overlapped handles
 * without changing callers or model math. The resident worker separately uses
 * unbuffered stdout on Win32 because the Microsoft CRT treats _IOLBF as full buffering.
 */
#ifndef K3_PORTABLE_IO_H
#define K3_PORTABLE_IO_H

#if defined(__APPLE__) && !defined(_DARWIN_C_SOURCE)
#define _DARWIN_C_SOURCE
#endif

#include <fcntl.h>
#include <stdint.h>
#include <stddef.h>
#include <sys/types.h>

#if defined(_WIN32)

#include <io.h>
#include <limits.h>
#include <windows.h>

#ifndef O_DIRECT
#define O_DIRECT 0
#endif
#ifndef POSIX_FADV_WILLNEED
#define POSIX_FADV_WILLNEED 3
#endif

static inline int posix_fadvise(int fd, int64_t off, int64_t len, int advice)
{
    (void)fd; (void)off; (void)len; (void)advice;
    return 0; /* advisory only */
}

/* Windows CRT descriptors default to text mode in some environments. Every model file
 * is binary, so switch a descriptor before its first positioned read. */
static inline int k3_set_direct(int fd)
{
    if (fd < 0) return -1;
    return _setmode(fd, _O_BINARY) < 0 ? -1 : 0;
}

/* UCRT has no pread(). A seek+read pair would race between the expert prefetch workers,
 * so protect each descriptor with a small hashed SRW-lock table and restore the original
 * offset before returning. Different descriptors still read concurrently. The loop is
 * needed because _read() takes an unsigned-int byte count.
 *
 * This path is intentionally conservative: it establishes exact native-Windows parity
 * first. Once that is gated, an overlapped Win32 reader can remove the per-fd lock. */
static inline ssize_t k3_pread(int fd, void *buf, size_t count, int64_t offset)
{
    static SRWLOCK locks[64]; /* zero is SRWLOCK_INIT */
    SRWLOCK *lock = &locks[(unsigned)fd & 63u];
    AcquireSRWLockExclusive(lock);

    (void)_setmode(fd, _O_BINARY);
    const __int64 saved = _lseeki64(fd, 0, SEEK_CUR);
    if (saved < 0 || _lseeki64(fd, (__int64)offset, SEEK_SET) < 0) {
        ReleaseSRWLockExclusive(lock);
        return (ssize_t)-1;
    }

    size_t done = 0;
    int failed = 0;
    while (done < count) {
        const size_t remain = count - done;
        const unsigned int take = (unsigned int)(remain > (size_t)INT_MAX ? INT_MAX : remain);
        const int got = _read(fd, (char *)buf + done, take);
        if (got <= 0) {
            failed = (got < 0);
            break;
        }
        done += (size_t)got;
        if ((unsigned int)got < take) break;
    }

    (void)_lseeki64(fd, saved, SEEK_SET);
    ReleaseSRWLockExclusive(lock);
    if (failed && done == 0) return (ssize_t)-1;
    return (ssize_t)done;
}

#define pread k3_pread

#elif defined(__APPLE__)

#ifndef O_DIRECT
#define O_DIRECT 0
#endif

#ifndef POSIX_FADV_WILLNEED
#define POSIX_FADV_WILLNEED 3
#endif

static inline int posix_fadvise(int fd, off_t off, off_t len, int advice)
{
    (void)fd; (void)off; (void)len; (void)advice;
    return 0;
}

/* Darwin's O_DIRECT equivalent, applied after open(). Failure is not fatal: callers
 * keep the descriptor and read through the page cache instead. */
static inline int k3_set_direct(int fd)
{
    if (fd < 0) return -1;
    return fcntl(fd, F_NOCACHE, 1);
}

#else /* Linux and other POSIX targets */

static inline int k3_set_direct(int fd) { (void)fd; return 0; }

#endif

#endif /* K3_PORTABLE_IO_H */