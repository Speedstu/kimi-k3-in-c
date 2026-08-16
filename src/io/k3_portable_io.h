/* k3_portable_io.h - shims for platform-specific low-level I/O.
 *
 * The engine wants three properties from its readers:
 *
 *   - positioned reads (pread semantics), because trunk/expert I/O can overlap;
 *   - a best-effort way to bypass the page cache for one-shot streamed weights;
 *   - binary file descriptors, mandatory on Windows for safetensors/trunk bytes.
 *
 * Linux supplies these directly. Darwin spells the no-cache hint differently. Native
 * Windows has no pread() in the UCRT, so K3 opens model files with FILE_FLAG_OVERLAPPED
 * and issues ReadFile requests with an explicit 64-bit offset. That removes the shared
 * file-pointer lock from the correctness-first Windows port: two expert readers hitting
 * the same shard can now be in flight at the same time.
 *
 * Windows is still BUFFERED here. FILE_FLAG_NO_BUFFERING has stricter offset/length/
 * buffer-alignment contracts and belongs behind a separate measured gate; exactness does
 * not depend on it. The resident worker separately uses unbuffered stdout on Win32
 * because the Microsoft CRT treats _IOLBF as full buffering.
 */
#ifndef K3_PORTABLE_IO_H
#define K3_PORTABLE_IO_H

#if defined(__APPLE__) && !defined(_DARWIN_C_SOURCE)
#define _DARWIN_C_SOURCE
#endif

#include <fcntl.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>

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

/* Keep the fd-shaped interface used by the rest of K3, but create the underlying native
 * handle ourselves so it is FILE_FLAG_OVERLAPPED. _open_osfhandle transfers HANDLE
 * ownership to the CRT descriptor; the existing close(fd) teardown remains correct. */
static inline int k3_open_read(const char *path, int want_direct, int *direct_active)
{
    (void)want_direct; /* no unbuffered Windows path until its alignment gate lands */
    if (direct_active) *direct_active = 0;

    HANDLE h = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING,
                           FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OVERLAPPED, NULL);
    if (h == INVALID_HANDLE_VALUE) return -1;

    const int fd = _open_osfhandle((intptr_t)h, _O_RDONLY | _O_BINARY);
    if (fd < 0) {
        CloseHandle(h);
        return -1;
    }
    return fd;
}

static inline int64_t k3_file_size(int fd)
{
    const intptr_t os = _get_osfhandle(fd);
    if (os == -1) return -1;
    LARGE_INTEGER n;
    if (!GetFileSizeEx((HANDLE)os, &n)) return -1;
    return (int64_t)n.QuadPart;
}

/* Kept for call sites that only need to force binary mode. Native no-cache support is
 * deliberately NOT implied by success here. */
static inline int k3_set_direct(int fd)
{
    if (fd < 0) return -1;
    return _setmode(fd, _O_BINARY) < 0 ? -1 : 0;
}

/* True positioned I/O on the same native handle. Each request has its own event and its
 * own OVERLAPPED offset, so concurrent calls do not share or mutate a file pointer.
 *
 * ReadFile's count is a DWORD. Cap individual requests below 2 GiB and continue until
 * count bytes are satisfied; this also keeps huge 1+ GiB trunk reads away from edge-case
 * transfer limits in storage drivers. */
static inline ssize_t k3_pread(int fd, void *buf, size_t count, int64_t offset)
{
    const intptr_t os = _get_osfhandle(fd);
    if (os == -1 || offset < 0) return (ssize_t)-1;
    HANDLE h = (HANDLE)os;
    HANDLE event = CreateEventA(NULL, TRUE, FALSE, NULL);
    if (!event) return (ssize_t)-1;

    size_t done = 0;
    int failed = 0;
    while (done < count) {
        const uint64_t at = (uint64_t)offset + (uint64_t)done;
        const size_t remain = count - done;
        const DWORD take = (DWORD)(remain > (size_t)0x7ffff000u ?
                                   (size_t)0x7ffff000u : remain);
        OVERLAPPED ov;
        memset(&ov, 0, sizeof ov);
        ov.Offset = (DWORD)(at & 0xffffffffu);
        ov.OffsetHigh = (DWORD)(at >> 32);
        ov.hEvent = event;
        ResetEvent(event);

        BOOL started = ReadFile(h, (char *)buf + done, take, NULL, &ov);
        if (!started) {
            const DWORD err = GetLastError();
            if (err == ERROR_HANDLE_EOF) break;
            if (err != ERROR_IO_PENDING) { failed = 1; break; }
        }

        DWORD got = 0;
        if (!GetOverlappedResult(h, &ov, &got, TRUE)) {
            const DWORD err = GetLastError();
            if (err == ERROR_HANDLE_EOF) break;
            failed = 1;
            break;
        }
        if (got == 0) break;
        done += (size_t)got;
        if (got < take) break;
    }

    CloseHandle(event);
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

static inline int k3_open_read(const char *path, int want_direct, int *direct_active)
{
    int fd = open(path, O_RDONLY);
    int active = 0;
    if (fd >= 0 && want_direct && fcntl(fd, F_NOCACHE, 1) == 0) active = 1;
    if (direct_active) *direct_active = active;
    return fd;
}

static inline int64_t k3_file_size(int fd)
{
    struct stat st;
    return fstat(fd, &st) == 0 ? (int64_t)st.st_size : -1;
}

/* Darwin's O_DIRECT equivalent, applied after open(). Failure is not fatal: callers
 * keep the descriptor and read through the page cache instead. */
static inline int k3_set_direct(int fd)
{
    if (fd < 0) return -1;
    return fcntl(fd, F_NOCACHE, 1);
}

#else /* Linux and other POSIX targets */

static inline int k3_open_read(const char *path, int want_direct, int *direct_active)
{
    int fd = -1;
    int active = 0;
    if (want_direct) {
        fd = open(path, O_RDONLY | O_DIRECT);
        if (fd >= 0) active = 1;
    }
    if (fd < 0) fd = open(path, O_RDONLY);
    if (direct_active) *direct_active = active;
    return fd;
}

static inline int64_t k3_file_size(int fd)
{
    struct stat st;
    return fstat(fd, &st) == 0 ? (int64_t)st.st_size : -1;
}

static inline int k3_set_direct(int fd) { (void)fd; return 0; }

#endif

#endif /* K3_PORTABLE_IO_H */