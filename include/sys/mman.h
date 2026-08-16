/* Native-Windows compatibility for the small POSIX mmap surface K3 uses.
 *
 * This header shadows <sys/mman.h> only because include/ precedes system include paths.
 * POSIX builds immediately include the real system header with include_next.
 */
#ifndef K3_COMPAT_SYS_MMAN_H
#define K3_COMPAT_SYS_MMAN_H

#if !defined(_WIN32)
#  include_next <sys/mman.h>
#else

#include <stddef.h>
#include <stdint.h>
#include <windows.h>

#ifndef PROT_READ
#define PROT_READ  0x1
#endif
#ifndef PROT_WRITE
#define PROT_WRITE 0x2
#endif
#ifndef MAP_PRIVATE
#define MAP_PRIVATE 0x02
#endif
#ifndef MAP_ANONYMOUS
#define MAP_ANONYMOUS 0x20
#endif
#ifndef MAP_ANON
#define MAP_ANON MAP_ANONYMOUS
#endif
#ifndef MAP_NORESERVE
#define MAP_NORESERVE 0x4000
#endif
#ifndef MAP_FAILED
#define MAP_FAILED ((void *)(intptr_t)-1)
#endif
#ifndef MADV_DONTNEED
#define MADV_DONTNEED 4
#endif
#ifndef MADV_HUGEPAGE
#define MADV_HUGEPAGE 14
#endif

/* K3 only mmaps anonymous read/write worker KV. MEM_COMMIT reserves pagefile charge but
 * Windows still allocates physical pages on first access, so ordinary contexts remain
 * demand-paged. The worker has its own large-context guard and will get a clean failure
 * if Windows cannot commit the requested reservation rather than falling into a giant
 * calloc. */
static inline void *mmap(void *addr, size_t length, int prot, int flags, int fd,
                         int64_t offset)
{
    (void)flags;
    if (length == 0 || fd != -1 || offset != 0 ||
        !(prot & PROT_READ) || !(prot & PROT_WRITE))
        return MAP_FAILED;
    void *p = VirtualAlloc(addr, length, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    return p ? p : MAP_FAILED;
}

static inline int munmap(void *addr, size_t length)
{
    (void)length;
    return VirtualFree(addr, 0, MEM_RELEASE) ? 0 : -1;
}

/* MADV_HUGEPAGE is a Linux performance hint and has no correctness effect here.
 * MADV_DONTNEED is used only for stale worker KV after cached=0 made it unreachable.
 * MEM_RESET tells Windows those committed pages are discardable without walking and
 * zeroing the old KV in userspace; their future contents are deliberately unspecified. */
static inline int madvise(void *addr, size_t length, int advice)
{
    if (!addr || length == 0) return 0;
    if (advice == MADV_DONTNEED)
        return VirtualAlloc(addr, length, MEM_RESET, PAGE_READWRITE) ? 0 : -1;
    return 0;
}

#endif /* _WIN32 */
#endif /* K3_COMPAT_SYS_MMAN_H */
