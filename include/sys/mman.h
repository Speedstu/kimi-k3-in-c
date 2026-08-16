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

/* Generic mmap compatibility remains reserve+commit so callers that expect immediately
 * writable anonymous memory keep POSIX-like semantics. The resident worker uses the
 * explicit k3_vm_* helpers below when it wants a huge address-space reservation whose
 * commit charge must grow only with rows actually reached. */
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

/* Reserve virtual address space with zero commit charge. PAGE_NOACCESS is intentional:
 * touching a row before k3_vm_commit_span() is a hard bug instead of silently consuming
 * multi-terabyte commit. */
static inline void *k3_vm_reserve(size_t length)
{
    if (length == 0) return NULL;
    return VirtualAlloc(NULL, length, MEM_RESERVE, PAGE_NOACCESS);
}

static inline size_t k3_vm_page_size(void)
{
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    return (size_t)si.dwPageSize;
}

/* Commit only a subrange of an existing reservation. Clamp the page-rounded span to the
 * reservation itself so a final unaligned row can never spill into a neighbouring VA
 * region. Re-committing an already committed page is valid and keeps this helper
 * idempotent across speculative replay. */
static inline int k3_vm_commit_span(void *base, size_t total,
                                    void *addr, size_t length)
{
    if (!base || total == 0 || !addr || length == 0) return 0;
    const size_t ps = k3_vm_page_size();
    const uintptr_t b0 = (uintptr_t)base;
    const uintptr_t b1 = b0 + total;
    uintptr_t a0 = (uintptr_t)addr;
    if (a0 < b0 || a0 >= b1 || length > (size_t)(b1 - a0)) return -1;
    uintptr_t a1 = a0 + length;
    a0 = (a0 / ps) * ps;
    a1 = ((a1 + ps - 1) / ps) * ps;
    if (a0 < b0) a0 = b0;
    if (a1 > b1) a1 = b1;
    if (a1 <= a0) return 0;
    return VirtualAlloc((void *)a0, (size_t)(a1 - a0),
                        MEM_COMMIT, PAGE_READWRITE) ? 0 : -1;
}

/* Drop commit charge for a subrange while keeping the enclosing address reservation.
 * Future use must call k3_vm_commit_span() again before access. */
static inline int k3_vm_decommit_span(void *base, size_t total,
                                      void *addr, size_t length)
{
    if (!base || total == 0 || !addr || length == 0) return 0;
    const size_t ps = k3_vm_page_size();
    const uintptr_t b0 = (uintptr_t)base;
    const uintptr_t b1 = b0 + total;
    uintptr_t a0 = (uintptr_t)addr;
    if (a0 < b0 || a0 >= b1 || length > (size_t)(b1 - a0)) return -1;
    uintptr_t a1 = a0 + length;
    a0 = (a0 / ps) * ps;
    a1 = ((a1 + ps - 1) / ps) * ps;
    if (a0 < b0) a0 = b0;
    if (a1 > b1) a1 = b1;
    if (a1 <= a0) return 0;
    return VirtualFree((void *)a0, (size_t)(a1 - a0), MEM_DECOMMIT) ? 0 : -1;
}

/* Generic MADV_DONTNEED retains immediately-accessible semantics via MEM_RESET. The
 * worker uses k3_vm_decommit_span() explicitly when it wants commit charge returned. */
static inline int madvise(void *addr, size_t length, int advice)
{
    if (!addr || length == 0) return 0;
    if (advice == MADV_DONTNEED)
        return VirtualAlloc(addr, length, MEM_RESET, PAGE_READWRITE) ? 0 : -1;
    return 0;
}

#endif /* _WIN32 */
#endif /* K3_COMPAT_SYS_MMAN_H */
