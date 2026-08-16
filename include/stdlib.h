/* Native-Windows stdlib compatibility for the one POSIX allocator K3 uses.
 * POSIX targets get their real system header unchanged.
 */
#ifndef K3_COMPAT_STDLIB_H
#define K3_COMPAT_STDLIB_H

#include_next <stdlib.h>

#if defined(_WIN32)
#include <errno.h>
#include <stddef.h>

/* Windows streaming currently uses buffered CRT I/O, so Linux's O_DIRECT destination
 * alignment is not a correctness requirement here. Use malloc deliberately: every K3
 * owner already releases these arenas with free(), and _aligned_malloc would require a
 * different teardown API. When a native unbuffered Win32 backend lands, its buffers will
 * use a paired Win32 allocator rather than changing this compatibility contract. */
static inline int posix_memalign(void **memptr, size_t alignment, size_t size)
{
    (void)alignment;
    if (!memptr) return EINVAL;
    void *p = malloc(size);
    if (!p) { *memptr = NULL; return ENOMEM; }
    *memptr = p;
    return 0;
}
#endif

#endif /* K3_COMPAT_STDLIB_H */
