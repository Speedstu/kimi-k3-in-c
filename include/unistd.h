/* Native-Windows compatibility for the tiny unistd surface K3 needs. */
#ifndef K3_COMPAT_UNISTD_H
#define K3_COMPAT_UNISTD_H

#include_next <unistd.h>

#if defined(_WIN32)
#include <windows.h>

#ifndef _SC_PAGESIZE
#define _SC_PAGESIZE 30
#endif

static inline long k3_sysconf(int name)
{
    if (name != _SC_PAGESIZE) return -1;
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    return (long)si.dwPageSize;
}
#define sysconf k3_sysconf
#endif

#endif /* K3_COMPAT_UNISTD_H */
