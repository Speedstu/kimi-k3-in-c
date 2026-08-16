/* Minimal native-Windows <sys/resource.h> compatibility for K3's peak RSS metric. */
#ifndef K3_COMPAT_SYS_RESOURCE_H
#define K3_COMPAT_SYS_RESOURCE_H

#if !defined(_WIN32)
#  include_next <sys/resource.h>
#else

#include <stdint.h>
#include <windows.h>
#include <psapi.h>

#ifndef RUSAGE_SELF
#define RUSAGE_SELF 0
#endif

struct rusage {
    int64_t ru_maxrss; /* kilobytes, matching Linux semantics expected by k3_run.c */
};

static inline int getrusage(int who, struct rusage *ru)
{
    if (who != RUSAGE_SELF || !ru) return -1;

    typedef BOOL (WINAPI *K3PMI)(HANDLE, PPROCESS_MEMORY_COUNTERS, DWORD);
    K3PMI fn = NULL;
    HMODULE mod = GetModuleHandleA("kernel32.dll");
    if (mod) fn = (K3PMI)(void *)GetProcAddress(mod, "K32GetProcessMemoryInfo");

    HMODULE psapi = NULL;
    if (!fn) {
        psapi = LoadLibraryA("psapi.dll");
        if (psapi) fn = (K3PMI)(void *)GetProcAddress(psapi, "GetProcessMemoryInfo");
    }
    if (!fn) {
        if (psapi) FreeLibrary(psapi);
        return -1;
    }

    PROCESS_MEMORY_COUNTERS pmc;
    pmc.cb = sizeof pmc;
    const BOOL ok = fn(GetCurrentProcess(), &pmc, (DWORD)sizeof pmc);
    if (psapi) FreeLibrary(psapi);
    if (!ok) return -1;

    ru->ru_maxrss = (int64_t)(pmc.PeakWorkingSetSize / 1024u);
    return 0;
}

#endif /* _WIN32 */
#endif /* K3_COMPAT_SYS_RESOURCE_H */
