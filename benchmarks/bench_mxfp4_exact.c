/* Permanent MXFP4 exactness/perf probe at both released K3 expert matrix shapes.
 * Compile this file + k3_ops.c with AVX2 and without AVX2; the printed hashes MUST match.
 * That compares the direct-nibble fast path against the unchanged generic/scalar path. */
#define _POSIX_C_SOURCE 200809L
#include "k3.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif

static double now_s(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }
static uint32_t rs=0x6d2b79f5u;
static uint32_t rnd(void){ uint32_t x=rs; x^=x<<13; x^=x>>17; x^=x<<5; return rs=x; }
static unsigned long long hashf(const float *v,int n){
    unsigned long long h=1469598103934665603ull;
    for(int i=0;i<n;i++){ union{float f; uint32_t u;} b; b.f=v[i]; for(int k=0;k<4;k++){ h^=(b.u>>(8*k))&255u; h*=1099511628211ull; }}
    return h;
}

static int one(const char *name,int in,int rows){
    const int group=K3_MXFP4_GROUP, pcols=in/2, ngrp=(in+group-1)/group;
    unsigned char *pk=malloc((size_t)rows*pcols), *sc=malloc((size_t)rows*ngrp);
    float *x=malloc((size_t)in*sizeof(float)), *y=malloc((size_t)rows*sizeof(float));
    if(!pk||!sc||!x||!y) return 2;
    for(size_t i=0;i<(size_t)rows*pcols;i++) pk[i]=(unsigned char)rnd();
    for(size_t i=0;i<(size_t)rows*ngrp;i++){
        sc[i]=(unsigned char)(120+(rnd()%15));
        if((rnd()&2047u)==0) sc[i]=255; /* exercise NaN/zero-scale skip */
    }
    for(int i=0;i<in;i++) x[i]=((int)(rnd()&0xffffu)-32768)*(1.0f/65536.0f);
    k3_matmul_mxfp4(y,x,pk,sc,in,rows,group); /* warm */
    const int reps=6;
    double t=now_s();
    for(int r=0;r<reps;r++) k3_matmul_mxfp4(y,x,pk,sc,in,rows,group);
    const double dt=(now_s()-t)/reps;
    printf("%s in=%d rows=%d hash=%016llx time_ms=%.3f\n",name,in,rows,hashf(y,rows),dt*1e3);
    free(pk);free(sc);free(x);free(y);return 0;
}

int main(void){
#ifdef __AVX2__
    puts("isa=avx2");
#else
    puts("isa=scalar");
#endif
#ifdef _OPENMP
    printf("threads=%d\n",omp_get_max_threads());
#endif
    if(one("w1w3",3584,3072)) return 2;
    /* Reset RNG so this second shape is deterministic across independently built binaries. */
    rs=0x13579bdfu;
    if(one("w2",3072,3584)) return 2;
    return 0;
}
