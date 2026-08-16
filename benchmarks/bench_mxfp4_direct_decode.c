#define _POSIX_C_SOURCE 200809L
#include "k3.h"
#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif

static double now_s(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }
static uint32_t rs=0x12345678u;
static uint32_t rnd(void){ uint32_t x=rs; x^=x<<13; x^=x>>17; x^=x<<5; return rs=x; }

static void candidate(float *y,const float *x,const unsigned char *packed,
                      const unsigned char *scales,int in,int rows,int group)
{
    if(group!=32 || (in&31)) { k3_matmul_mxfp4(y,x,packed,scales,in,rows,group); return; }
    const int pcols=in/2, ngrp=in/32;
    const __m128i mask=_mm_set1_epi8(0x0f);
    const __m128i lut=_mm_setr_epi8(0,1,2,3,4,6,8,12, 0,-1,-2,-3,-4,-6,-8,-12);
    const __m256d half=_mm256_set1_pd(0.5);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(rows>64)
#endif
    for(int r=0;r<rows;r++){
        const unsigned char *pr=packed+(size_t)r*pcols;
        const unsigned char *sr=scales+(size_t)r*ngrp;
        double acc=0.0;
        for(int g=0;g<ngrp;g++){
            const unsigned char sb=sr[g];
            if(sb==255) continue;
            const unsigned char *pb=pr+(size_t)g*16;
            const float *xg=x+(size_t)g*32;
            __m128i b=_mm_loadu_si128((const __m128i*)pb);
            __m128i lo=_mm_shuffle_epi8(lut,_mm_and_si128(b,mask));
            __m128i hi=_mm_shuffle_epi8(lut,_mm_and_si128(_mm_srli_epi16(b,4),mask));
            __m128i q0=_mm_unpacklo_epi8(lo,hi);
            __m128i q1=_mm_unpackhi_epi8(lo,hi);
            __m256d v0=_mm256_setzero_pd(), v1=_mm256_setzero_pd();
#define DO4(V,Q,SHIFT,XOFF) do { \
            __m128i qq=_mm_cvtepi8_epi32(_mm_srli_si128((Q),(SHIFT))); \
            __m256d qw=_mm256_mul_pd(_mm256_cvtepi32_pd(qq),half); \
            (V)=_mm256_fmadd_pd(qw,_mm256_cvtps_pd(_mm_loadu_ps(xg+(XOFF))),(V)); \
        } while(0)
            DO4(v0,q0,0,0);   DO4(v1,q0,4,4);
            DO4(v0,q0,8,8);   DO4(v1,q0,12,12);
            DO4(v0,q1,0,16);  DO4(v1,q1,4,20);
            DO4(v0,q1,8,24);  DO4(v1,q1,12,28);
#undef DO4
            double a[4];
            _mm256_storeu_pd(a,_mm256_add_pd(v0,v1));
            const double sub=(a[0]+a[1])+(a[2]+a[3]);
            const double scale=ldexp(1.0,(int)sb-127);
            acc += sub*scale;
        }
        y[r]=(float)acc;
    }
}

static unsigned long long hashf(const float *v,int n){
    unsigned long long h=1469598103934665603ull;
    for(int i=0;i<n;i++){ union{float f; uint32_t u;} b; b.f=v[i]; for(int k=0;k<4;k++){ h^=(b.u>>(8*k))&255u; h*=1099511628211ull; }}
    return h;
}

int main(void){
    enum { IN=3584, ROWS=3072, GROUP=32, REPS=8 };
    const int pcols=IN/2, ngrp=IN/GROUP;
    unsigned char *pk=malloc((size_t)ROWS*pcols), *sc=malloc((size_t)ROWS*ngrp);
    float *x=malloc((size_t)IN*sizeof(float)), *a=malloc((size_t)ROWS*sizeof(float)), *b=malloc((size_t)ROWS*sizeof(float));
    if(!pk||!sc||!x||!a||!b) return 2;
    for(size_t i=0;i<(size_t)ROWS*pcols;i++) pk[i]=(unsigned char)rnd();
    for(size_t i=0;i<(size_t)ROWS*ngrp;i++){ unsigned z=rnd()%9; sc[i]=(unsigned char)(123+z); if((rnd()&4095u)==0) sc[i]=255; }
    for(int i=0;i<IN;i++) x[i]=((int)(rnd()&0xffff)-32768)*(1.0f/65536.0f);

    k3_matmul_mxfp4(a,x,pk,sc,IN,ROWS,GROUP);
    candidate(b,x,pk,sc,IN,ROWS,GROUP);
    printf("old hash %016llx\nnew hash %016llx\n",hashf(a,ROWS),hashf(b,ROWS));
    if(memcmp(a,b,(size_t)ROWS*sizeof(float))){
        int bad=0; for(int i=0;i<ROWS;i++) if(memcmp(a+i,b+i,sizeof(float))){ if(bad<8) printf("mismatch row %d old %.9g new %.9g\n",i,a[i],b[i]); bad++; }
        printf("BIT PARITY FAIL: %d/%d rows\n",bad,ROWS); return 1;
    }
    puts("BIT PARITY PASS");

    double oldt=0,newt=0;
    for(int r=0;r<REPS;r++){
        double t=now_s(); k3_matmul_mxfp4(a,x,pk,sc,IN,ROWS,GROUP); oldt+=now_s()-t;
        t=now_s(); candidate(b,x,pk,sc,IN,ROWS,GROUP); newt+=now_s()-t;
    }
    oldt/=REPS; newt/=REPS;
#ifdef _OPENMP
    printf("threads %d\n",omp_get_max_threads());
#endif
    printf("old %.3f ms new %.3f ms speedup %.3fx\n",oldt*1e3,newt*1e3,oldt/newt);
    printf("projected 16*3*92: old %.3f s/token new %.3f s/token saved %.3f s/token (compute-only projection)\n",
           oldt*16*3*92,newt*16*3*92,(oldt-newt)*16*3*92);
    free(pk);free(sc);free(x);free(a);free(b);return 0;
}
