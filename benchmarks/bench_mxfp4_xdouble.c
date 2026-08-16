#define _POSIX_C_SOURCE 200809L
#include "k3.h"
#include <immintrin.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif

static double now_s(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
static uint32_t rs=0x51f15e5du;
static uint32_t rnd(void){uint32_t x=rs;x^=x<<13;x^=x>>17;x^=x<<5;return rs=x;}
static unsigned long long hashf(const float*v,int n){unsigned long long h=1469598103934665603ull;for(int i=0;i<n;i++){union{float f;uint32_t u;}b;b.f=v[i];for(int k=0;k<4;k++){h^=(b.u>>(8*k))&255u;h*=1099511628211ull;}}return h;}
static double e8half[256];
static void init_scale(void){for(int i=0;i<255;i++){union{uint32_t u;float f;}v;if(i==0)v.f=0x1p-127f;else v.u=(uint32_t)i<<23;e8half[i]=(double)v.f*0.5;}e8half[255]=0.0;}

static void cand(float*y,const float*x,const unsigned char*packed,const unsigned char*scales,int in,int rows){
    const int pcols=in/2,ngrp=in/32;
    double *xd=(double*)malloc((size_t)in*sizeof(double)); if(!xd){abort();}
    for(int i=0;i<in;i++)xd[i]=(double)x[i];
    const __m128i mask=_mm_set1_epi8(0x0f);
    const __m128i lut=_mm_setr_epi8(0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(rows>64)
#endif
    for(int r=0;r<rows;r++){
        const unsigned char*pr=packed+(size_t)r*pcols;const unsigned char*sr=scales+(size_t)r*ngrp;double acc=0.0;
        for(int g=0;g<ngrp;g++){
            const unsigned char sb=sr[g];if(sb==255)continue;
            const unsigned char*pb=pr+(size_t)g*16;const double*xg=xd+(size_t)g*32;
            const __m128i b=_mm_loadu_si128((const __m128i*)pb);
            const __m128i lo=_mm_shuffle_epi8(lut,_mm_and_si128(b,mask));
            const __m128i hi=_mm_shuffle_epi8(lut,_mm_and_si128(_mm_srli_epi16(b,4),mask));
            const __m128i q0=_mm_unpacklo_epi8(lo,hi),q1=_mm_unpackhi_epi8(lo,hi);
            const __m256i i0=_mm256_cvtepi8_epi32(q0),i1=_mm256_cvtepi8_epi32(_mm_srli_si128(q0,8));
            const __m256i i2=_mm256_cvtepi8_epi32(q1),i3=_mm256_cvtepi8_epi32(_mm_srli_si128(q1,8));
            __m256d v0=_mm256_setzero_pd(),v1=_mm256_setzero_pd();
#define F4(V,I128,O) do{(V)=_mm256_fmadd_pd(_mm256_cvtepi32_pd((I128)),_mm256_loadu_pd(xg+(O)),(V));}while(0)
            F4(v0,_mm256_castsi256_si128(i0),0);F4(v1,_mm256_extracti128_si256(i0,1),4);
            F4(v0,_mm256_castsi256_si128(i1),8);F4(v1,_mm256_extracti128_si256(i1,1),12);
            F4(v0,_mm256_castsi256_si128(i2),16);F4(v1,_mm256_extracti128_si256(i2,1),20);
            F4(v0,_mm256_castsi256_si128(i3),24);F4(v1,_mm256_extracti128_si256(i3,1),28);
#undef F4
            double a[4];_mm256_storeu_pd(a,_mm256_add_pd(v0,v1));const double sub2=(a[0]+a[1])+(a[2]+a[3]);acc+=sub2*e8half[sb];
        }
        y[r]=(float)acc;
    }
    free(xd);
}

int main(void){enum{IN=3584,ROWS=3072,REPS=10};const int pcols=IN/2,ngrp=IN/32;init_scale();
unsigned char*pk=malloc((size_t)ROWS*pcols),*sc=malloc((size_t)ROWS*ngrp);float*x=malloc((size_t)IN*4),*a=malloc((size_t)ROWS*4),*b=malloc((size_t)ROWS*4);if(!pk||!sc||!x||!a||!b)return 2;
for(size_t i=0;i<(size_t)ROWS*pcols;i++)pk[i]=(unsigned char)rnd();for(size_t i=0;i<(size_t)ROWS*ngrp;i++){sc[i]=(unsigned char)(120+rnd()%15);if((rnd()&4095u)==0)sc[i]=255;}for(int i=0;i<IN;i++)x[i]=((int)(rnd()&0xffffu)-32768)*(1.0f/65536.0f);
k3_matmul_mxfp4(a,x,pk,sc,IN,ROWS,32);cand(b,x,pk,sc,IN,ROWS);printf("base=%016llx cand=%016llx\n",hashf(a,ROWS),hashf(b,ROWS));if(memcmp(a,b,(size_t)ROWS*4)){puts("PARITY FAIL");return 1;}puts("PARITY PASS");
double ta=0,tb=0;for(int r=0;r<REPS;r++){double t=now_s();k3_matmul_mxfp4(a,x,pk,sc,IN,ROWS,32);ta+=now_s()-t;t=now_s();cand(b,x,pk,sc,IN,ROWS);tb+=now_s()-t;}ta/=REPS;tb/=REPS;
#ifdef _OPENMP
printf("threads=%d ",omp_get_max_threads());
#endif
printf("base=%.3fms xdouble=%.3fms speedup=%.3fx\n",ta*1e3,tb*1e3,ta/tb);return 0;}
