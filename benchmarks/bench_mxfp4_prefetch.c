#define _POSIX_C_SOURCE 200809L
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
static double now_s(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
static uint32_t rs=0x27182818u; static uint32_t rnd(void){uint32_t x=rs;x^=x<<13;x^=x>>17;x^=x<<5;return rs=x;}
static double hs[256]; static void init_s(void){for(int i=0;i<256;i++)hs[i]=(i==255)?0.0:ldexp(1.0,i-127)*0.5;}
static void run(float*y,const double*x,const unsigned char*pk,const unsigned char*sc,int in,int rows,int ahead){
 const int pb=in/2,ng=in/32;const __m128i mask=_mm_set1_epi8(0x0f),lut=_mm_setr_epi8(0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(rows>64)
#endif
 for(int r=0;r<rows;r++){const unsigned char*pr=pk+(size_t)r*pb,*sr=sc+(size_t)r*ng;double acc=0.0;
  for(int g=0;g<ng;g++){if(ahead&&g+ahead<ng){__builtin_prefetch(pr+(size_t)(g+ahead)*16,0,0);__builtin_prefetch(sr+g+ahead,0,0);}unsigned char sb=sr[g];if(sb==255)continue;const unsigned char*p=pr+(size_t)g*16,*dummy=p; (void)dummy;const double*xg=x+(size_t)g*32;
   __m128i b=_mm_loadu_si128((const __m128i*)p),lo=_mm_shuffle_epi8(lut,_mm_and_si128(b,mask)),hi=_mm_shuffle_epi8(lut,_mm_and_si128(_mm_srli_epi16(b,4),mask)),q0=_mm_unpacklo_epi8(lo,hi),q1=_mm_unpackhi_epi8(lo,hi);__m256i i0=_mm256_cvtepi8_epi32(q0),i1=_mm256_cvtepi8_epi32(_mm_srli_si128(q0,8)),i2=_mm256_cvtepi8_epi32(q1),i3=_mm256_cvtepi8_epi32(_mm_srli_si128(q1,8));__m256d v0=_mm256_setzero_pd(),v1=_mm256_setzero_pd();
#define F(V,I,O) do{(V)=_mm256_fmadd_pd(_mm256_cvtepi32_pd((I)),_mm256_loadu_pd(xg+(O)),(V));}while(0)
   F(v0,_mm256_castsi256_si128(i0),0);F(v1,_mm256_extracti128_si256(i0,1),4);F(v0,_mm256_castsi256_si128(i1),8);F(v1,_mm256_extracti128_si256(i1,1),12);F(v0,_mm256_castsi256_si128(i2),16);F(v1,_mm256_extracti128_si256(i2,1),20);F(v0,_mm256_castsi256_si128(i3),24);F(v1,_mm256_extracti128_si256(i3,1),28);
#undef F
   double a[4];_mm256_storeu_pd(a,_mm256_add_pd(v0,v1));acc+=((a[0]+a[1])+(a[2]+a[3]))*hs[sb];}y[r]=(float)acc;}}
static unsigned long long hf(const float*v,int n){unsigned long long h=1469598103934665603ull;for(int i=0;i<n;i++){union{float f;uint32_t u;}b;b.f=v[i];for(int k=0;k<4;k++){h^=(b.u>>(8*k))&255u;h*=1099511628211ull;}}return h;}
int main(void){enum{IN=3584,ROWS=3072,REPS=10};const size_t np=(size_t)ROWS*(IN/2),ns=(size_t)ROWS*(IN/32);init_s();unsigned char*pk=malloc(np),*sc=malloc(ns);double*x=malloc((size_t)IN*8);float*y=malloc((size_t)ROWS*4),*ref=malloc((size_t)ROWS*4);if(!pk||!sc||!x||!y||!ref)return 2;for(size_t i=0;i<np;i++)pk[i]=(unsigned char)rnd();for(size_t i=0;i<ns;i++)sc[i]=(unsigned char)(120+rnd()%15);for(int i=0;i<IN;i++){float f=((int)(rnd()&65535)-32768)*(1.0f/65536.0f);x[i]=(double)f;}run(ref,x,pk,sc,IN,ROWS,0);printf("ref=%016llx\n",hf(ref,ROWS));int ds[4]={0,4,8,16};for(int di=0;di<4;di++){int d=ds[di];run(y,x,pk,sc,IN,ROWS,d);if(memcmp(ref,y,(size_t)ROWS*4)){printf("d=%d PARITY FAIL\n",d);return 1;}double dt=0;for(int q=0;q<REPS;q++){double t=now_s();run(y,x,pk,sc,IN,ROWS,d);dt+=now_s()-t;}printf("ahead=%d time_ms=%.3f\n",d,dt*1e3/REPS);}return 0;}
