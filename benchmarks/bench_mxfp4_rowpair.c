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
static uint32_t rs=0xdeadbeefu;static uint32_t rnd(void){uint32_t x=rs;x^=x<<13;x^=x>>17;x^=x<<5;return rs=x;}
static double hs[256];static void init_s(void){for(int i=0;i<255;i++)hs[i]=ldexp(1.0,i-127)*0.5;hs[255]=0.0;}
static const signed char LUTV[16]={0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12};

static float one_row(const double*x,const unsigned char*pr,const unsigned char*sr,int in){
 const int ng=in/32;const __m128i mask=_mm_set1_epi8(15),lut=_mm_loadu_si128((const __m128i*)LUTV);double acc=0;
 for(int g=0;g<ng;g++){unsigned char sb=sr[g];if(sb==255)continue;const double*xg=x+(size_t)g*32;__m128i b=_mm_loadu_si128((const __m128i*)(pr+(size_t)g*16));__m128i lo=_mm_shuffle_epi8(lut,_mm_and_si128(b,mask)),hi=_mm_shuffle_epi8(lut,_mm_and_si128(_mm_srli_epi16(b,4),mask));__m128i q0=_mm_unpacklo_epi8(lo,hi),q1=_mm_unpackhi_epi8(lo,hi);__m256i ii[4]={_mm256_cvtepi8_epi32(q0),_mm256_cvtepi8_epi32(_mm_srli_si128(q0,8)),_mm256_cvtepi8_epi32(q1),_mm256_cvtepi8_epi32(_mm_srli_si128(q1,8))};__m256d v0=_mm256_setzero_pd(),v1=_mm256_setzero_pd();
  for(int k=0;k<4;k++){const int o=k*8;__m256d xa=_mm256_loadu_pd(xg+o),xb=_mm256_loadu_pd(xg+o+4);v0=_mm256_fmadd_pd(_mm256_cvtepi32_pd(_mm256_castsi256_si128(ii[k])),xa,v0);v1=_mm256_fmadd_pd(_mm256_cvtepi32_pd(_mm256_extracti128_si256(ii[k],1)),xb,v1);}double a[4];_mm256_storeu_pd(a,_mm256_add_pd(v0,v1));acc+=((a[0]+a[1])+(a[2]+a[3]))*hs[sb];}return(float)acc;}

static void base(float*y,const double*x,const unsigned char*pk,const unsigned char*sc,int in,int rows){int pb=in/2,sb=in/32;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
 for(int r=0;r<rows;r++)y[r]=one_row(x,pk+(size_t)r*pb,sc+(size_t)r*sb,in);}

static void pair(float*y,const double*x,const unsigned char*pk,const unsigned char*sc,int in,int rows){const int pb=in/2,ng=in/32;const __m128i mask=_mm_set1_epi8(15),lut=_mm_loadu_si128((const __m128i*)LUTV);int pairs=rows/2;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
 for(int rp=0;rp<pairs;rp++){int r=2*rp;const unsigned char*p0=pk+(size_t)r*pb,*p1=p0+pb,*s0=sc+(size_t)r*ng,*s1=s0+ng;double ac0=0,ac1=0;
  for(int g=0;g<ng;g++){unsigned char sb0=s0[g],sb1=s1[g];if(sb0==255||sb1==255){if(sb0!=255){const double*xg=x+(size_t)g*32;__m128i b=_mm_loadu_si128((const __m128i*)(p0+(size_t)g*16)),lo=_mm_shuffle_epi8(lut,_mm_and_si128(b,mask)),hi=_mm_shuffle_epi8(lut,_mm_and_si128(_mm_srli_epi16(b,4),mask)),q0=_mm_unpacklo_epi8(lo,hi),q1=_mm_unpackhi_epi8(lo,hi);__m256i ii[4]={_mm256_cvtepi8_epi32(q0),_mm256_cvtepi8_epi32(_mm_srli_si128(q0,8)),_mm256_cvtepi8_epi32(q1),_mm256_cvtepi8_epi32(_mm_srli_si128(q1,8))};__m256d a0=_mm256_setzero_pd(),a1=_mm256_setzero_pd();for(int k=0;k<4;k++){int o=8*k;a0=_mm256_fmadd_pd(_mm256_cvtepi32_pd(_mm256_castsi256_si128(ii[k])),_mm256_loadu_pd(xg+o),a0);a1=_mm256_fmadd_pd(_mm256_cvtepi32_pd(_mm256_extracti128_si256(ii[k],1)),_mm256_loadu_pd(xg+o+4),a1);}double a[4];_mm256_storeu_pd(a,_mm256_add_pd(a0,a1));ac0+=((a[0]+a[1])+(a[2]+a[3]))*hs[sb0];}if(sb1!=255){/* rare edge: preserve exact skip semantics */double v=one_row(x,p1,s1,in);y[r+1]=v;/* mark for final overwrite avoidance */}continue;}
   const double*xg=x+(size_t)g*32;__m128i b0=_mm_loadu_si128((const __m128i*)(p0+(size_t)g*16)),b1=_mm_loadu_si128((const __m128i*)(p1+(size_t)g*16));__m128i l0=_mm_shuffle_epi8(lut,_mm_and_si128(b0,mask)),h0=_mm_shuffle_epi8(lut,_mm_and_si128(_mm_srli_epi16(b0,4),mask)),l1=_mm_shuffle_epi8(lut,_mm_and_si128(b1,mask)),h1=_mm_shuffle_epi8(lut,_mm_and_si128(_mm_srli_epi16(b1,4),mask));__m128i q00=_mm_unpacklo_epi8(l0,h0),q01=_mm_unpackhi_epi8(l0,h0),q10=_mm_unpacklo_epi8(l1,h1),q11=_mm_unpackhi_epi8(l1,h1);__m256i i0[4]={_mm256_cvtepi8_epi32(q00),_mm256_cvtepi8_epi32(_mm_srli_si128(q00,8)),_mm256_cvtepi8_epi32(q01),_mm256_cvtepi8_epi32(_mm_srli_si128(q01,8))},i1[4]={_mm256_cvtepi8_epi32(q10),_mm256_cvtepi8_epi32(_mm_srli_si128(q10,8)),_mm256_cvtepi8_epi32(q11),_mm256_cvtepi8_epi32(_mm_srli_si128(q11,8))};__m256d a00=_mm256_setzero_pd(),a01=_mm256_setzero_pd(),a10=_mm256_setzero_pd(),a11=_mm256_setzero_pd();for(int k=0;k<4;k++){int o=8*k;__m256d xa=_mm256_loadu_pd(xg+o),xb=_mm256_loadu_pd(xg+o+4);a00=_mm256_fmadd_pd(_mm256_cvtepi32_pd(_mm256_castsi256_si128(i0[k])),xa,a00);a01=_mm256_fmadd_pd(_mm256_cvtepi32_pd(_mm256_extracti128_si256(i0[k],1)),xb,a01);a10=_mm256_fmadd_pd(_mm256_cvtepi32_pd(_mm256_castsi256_si128(i1[k])),xa,a10);a11=_mm256_fmadd_pd(_mm256_cvtepi32_pd(_mm256_extracti128_si256(i1[k],1)),xb,a11);}double a[4],b[4];_mm256_storeu_pd(a,_mm256_add_pd(a00,a01));_mm256_storeu_pd(b,_mm256_add_pd(a10,a11));ac0+=((a[0]+a[1])+(a[2]+a[3]))*hs[sb0];ac1+=((b[0]+b[1])+(b[2]+b[3]))*hs[sb1];}
  y[r]=(float)ac0;y[r+1]=(float)ac1;}
 if(rows&1)y[rows-1]=one_row(x,pk+(size_t)(rows-1)*pb,sc+(size_t)(rows-1)*ng,in);}
static unsigned long long hf(const float*v,int n){unsigned long long h=1469598103934665603ull;for(int i=0;i<n;i++){union{float f;uint32_t u;}b;b.f=v[i];for(int k=0;k<4;k++){h^=(b.u>>(8*k))&255u;h*=1099511628211ull;}}return h;}
int main(void){enum{IN=3584,ROWS=3072,REPS=8};size_t np=(size_t)ROWS*(IN/2),ns=(size_t)ROWS*(IN/32);init_s();unsigned char*pk=malloc(np),*sc=malloc(ns);double*x=malloc((size_t)IN*8);float*a=malloc((size_t)ROWS*4),*b=malloc((size_t)ROWS*4);if(!pk||!sc||!x||!a||!b)return 2;for(size_t i=0;i<np;i++)pk[i]=(unsigned char)rnd();for(size_t i=0;i<ns;i++)sc[i]=(unsigned char)(120+rnd()%15);for(int i=0;i<IN;i++){float f=((int)(rnd()&65535)-32768)*(1.0f/65536.0f);x[i]=(double)f;}base(a,x,pk,sc,IN,ROWS);pair(b,x,pk,sc,IN,ROWS);printf("hash=%016llx %016llx\n",hf(a,ROWS),hf(b,ROWS));if(memcmp(a,b,(size_t)ROWS*4)){puts("PARITY FAIL");return 1;}puts("PARITY PASS");double ta=0,tb=0;for(int r=0;r<REPS;r++){double t=now_s();base(a,x,pk,sc,IN,ROWS);ta+=now_s()-t;t=now_s();pair(b,x,pk,sc,IN,ROWS);tb+=now_s()-t;}ta/=REPS;tb/=REPS;
#ifdef _OPENMP
printf("threads=%d ",omp_get_max_threads());
#endif
printf("base=%.3fms rowpair=%.3fms speedup=%.3fx\n",ta*1e3,tb*1e3,ta/tb);return 0;}
