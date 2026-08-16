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
static uint32_t rs=0x91e10da5u;
static uint32_t rnd(void){uint32_t x=rs;x^=x<<13;x^=x>>17;x^=x<<5;return rs=x;}
static double hs[256];
static void init_s(void){for(int i=0;i<255;i++)hs[i]=ldexp(1.0,i-127)*0.5;hs[255]=0.0;}

static float row32(const double*x,const unsigned char*pk,const unsigned char*sc,int in){
    const int ng=in/32;
    const __m128i mask=_mm_set1_epi8(0x0f);
    const __m128i lut=_mm_setr_epi8(0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12);
    double acc=0.0;
    for(int g=0;g<ng;g++){
        const unsigned char sb=sc[g]; if(sb==255)continue;
        const __m128i b=_mm_loadu_si128((const __m128i*)(pk+(size_t)g*16));
        const __m128i lo=_mm_shuffle_epi8(lut,_mm_and_si128(b,mask));
        const __m128i hi=_mm_shuffle_epi8(lut,_mm_and_si128(_mm_srli_epi16(b,4),mask));
        const __m128i q0=_mm_unpacklo_epi8(lo,hi),q1=_mm_unpackhi_epi8(lo,hi);
        const __m256i i0=_mm256_cvtepi8_epi32(q0),i1=_mm256_cvtepi8_epi32(_mm_srli_si128(q0,8));
        const __m256i i2=_mm256_cvtepi8_epi32(q1),i3=_mm256_cvtepi8_epi32(_mm_srli_si128(q1,8));
        const double*xg=x+(size_t)g*32; __m256d v0=_mm256_setzero_pd(),v1=_mm256_setzero_pd();
#define F(V,I,O) do{(V)=_mm256_fmadd_pd(_mm256_cvtepi32_pd((I)),_mm256_loadu_pd(xg+(O)),(V));}while(0)
        F(v0,_mm256_castsi256_si128(i0),0);F(v1,_mm256_extracti128_si256(i0,1),4);
        F(v0,_mm256_castsi256_si128(i1),8);F(v1,_mm256_extracti128_si256(i1,1),12);
        F(v0,_mm256_castsi256_si128(i2),16);F(v1,_mm256_extracti128_si256(i2,1),20);
        F(v0,_mm256_castsi256_si128(i3),24);F(v1,_mm256_extracti128_si256(i3,1),28);
#undef F
        double a[4];_mm256_storeu_pd(a,_mm256_add_pd(v0,v1));
        acc+=((a[0]+a[1])+(a[2]+a[3]))*hs[sb];
    }
    return (float)acc;
}

static void one(float*y,const double*x,const unsigned char*pk,const unsigned char*sc,int in,int rows){
    const int pb=in/2,sb=in/32;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(rows>64)
#endif
    for(int r=0;r<rows;r++)y[r]=row32(x,pk+(size_t)r*pb,sc+(size_t)r*sb,in);
}
static void pair(float*y1,float*y3,const double*x,const unsigned char*p1,const unsigned char*s1,const unsigned char*p3,const unsigned char*s3,int in,int rows){
    const int pb=in/2,sb=in/32;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(rows>64)
#endif
    for(int r=0;r<rows;r++){
        y1[r]=row32(x,p1+(size_t)r*pb,s1+(size_t)r*sb,in);
        y3[r]=row32(x,p3+(size_t)r*pb,s3+(size_t)r*sb,in);
    }
}
static unsigned long long hf(const float*v,int n){unsigned long long h=1469598103934665603ull;for(int i=0;i<n;i++){union{float f;uint32_t u;}b;b.f=v[i];for(int k=0;k<4;k++){h^=(b.u>>(8*k))&255;h*=1099511628211ull;}}return h;}
int main(void){enum{IN=3584,ROWS=3072,R=8};const size_t np=(size_t)ROWS*(IN/2),ns=(size_t)ROWS*(IN/32);init_s();
unsigned char*p1=malloc(np),*p3=malloc(np),*s1=malloc(ns),*s3=malloc(ns);double*x=malloc((size_t)IN*8);float*a1=malloc((size_t)ROWS*4),*a3=malloc((size_t)ROWS*4),*b1=malloc((size_t)ROWS*4),*b3=malloc((size_t)ROWS*4);if(!p1||!p3||!s1||!s3||!x||!a1||!a3||!b1||!b3)return 2;
for(size_t i=0;i<np;i++){p1[i]=(unsigned char)rnd();p3[i]=(unsigned char)rnd();}for(size_t i=0;i<ns;i++){s1[i]=(unsigned char)(120+rnd()%15);s3[i]=(unsigned char)(120+rnd()%15);}for(int i=0;i<IN;i++){float f=((int)(rnd()&65535)-32768)*(1.0f/65536.0f);x[i]=(double)f;}
one(a1,x,p1,s1,IN,ROWS);one(a3,x,p3,s3,IN,ROWS);pair(b1,b3,x,p1,s1,p3,s3,IN,ROWS);printf("hash1 %016llx %016llx hash3 %016llx %016llx\n",hf(a1,ROWS),hf(b1,ROWS),hf(a3,ROWS),hf(b3,ROWS));if(memcmp(a1,b1,(size_t)ROWS*4)||memcmp(a3,b3,(size_t)ROWS*4)){puts("PARITY FAIL");return 1;}puts("PARITY PASS");double ta=0,tb=0;for(int q=0;q<R;q++){double t=now_s();one(a1,x,p1,s1,IN,ROWS);one(a3,x,p3,s3,IN,ROWS);ta+=now_s()-t;t=now_s();pair(b1,b3,x,p1,s1,p3,s3,IN,ROWS);tb+=now_s()-t;}ta/=R;tb/=R;
#ifdef _OPENMP
printf("threads=%d ",omp_get_max_threads());
#endif
printf("separate=%.3fms pair=%.3fms speedup=%.3fx\n",ta*1e3,tb*1e3,ta/tb);return 0;}
