/* Exercises ignuda's cuBLAS shim: C = alpha*A*B + beta*C, column-major. */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
typedef void* cublasHandle_t;
extern int cublasCreate_v2(cublasHandle_t*);
extern int cublasDestroy_v2(cublasHandle_t);
extern int cublasSgemm_v2(cublasHandle_t,int,int,int,int,int,const float*,const float*,int,const float*,int,const float*,float*,int);
extern int cudaMalloc(void**,size_t);
extern int cudaMemcpy(void*,const void*,size_t,int);
extern int cudaFree(void*);
extern int cudaDeviceSynchronize(void);

int main(){
    const int M=64,N=48,K=32;
    float *A=malloc(sizeof(float)*M*K),*B=malloc(sizeof(float)*K*N),*C=malloc(sizeof(float)*M*N),*ref=malloc(sizeof(float)*M*N);
    for(int i=0;i<M*K;i++)A[i]=(i%7)-3;
    for(int i=0;i<K*N;i++)B[i]=(i%5)-2;
    float alpha=2.0f,beta=0.0f;
    // column-major reference: A is MxK (lda=M), B is KxN (ldb=K), C is MxN (ldc=M)
    for(int c=0;c<N;c++)for(int r=0;r<M;r++){float acc=0;for(int k=0;k<K;k++)acc+=A[(size_t)k*M+r]*B[(size_t)c*K+k];ref[(size_t)c*M+r]=alpha*acc;}
    void *dA,*dB,*dC; cudaMalloc(&dA,sizeof(float)*M*K);cudaMalloc(&dB,sizeof(float)*K*N);cudaMalloc(&dC,sizeof(float)*M*N);
    cudaMemcpy(dA,A,sizeof(float)*M*K,1);cudaMemcpy(dB,B,sizeof(float)*K*N,1);
    cublasHandle_t h; if(cublasCreate_v2(&h)){printf("cublasCreate failed\n");return 1;}
    int st=cublasSgemm_v2(h,0,0,M,N,K,&alpha,dA,M,dB,K,&beta,dC,M);
    if(st){printf("sgemm status %d\n",st);return 1;}
    cudaDeviceSynchronize();
    cudaMemcpy(C,dC,sizeof(float)*M*N,2);
    int bad=0; double maxerr=0;
    for(int i=0;i<M*N;i++){double e=fabs(C[i]-ref[i]);if(e>maxerr)maxerr=e;bad+=e>1e-3;}
    printf("sgemm: %s (max abs err %g)\n",bad?"FAIL":"ok",maxerr);
    cublasDestroy_v2(h);cudaFree(dA);cudaFree(dB);cudaFree(dC);
    return bad!=0;
}
