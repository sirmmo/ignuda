/*
 * Minimal clean-room declaration of the CUDA Driver API symbols used by the
 * ignuda samples. This is NOT NVIDIA's header: it is a hand-written subset of
 * the publicly documented driver API (https://docs.nvidia.com/cuda/cuda-driver-api/)
 * sufficient to compile `driver_test.c` without a CUDA toolkit installed.
 *
 * It exists so the sample can be built in environments (such as CI) that have
 * no NVIDIA software at all. If a real CUDA toolkit is on the include path,
 * that header is used instead and this one is never seen.
 */
#ifndef IGNUDA_SAMPLES_CUDA_H
#define IGNUDA_SAMPLES_CUDA_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef int CUresult;
#define CUDA_SUCCESS 0

typedef int CUdevice;
typedef unsigned long long CUdeviceptr; /* 64-bit host: CUdeviceptr_v2 */

typedef struct CUctx_st  *CUcontext;
typedef struct CUmod_st  *CUmodule;
typedef struct CUfunc_st *CUfunction;
typedef struct CUstream_st *CUstream;

/* The driver ABI renames several entry points to their _v2 form; the public
 * header does this through macros, and the loader exports only the _v2 names. */
#define cuCtxCreate   cuCtxCreate_v2
#define cuCtxDestroy  cuCtxDestroy_v2
#define cuMemAlloc    cuMemAlloc_v2
#define cuMemFree     cuMemFree_v2
#define cuMemcpyHtoD  cuMemcpyHtoD_v2
#define cuMemcpyDtoH  cuMemcpyDtoH_v2

CUresult cuInit(unsigned int flags);
CUresult cuDeviceGet(CUdevice *device, int ordinal);
CUresult cuDeviceGetName(char *name, int len, CUdevice dev);
CUresult cuCtxCreate_v2(CUcontext *pctx, unsigned int flags, CUdevice dev);
CUresult cuCtxDestroy_v2(CUcontext ctx);
CUresult cuCtxSynchronize(void);
CUresult cuModuleLoadData(CUmodule *module, const void *image);
CUresult cuModuleGetFunction(CUfunction *func, CUmodule mod, const char *name);
CUresult cuModuleUnload(CUmodule hmod);
CUresult cuMemAlloc_v2(CUdeviceptr *dptr, size_t bytesize);
CUresult cuMemFree_v2(CUdeviceptr dptr);
CUresult cuMemcpyHtoD_v2(CUdeviceptr dst, const void *src, size_t n);
CUresult cuMemcpyDtoH_v2(void *dst, CUdeviceptr src, size_t n);
CUresult cuGetErrorName(CUresult error, const char **str);
CUresult cuLaunchKernel(CUfunction f,
                        unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
                        unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
                        unsigned int sharedMemBytes, CUstream hStream,
                        void **kernelParams, void **extra);

#ifdef __cplusplus
}
#endif

#endif /* IGNUDA_SAMPLES_CUDA_H */
