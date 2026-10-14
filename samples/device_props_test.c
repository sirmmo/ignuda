/* Checks cudaGetDeviceProperties(_v2): the CUDA 12 cudaDeviceProp layout is
 * reproduced here independently, so if ignuda's struct offsets were wrong the
 * sanity checks below (which read fields at very different depths) would fail.
 * Run with IGNUDA_CC=5.2 to pin the expected compute capability. */
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

typedef struct {
    char name[256];
    char uuid[16];
    char luid[8];
    unsigned int luidDeviceNodeMask;
    size_t totalGlobalMem;
    size_t sharedMemPerBlock;
    int regsPerBlock;
    int warpSize;
    size_t memPitch;
    int maxThreadsPerBlock;
    int maxThreadsDim[3];
    int maxGridSize[3];
    int clockRate;
    size_t totalConstMem;
    int major;
    int minor;
    size_t textureAlignment;
    size_t texturePitchAlignment;
    int deviceOverlap;
    int multiProcessorCount;
    int kernelExecTimeoutEnabled;
    int integrated;
    int canMapHostMemory;
    int computeMode;
    int maxTexture1D;
    int maxTexture1DMipmap;
    int maxTexture1DLinear;
    int maxTexture2D[2];
    int maxTexture2DMipmap[2];
    int maxTexture2DLinear[3];
    int maxTexture2DGather[2];
    int maxTexture3D[3];
    int maxTexture3DAlt[3];
    int maxTextureCubemap;
    int maxTexture1DLayered[2];
    int maxTexture2DLayered[3];
    int maxTextureCubemapLayered[2];
    int maxSurface1D;
    int maxSurface2D[2];
    int maxSurface3D[3];
    int maxSurface1DLayered[2];
    int maxSurface2DLayered[3];
    int maxSurfaceCubemap;
    int maxSurfaceCubemapLayered[2];
    size_t surfaceAlignment;
    int concurrentKernels;
    int ECCEnabled;
    int pciBusID;
    int pciDeviceID;
    int pciDomainID;
    int tccDriver;
    int asyncEngineCount;
    int unifiedAddressing;
    int memoryClockRate;
    int memoryBusWidth;
    int l2CacheSize;
    int persistingL2CacheMaxSize;
    int maxThreadsPerMultiProcessor;
    int streamPrioritiesSupported;
    int globalL1CacheSupported;
    int localL1CacheSupported;
    size_t sharedMemPerMultiprocessor;
    int regsPerMultiprocessor;
    int managedMemory;
    int isMultiGpuBoard;
    int multiGpuBoardGroupID;
    int hostNativeAtomicSupported;
    int singleToDoublePrecisionPerfRatio;
    int pageableMemoryAccess;
    int concurrentManagedAccess;
    int computePreemptionSupported;
    int canUseHostPointerForRegisteredMem;
    int cooperativeLaunch;
    int cooperativeMultiDeviceLaunch;
    size_t sharedMemPerBlockOptin;
    int pageableMemoryAccessUsesHostPageTables;
    int directManagedMemAccessFromHost;
    int maxBlocksPerMultiProcessor;
    int accessPolicyMaxWindowSize;
    size_t reservedSharedMemPerBlock;
    int hostRegisterSupported;
    int sparseCudaArraySupported;
    int hostRegisterReadOnlySupported;
    int timelineSemaphoreInteropSupported;
    int memoryPoolsSupported;
    int gpuDirectRDMASupported;
    unsigned int gpuDirectRDMAFlushWritesOptions;
    int gpuDirectRDMAWritesOrdering;
    unsigned int memoryPoolSupportedHandleTypes;
    int deferredMappingCudaArraySupported;
    int ipcEventSupported;
    int clusterLaunch;
    int unifiedFunctionPointers;
    int reserved[63];
} cudaDeviceProp;

extern int cudaGetDeviceProperties_v2(cudaDeviceProp *, int);
extern int cudaDeviceGetPCIBusId(char *, int, int);
extern int cudaSetDeviceFlags(unsigned int);

int main(void) {
    if (sizeof(cudaDeviceProp) != 1032) {
        printf("FAIL: sizeof(cudaDeviceProp)=%zu, expected 1032\n", sizeof(cudaDeviceProp));
        return 1;
    }
    cudaDeviceProp p;
    int st = cudaGetDeviceProperties_v2(&p, 0);
    if (st) { printf("cudaGetDeviceProperties_v2 -> %d\n", st); return 1; }

    printf("name=\"%s\" cc=%d.%d warp=%d SMs=%d\n", p.name, p.major, p.minor, p.warpSize,
           p.multiProcessorCount);
    printf("totalGlobalMem=%zu sharedMemPerBlock=%zu sharedMemPerBlockOptin=%zu integrated=%d\n",
           p.totalGlobalMem, p.sharedMemPerBlock, p.sharedMemPerBlockOptin, p.integrated);

    int fails = 0;
    if (strstr(p.name, "ignuda") == NULL) { printf("FAIL: name lacks 'ignuda'\n"); fails++; }
    if (p.warpSize != 32) { printf("FAIL: warpSize %d != 32\n", p.warpSize); fails++; }
    if (p.multiProcessorCount <= 0) { printf("FAIL: multiProcessorCount %d\n", p.multiProcessorCount); fails++; }
    if (p.totalGlobalMem == 0) { printf("FAIL: totalGlobalMem is 0\n"); fails++; }
    if (p.sharedMemPerBlock == 0) { printf("FAIL: sharedMemPerBlock is 0\n"); fails++; }
    if (p.sharedMemPerBlockOptin == 0) { printf("FAIL: sharedMemPerBlockOptin is 0\n"); fails++; }
    if (p.major <= 0 || p.major > 12) { printf("FAIL: implausible major %d\n", p.major); fails++; }

    const char *want = getenv("IGNUDA_CC");
    if (want) {
        int wj = 0, wn = 0;
        sscanf(want, "%d.%d", &wj, &wn);
        if (p.major != wj || p.minor != wn) {
            printf("FAIL: cc %d.%d != IGNUDA_CC %d.%d\n", p.major, p.minor, wj, wn);
            fails++;
        }
    }

    char bus[16] = {0};
    if (cudaDeviceGetPCIBusId(bus, sizeof(bus), 0) == 0) printf("pciBusId=%s\n", bus);
    cudaSetDeviceFlags(0);

    printf("%s\n", fails ? "SOME TESTS FAILED" : "ALL OK");
    return fails != 0;
}
