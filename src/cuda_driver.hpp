// Derived from https://github.com/rgl-epfl/cholespy/blob/main/src/cuda_driver.h (3-clause BSD License)

#pragma once

#include <cstdio>
using size_t = std::size_t;

#define CUDA_ERROR_DEINITIALIZED 4
#define CUDA_SUCCESS 0

using CUcontext = struct CUctx_st*;
using CUmodule = struct CUmod_st*;
using CUstream = struct CUstream_st*;
using CUfunction = struct CUfunc_st*;
using CUresult = int;
using CUdevice = int;
using CUdeviceptr = void*;

enum CUjit_option
{
    CU_JIT_MAX_REGISTERS = 0,
    CU_JIT_THREADS_PER_BLOCK = 1,
    CU_JIT_WALL_TIME = 2,
    CU_JIT_INFO_LOG_BUFFER = 3,
    CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES = 4,
    CU_JIT_ERROR_LOG_BUFFER = 5,
    CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES = 6,
    CU_JIT_OPTIMIZATION_LEVEL = 7,
    CU_JIT_TARGET_FROM_CUCONTEXT = 8,
    CU_JIT_TARGET = 9,
    CU_JIT_FALLBACK_STRATEGY = 10,
    CU_JIT_GENERATE_DEBUG_INFO = 11,
    CU_JIT_LOG_VERBOSE = 12,
    CU_JIT_GENERATE_LINE_INFO = 13,
    CU_JIT_CACHE_MODE = 14,
    CU_JIT_NEW_SM3X_OPT = 15,
    CU_JIT_FAST_COMPILE = 16,
    CU_JIT_GLOBAL_SYMBOL_NAMES = 17,
    CU_JIT_GLOBAL_SYMBOL_ADDRESSES = 18,
    CU_JIT_GLOBAL_SYMBOL_COUNT = 19,
    CU_JIT_LTO = 20,
    CU_JIT_FTZ = 21,
    CU_JIT_PREC_DIV = 22,
    CU_JIT_PREC_SQRT = 23,
    CU_JIT_FMA = 24,
    CU_JIT_REFERENCED_KERNEL_NAMES = 25,
    CU_JIT_REFERENCED_KERNEL_COUNT = 26,
    CU_JIT_REFERENCED_VARIABLE_NAMES = 27,
    CU_JIT_REFERENCED_VARIABLE_COUNT = 28,
    CU_JIT_OPTIMIZE_UNUSED_DEVICE_VARIABLES = 29,
    CU_JIT_POSITION_INDEPENDENT_CODE = 30,
    CU_JIT_MIN_CTA_PER_SM = 31,
    CU_JIT_MAX_THREADS_PER_BLOCK = 32,
    CU_JIT_OVERRIDE_DIRECTIVE_VALUES = 33,
    CU_JIT_NUM_OPTIONS
};

extern CUresult(*cuDeviceGet)(CUdevice*, int);
extern CUresult(*cuDevicePrimaryCtxRelease)(CUdevice);
extern CUresult(*cuDevicePrimaryCtxRetain)(CUcontext*, CUdevice);
extern CUresult(*cuGetErrorName)(CUresult, const char**);
extern CUresult(*cuGetErrorString)(CUresult, const char**);
extern CUresult(*cuInit)(unsigned int);
extern CUresult(*cuLaunchKernel)(CUfunction f,
    unsigned int  gridDimX, unsigned int  gridDimY, unsigned int  gridDimZ,
    unsigned int  blockDimX, unsigned int  blockDimY, unsigned int  blockDimZ,
    unsigned int  sharedMemBytes, CUstream hStream, void** kernelParams, void** extra);
extern CUresult(*cuMemAlloc)(void**, size_t);
extern CUresult(*cuMemAllocManaged)(CUdeviceptr* dptr, size_t bytesize, unsigned int flags);
extern CUresult(*cuMemFree)(void*);
extern CUresult(*cuMemcpyAsync)(void*, const void*, size_t, CUstream);
extern CUresult(*cuMemsetD32Async)(void*, unsigned int, size_t, CUstream);
extern CUresult(*cuMemsetD8Async)(void*, unsigned char, size_t, CUstream);
extern CUresult(*cuModuleGetFunction)(CUfunction*, CUmodule, const char*);
extern CUresult(*cuModuleLoadData)(CUmodule*, const void*);
extern CUresult(*cuModuleLoadDataEx)(CUmodule* module, const void* image, unsigned int  numOptions, CUjit_option* options, void** optionValues);
extern CUresult(*cuModuleUnload)(CUmodule);
extern CUresult(*cuCtxSynchronize)(void);
extern CUresult(*cuCtxPushCurrent)(CUcontext);
extern CUresult(*cuCtxPopCurrent)(CUcontext*);
extern CUresult(*cuStreamSynchronize)( CUstream hStream );

using CUevent = struct CUevent_st*;

enum CUevent_flags
{
    CU_EVENT_DEFAULT = 0x0,
    CU_EVENT_BLOCKING_SYNC = 0x1,
    CU_EVENT_DISABLE_TIMING = 0x2,
    CU_EVENT_INTERPROCESS = 0x4
};

extern CUresult(*cuEventCreate)(CUevent* phEvent, unsigned int  Flags);
extern CUresult(*cuEventDestroy)(CUevent hEvent);
extern CUresult(*cuEventElapsedTime)(float* pMilliseconds, CUevent hStart, CUevent hEnd);
extern CUresult(*cuEventQuery)(CUevent hEvent);
extern CUresult(*cuEventRecord)(CUevent hEvent, CUstream hStream);
extern CUresult(*cuEventRecordWithFlags)(CUevent hEvent, CUstream hStream, unsigned int  flags);
extern CUresult(*cuEventSynchronize)(CUevent hEvent);

enum CUmemAttach_flags
{
    CU_MEM_ATTACH_GLOBAL = 0x1,
    CU_MEM_ATTACH_HOST   = 0x2,
    CU_MEM_ATTACH_SINGLE = 0x4
};

// Assert that a CUDA operation is correctly issued
#define cuda_check(err) cuda_check_impl(err, __FILE__, __LINE__)

extern CUdevice cu_device;
extern CUcontext cu_context;

extern bool init_cuda();
extern void shutdown_cuda();
extern void cuda_check_impl(CUresult errval, const char* file, const int line);

struct ScopedCudaContext {
    ScopedCudaContext(CUcontext ctx) {
        cuda_check(cuCtxPushCurrent(ctx));
    }
    ~ScopedCudaContext() {
        cuda_check(cuCtxPopCurrent(nullptr));
    }
};