// Derived from https://github.com/rgl-epfl/cholespy/blob/main/src/cuda_driver.cpp

#include "cuda_driver.hpp"

#include <cstring>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>

#if defined(_WIN32)
#  include <windows.h>
#  define dlsym(ptr, name) GetProcAddress((HMODULE) ptr, name)
#else
#  include <dlfcn.h>
#endif

static void* handle = nullptr;

CUresult(*cuDeviceGet)(CUdevice*, int) = nullptr;
CUresult(*cuDeviceGetCount)(int*) = nullptr;
CUresult(*cuDevicePrimaryCtxRelease)(CUdevice) = nullptr;
CUresult(*cuDevicePrimaryCtxRetain)(CUcontext*, CUdevice) = nullptr;
CUresult(*cuGetErrorName)(CUresult, const char**) = nullptr;
CUresult(*cuGetErrorString)(CUresult, const char**) = nullptr;
CUresult(*cuInit)(unsigned int) = nullptr;
CUresult(*cuLaunchKernel)(CUfunction f, unsigned int, unsigned int,
    unsigned int, unsigned int, unsigned int,
    unsigned int, unsigned int, CUstream, void**,
    void**) = nullptr;
CUresult(*cuMemAlloc)(void**, size_t) = nullptr;
CUresult(*cuMemAllocManaged)(CUdeviceptr*, size_t, unsigned int) = nullptr;
CUresult(*cuMemFree)(void*) = nullptr;
CUresult(*cuMemcpyAsync)(void*, const void*, size_t, CUstream) = nullptr;
CUresult(*cuMemsetD32Async)(void*, unsigned int, size_t, CUstream) = nullptr;
CUresult(*cuMemsetD8Async)(void*, unsigned char, size_t, CUstream) = nullptr;
CUresult(*cuModuleGetFunction)(CUfunction*, CUmodule, const char*) = nullptr;
CUresult(*cuModuleLoadData)(CUmodule*, const void*) = nullptr;
CUresult(*cuModuleLoadDataEx)(CUmodule* module, const void* image, unsigned int  numOptions, CUjit_option* options, void** optionValues) = nullptr;
CUresult(*cuModuleUnload)(CUmodule) = nullptr;
CUresult(*cuCtxSynchronize)(void) = nullptr;
CUresult(*cuCtxPushCurrent)(CUcontext) = nullptr;
CUresult(*cuCtxPopCurrent)(CUcontext*) = nullptr;
CUresult(*cuStreamSynchronize)(CUstream) = nullptr;

CUresult (*cuEventCreate)(CUevent* phEvent, unsigned int  Flags) = nullptr;
CUresult (*cuEventDestroy)(CUevent hEvent) = nullptr;
CUresult (*cuEventElapsedTime)(float* pMilliseconds, CUevent hStart, CUevent hEnd) = nullptr;
CUresult (*cuEventQuery)(CUevent hEvent) = nullptr;
CUresult (*cuEventRecord)(CUevent hEvent, CUstream hStream) = nullptr;
CUresult (*cuEventRecordWithFlags)(CUevent hEvent, CUstream hStream, unsigned int  flags) = nullptr;
CUresult (*cuEventSynchronize)(CUevent hEvent) = nullptr;

CUdevice cu_device;
CUcontext cu_context;
CUmodule cu_module;

void cuda_check_impl(CUresult errval, const char* file, const int line) {
    if (errval != CUDA_SUCCESS && errval != CUDA_ERROR_DEINITIALIZED) {
        const char* name = nullptr, * msg = nullptr;
        cuGetErrorName(errval, &name);
        cuGetErrorString(errval, &msg);
        fprintf(stderr, "cuda_check(): API error = %04d (%s): \"%s\" in "
            "%s:%i.\n", (int)errval, name, msg, file, line);
    }
}

bool init_cuda() {

    if (handle)
        return true;

#if defined(_WIN32)
    handle = (void*)LoadLibraryA("nvcuda.dll");
#elif defined(__APPLE__)
    handle = nullptr;
#else
    handle = dlopen("libcuda.so", RTLD_LAZY);
#endif

    if (!handle)
        return false;

    const char* symbol = nullptr;

#define LOAD(name, ...)                                      \
        symbol = strlen(__VA_ARGS__ "") > 0                      \
            ? (#name "_" __VA_ARGS__) : #name;                   \
        name = decltype(name)(dlsym(handle, symbol));  \
        if (!name)                                               \
            break;                                               \
        symbol = nullptr

    do {
        LOAD(cuDevicePrimaryCtxRelease, "v2");
        LOAD(cuDevicePrimaryCtxRetain);
        LOAD(cuDeviceGet);
		LOAD(cuDeviceGetCount);
        LOAD(cuCtxSynchronize);
        LOAD(cuCtxPushCurrent, "v2");
        LOAD(cuCtxPopCurrent, "v2");
        LOAD(cuStreamSynchronize);
        LOAD(cuGetErrorName);
        LOAD(cuGetErrorString);
        LOAD(cuInit);
        LOAD(cuMemAlloc, "v2");
        LOAD(cuMemAllocManaged);
        LOAD(cuMemFree, "v2");

        // We dispatch to the legacy CUDA stream. 
        // That makes it easier to reliably exchange information with packages that
        // enqueue work on other CUDA streams 
        LOAD(cuLaunchKernel);
        LOAD(cuMemcpyAsync);
        LOAD(cuMemsetD8Async);
        LOAD(cuMemsetD32Async);

        LOAD(cuModuleGetFunction);
        LOAD(cuModuleLoadData);
        LOAD(cuModuleLoadDataEx);
        LOAD(cuModuleUnload);

        LOAD(cuEventCreate);
        LOAD(cuEventDestroy, "v2");
        LOAD(cuEventElapsedTime);
        LOAD(cuEventQuery);
        LOAD(cuEventRecord);
        LOAD(cuEventRecordWithFlags);
        LOAD(cuEventSynchronize);
    } while (false);

    if (symbol) {
        fprintf(stderr,
            "cuda_init(): could not find symbol \"%s\" -- disabling "
            "CUDA backend!", symbol);
        return false;
    }

    cuda_check(cuInit(0));
    cuda_check(cuDeviceGet(&cu_device, 0));
    cuda_check(cuDevicePrimaryCtxRetain(&cu_context, cu_device));

    return true;
}

void shutdown_cuda() {
    if (!handle)
        return;

    cuda_check(cuDevicePrimaryCtxRelease(cu_device));

#if defined(_WIN32)
    FreeLibrary((HMODULE)handle);
#elif !defined(__APPLE__)
    dlclose(handle);
#endif
}
