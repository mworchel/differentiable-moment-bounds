#include "cuda_module.hpp"

#include <iterator>

#include "generated/kernels.ptx.h"

void load_module()
{
    cuda_check(cuCtxPushCurrent(cu_context));
    cuda_check(cuModuleLoadData(&cu_module, (void*)ptx_bytes));
    //CUjit_option options[] = {
    //    CU_JIT_WALL_TIME,
    //};
    //float time(0);
    //void* option_values[] = {
    //    &time
    //};

    //cuda_check(cuModuleLoadDataEx(&cu_module, (void*)ptx_bytes, std::size(options), options, option_values));
}

void unload_module()
{
    cuda_check(cuCtxPushCurrent(cu_context));
    cuda_check(cuModuleUnload(cu_module));
}