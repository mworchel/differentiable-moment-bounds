#include <iostream>
#include <numeric>

#include "cuda_driver.hpp"
#include "dispatch.hpp"
#include "kernel_flags.hpp"
#include "moment_problem.hpp"

int main()
{
    // This is not a real 'test', but just some code to play with the dispatch routines.

	init_cuda();
	load_module();

    int const device_id = 0;

    ScopedCudaContext guard(cu_context);

    constexpr unsigned int size = 10;
    constexpr unsigned int N = 3;
	constexpr unsigned int NumMoments = 2 * N + 1;

    float* moments = nullptr;
    cuda_check(cuMemAllocManaged((void**)&moments, sizeof(float) * NumMoments * size, CU_MEM_ATTACH_HOST));
    std::iota(moments, moments + NumMoments * size, 0.f);

    float* etas = nullptr;
    cuda_check(cuMemAllocManaged((void**)&etas, sizeof(float) * size, CU_MEM_ATTACH_HOST));

    float* bounds = nullptr;
    cuda_check(cuMemAllocManaged((void**)&bounds, sizeof(float) * size, CU_MEM_ATTACH_HOST));
    
    dm::MomentBoundParams<float> params{ .bias = 0, .overestimation_weight = 0.f,.newton_tolerance = 1e-6 };

    compute_moment_bounds_gpu<N, float, +ComputeMomentBoundsFlags::None>(device_id, size, moments, etas, bounds, nullptr, nullptr, params);

    cuda_check(cuStreamSynchronize(0));

    try
    {
        detect_bound_errors_gpu<float>(device_id, size, bounds);
    }
    catch (std::exception const& e)
    {
        std::cout << e.what() << std::endl;
    }

    for (size_t i = 0; i < size; i++)
    {
        std::cout << i << ": " << bounds[i] << std::endl;
    }

    unload_module();
    shutdown_cuda();
}