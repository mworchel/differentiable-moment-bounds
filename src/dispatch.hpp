#pragma once

#include <stdexcept>
#include <thread>
#include <vector>

#include "log.hpp"
#include "kernels.hpp"
#include "kernel_flags.hpp"
#include "moment_problem.hpp"
#include "cuda_module.hpp"
#include "cuda_driver.hpp"

///////////////////////////////////////////
// Utilities
///////////////////////////////////////////

template<typename Func>
void parallel_for(Func func, unsigned int size, unsigned int num_threads = 0, unsigned int min_size = 0)
{
    if (size < min_size) // No parallelization
    {
        for (unsigned int i = 0; i < size; ++i)
            func(i);

        return;
    }

    if (num_threads == 0)
        num_threads = std::thread::hardware_concurrency();
    unsigned int work_per_thread = (size + num_threads - 1) / num_threads;

    auto work_fn = [size, work_per_thread, &func](unsigned int thread_idx) {
        for (unsigned int i = thread_idx * work_per_thread; (i < (thread_idx + 1) * work_per_thread) && (i < size); ++i)
        {
            func(i);
        }
        };

    std::vector<std::thread> threads;
    for (unsigned int i = 0; i < num_threads; i++)
    {
        threads.emplace_back(work_fn, i);
    }

    for (std::thread& thread : threads)
    {
        thread.join();
    }
}

// Performs a templated call to a function `func<Flags>` based on the given runtime `flags`.
template<uint32_t Flags, uint32_t All, typename Func>
void run_with_flags(Func&& func, uint32_t flags)
{
    if constexpr (Flags > All)
        throw std::runtime_error("Internal error.");
    else
    {
        if (Flags == flags)
            func.template operator()<Flags>();
        else
            // Simply iterate all possible bits
            run_with_flags<Flags + 1, All, Func>(std::forward<Func>(func), flags);
    }
}


///////////////////////////////////////////
// Compute moment bounds (primal)
///////////////////////////////////////////

// The CPU implementation of k_compute_moment_bounds in kernels.cu
template<unsigned int N, typename Float, uint32_t Flags>
void compute_moment_bounds_cpu(unsigned int size, Float const* moments, Float const* etas, Float* bounds, Float* roots, Float* weights, dm::MomentBoundParams<Float> const& params)
{
    auto loop_body = [&](unsigned int i) {
        // Load the moments
        Float m[2 * N + 1];
        for (unsigned int j = 0; j < 2 * N + 1; j++)
        {
            m[j] = moments[j * size + i];
        }

        Float bound;
        dm::MomentBoundResult result;
        if constexpr (Flags & +ComputeMomentBoundsFlags::RetainRoots)
        {
            Float r[N + 1];

            // Optional: retain the weights as well (no-op if not requested)
            Float w[N + 1];
            Float* w_ptr = nullptr;
            if constexpr (Flags & +ComputeMomentBoundsFlags::RetainWeights)
                w_ptr = w;

            result = dm::compute_moment_bound<N, Float>(params, m, etas[i], &bound, nullptr, nullptr, r, w_ptr);

            for (unsigned int j = 0; j < N + 1; ++j)
            {
                roots[j * size + i] = r[j];
                if constexpr (Flags & +ComputeMomentBoundsFlags::RetainWeights)
                    weights[j * size + i] = w[j];
            }
        }
        else
            result = dm::compute_moment_bound<N, Float>(params, m, etas[i], &bound);

        if (result != dm::MomentBoundResult::Success) [[unlikely]]
        {
            bound = dm::encode_bits<Float>(+result);
        }

        bounds[i] = bound;
        };

    parallel_for(loop_body, size, /*num_threads=*/0u, /*min_size=*/1000u);
}

template<unsigned int N, typename Float, uint32_t Flags>
void compute_moment_bounds_gpu(int device_id, unsigned int size, Float const* moments, Float const* etas, Float* bounds, Float* roots, Float* weights, dm::MomentBoundParams<Float> const& params)
{
    if (!cu_context)
        throw std::runtime_error("CUDA context is not initialized.");

    // TODO: Handle non-default device (also affects the kernel cache)
    if (device_id != 0)
        throw std::invalid_argument(format_message("Currently only CUDA device 0 is supported, but received arrays on CUDA device %d", device_id));

    ScopedCudaContext context(cu_context);

    CUfunction kernel = get_kernel<Kernels::ComputeMomentBounds, Float, N, Flags>(device_id);

    if (!kernel)
        throw std::invalid_argument(format_message("Unable to find CUDA kernel for type %s and %d moments (n=%d)", typeid(Float).name(), 2 * N + 1, N));

    void* args[] = {
        &size,
        &moments,
        &etas,
        (void*)&params,
        &bounds,
        &roots,
        &weights
    };

    unsigned int num_threads = 128;
    unsigned int num_blocks = (size + num_threads - 1) / num_threads;
    cuda_check(cuLaunchKernel(kernel, num_blocks, 1, 1, num_threads, 1, 1, 0, 0, args, nullptr));
}


///////////////////////////////////////////
// Compute moment bounds (backward)
///////////////////////////////////////////

struct BackwardStorageDesc
{
    // TODO: Byte size/offsets would probably be more intuitive (but low priority as this is an internal detail anyway)

    unsigned int total_size{ 0 };   // Total storage size in floating point elements
    unsigned int offset_roots{ 0 }; // Offset to the roots, in floating point elements
};

// Returns a descriptor of the storage required for the backward pass (depending on the current set of active `flags`).
template<typename Float>
void get_backward_storage_desc(unsigned int num_elements, unsigned int num_moments, uint32_t flags, BackwardStorageDesc* desc)
{
    if (!desc)
        return;

    // TODO: Check that num_moments > 0

    unsigned int n = (num_moments - 1) / 2;

    // NOTE: Be mindful of alignment when extending this at some point.

    desc->total_size = 0;
    if (flags & +ComputeMomentBoundsFlags::RetainRoots)
    {
        desc->offset_roots = desc->total_size;

        // n+1 floats for the n+1 roots
        desc->total_size += num_elements * (n + 1);
    }
}

// The CPU implementation of k_compute_moment_bounds_backward in kernels.cu
template<unsigned int N, typename Float, uint32_t Flags>
void compute_moment_bounds_backward_cpu(unsigned int size, Float const* moments, Float const* etas, Float const* bounds, Float const* roots_in, dm::MomentBoundParams<Float> const& params, Float* dmoments, Float* detas, Float const* dbounds)
{
    constexpr unsigned int NumMoments = 2 * N + 1;
    constexpr bool UseExternalRoots = Flags & +ComputeMomentBoundsFlags::RetainRoots;
    // TODO: RetainWeights has currently no effect on the backward pass

    auto loop_body = [&](unsigned int i) {
        // Load the moments and eta from global memory and compute the bound
        Float m[NumMoments];
        for (unsigned int j = 0; j < NumMoments; ++j)
        {
            m[j] = moments[j * size + i];
        }

        Float eta = etas[i];

        // Recompute the bounds
        Float bound;
        Float L[dm::SymmetricMatrix<Float, N + 1>::data_size];
        Float coeffs[N + 1];
        Float weights[N + 1];
        Float roots[N + 1];
        if constexpr (UseExternalRoots)
        {
            for (unsigned int j = 0; j < N + 1; ++j)
                roots[j] = roots_in[j * size + i];
        }
        dm::MomentBoundResult result = dm::compute_moment_bound<N, Float, UseExternalRoots>(params, m, eta, &bound,
            L, coeffs, UseExternalRoots ? nullptr : roots, weights,
            roots);

        Float dm[NumMoments] = { 0 };
        Float deta(0);
        Float dbound = dbounds[i];
        dm::compute_moment_bound_backward<N, Float>(params, m, eta, bound,
            L, coeffs, roots, weights,
            dm, &deta, dbound);

        for (unsigned int j = 0; j < NumMoments; ++j)
        {
            dmoments[j * size + i] = result == dm::MomentBoundResult::Success ? dm[j] : dm::encode_bits<Float>(+result);
        }
        detas[i] = result == dm::MomentBoundResult::Success ? deta : dm::encode_bits<Float>(+result);
    };

    parallel_for(loop_body, size, /*num_threads=*/0u, /*min_size=*/1000u);
}

template<unsigned int N, typename Float, uint32_t Flags>
void compute_moment_bounds_backward_gpu(int device_id, unsigned int size, Float const* moments, Float const* etas, Float const* bounds, Float const* roots, dm::MomentBoundParams<Float> const& params, Float* dmoments, Float* detas, Float const* dbounds)
{
    if (!cu_context)
        throw std::runtime_error("CUDA context is not initialized.");

    // TODO: Handle non-default device (also affects the kernel cache)
    if (device_id != 0)
        throw std::invalid_argument(format_message("Currently only CUDA device 0 is supported, but received arrays on CUDA device %d", device_id));

    ScopedCudaContext context(cu_context);

    CUfunction kernel = get_kernel<Kernels::ComputeMomentBoundsBackward, Float, N, Flags>(device_id);

    if (!kernel)
        throw std::invalid_argument(format_message("Unable to find CUDA kernel for type %s and %d moments (n=%d)", typeid(Float).name(), 2 * N + 1, N));

    void* args[] = {
        &size,
        &moments,
        &etas,
        (void*)&params,
        &bounds,
        &roots,
        &dmoments,
        &detas,
        &dbounds
    };

    unsigned int num_threads = 128;
    unsigned int num_blocks = (size + num_threads - 1) / num_threads;
    cuda_check(cuLaunchKernel(kernel, num_blocks, 1, 1, num_threads, 1, 1, 0, 0, args, nullptr));
}


///////////////////////////////////////////
// Detect errors in the computed bounds
///////////////////////////////////////////

// The CPU implementation of k_detect_bound_errors in kernels.cu
template<typename Float>
void detect_bound_errors_cpu(unsigned int size, Float const* bounds)
{
    for (unsigned int i = 0; i < size; i++)
    {
        Float bound = bounds[i];

        // Bound is a finite value, so probably ok
        if (std::isfinite(bound))
            continue;

        // Infinity is an error, but this is not tied to a specific result code.
        if (std::isinf(bound))
            throw std::runtime_error(format_message("Detected error for element %d but the error is unknown.", i));

        // The bound is NaN, so it should contain the encoded result.
        uint32_t result = dm::decode_bits(bound);

        if (result > +dm::MomentBoundResult::Success &&
            result < +dm::MomentBoundResult::Count)
        {
            // The NaN contains a result code, so trigger an error
            throw std::runtime_error(format_message("Detected error for element %d: %s.", i, dm::get_result_string(static_cast<dm::MomentBoundResult>(result))));
        }
        else
        {
            // The NaN doesn't seem to contain any information.
            // FIXME: This indicates that not all errors are considered when computing the bound.
            throw std::runtime_error(format_message("Detected error for element %d but the error is unknown.", i));
        }
    }
}

template<typename Float>
void detect_bound_errors_gpu(int device_id, unsigned int size, Float* bounds)
{
    if (!cu_context)
        throw std::runtime_error("CUDA context is not initialized.");

    // TODO: Handle non-default device (also affects the kernel cache)
    if (device_id != 0)
        throw std::invalid_argument(format_message("Currently only CUDA device 0 is supported, but received arrays on CUDA device %d", device_id));

    ScopedCudaContext context(cu_context);

    CUfunction kernel = get_kernel<Kernels::DetectBoundErrors, Float>(device_id);

    if (!kernel)
        throw std::invalid_argument(format_message("Unable to find CUDA kernel for type %s", typeid(Float).name()));

	static_assert(sizeof(dm::MomentBoundResult) == sizeof(int), "MomentBoundResult must fit in an int");

    // Inefficient to alloc/dealloc, but this kernel is not meant to be efficient :)
    int* error_info_;
    cuda_check(cuMemAlloc((void**)&error_info_, 2 * sizeof(int)));
    // We intentially only set one value of error_info (the first), because that's the error. 
    // If this is overwritten, the second entry (thread index) will be set anyway.
    cuda_check(cuMemsetD32Async(error_info_, +dm::MomentBoundResult::Success, 1, 0));

    void* args[] = {
        &size,
        &bounds,
        &error_info_
    };

    unsigned int num_threads = 128;
    unsigned int num_blocks = (size + num_threads - 1) / num_threads;
    cuda_check(cuLaunchKernel(kernel, num_blocks, 1, 1, num_threads, 1, 1, 0, 0, args, nullptr));

    int error_info[2];
    cuda_check(cuMemcpyAsync(error_info, error_info_, 2 * sizeof(int), 0));
    cuda_check(cuCtxSynchronize());

    cuda_check(cuMemFree(error_info_));

    // Handle the reported error
    if (error_info[0] != +dm::MomentBoundResult::Success)
    {
        dm::MomentBoundResult result = static_cast<dm::MomentBoundResult>(error_info[0]);
        unsigned int index = error_info[1];
        if (result == dm::MomentBoundResult::Unknown)
            throw std::runtime_error(format_message("Detected error for element %d but the error is unknown.", index));
        else
            throw std::runtime_error(format_message("Detected error for element %d: %s.", index, dm::get_result_string(result)));
    }
}

///////////////////////////////////////////
// Compute singularities
///////////////////////////////////////////

// The CPU implementation of k_compute_singularities in kernels.cu
template<unsigned int N, typename Float>
void compute_singularities_cpu(unsigned int size, dm::MomentBoundParams<Float> const& params, Float const* moments, Float* singularities)
{
    // TODO: share code with the GPU kernel
  
    constexpr unsigned int NumMoments = 2 * N + 1;

    auto loop_body = [&](unsigned int i) {
        // Load the moments and eta from global memory and compute the bound
        Float m[NumMoments];
        for (unsigned int j = 0; j < NumMoments; ++j)
        {
            m[j] = moments[j * size + i];
        }
        Float eta = 0;

        // Use `compute_moment_bound` to get the Cholesky decomposition of the Hankel matrix.
        // This function doesn't need to be efficient anyway.
        Float bound(0);
        Float L_data[dm::LowerTriangularMatrix<Float, N + 1>::data_size];
        dm::MomentBoundResult result = dm::compute_moment_bound<N, Float>(params, m, eta, &bound, L_data, nullptr, nullptr, nullptr);

        dm::LowerTriangularMatrix<Float const, N + 1> L{ .data = L_data };

        Float Pn_coeffs[N + 1];
        dm::compute_Pn_coefficients<N, Float>(L, Pn_coeffs);

        Float Pn_roots[N];
        dm::find_real_polynomial_roots<N, Float>(Pn_coeffs, Pn_roots, params.newton_tolerance, params.newton_max_iterations);

        if (result != dm::MomentBoundResult::Success) [[unlikely]]
        {
            for (unsigned int j = 0; j < N; ++j)
                Pn_roots[j] = dm::encode_bits<Float>(+result);
        }

        for (unsigned int j = 0; j < N; ++j)
            singularities[j * size + i] = Pn_roots[j];
    };

    parallel_for(loop_body, size, /*num_threads=*/0u, /*min_size=*/1000u);
}

template<unsigned int N, typename Float>
void compute_singularities_gpu(int device_id, unsigned int size, dm::MomentBoundParams<Float> const& params, Float const* moments, Float* singularities)
{
    if (!cu_context)
        throw std::runtime_error("CUDA context is not initialized.");

    // TODO: Handle non-default device (also affects the kernel cache)
    if (device_id != 0)
        throw std::invalid_argument(format_message("Currently only CUDA device 0 is supported, but received arrays on CUDA device %d", device_id));

    ScopedCudaContext context(cu_context);

    CUfunction kernel = get_kernel<Kernels::ComputeSingularities, Float, N>(device_id);

    if (!kernel)
        throw std::invalid_argument(format_message("Unable to find CUDA kernel for type %s and %d moments (n=%d)", typeid(Float).name(), 2 * N + 1, N));

    void* args[] = {
        &size,
        &moments,
        (void*)&params,
        &singularities
    };

    unsigned int num_threads = 128;
    unsigned int num_blocks = (size + num_threads - 1) / num_threads;
    cuda_check(cuLaunchKernel(kernel, num_blocks, 1, 1, num_threads, 1, 1, 0, 0, args, nullptr));
}