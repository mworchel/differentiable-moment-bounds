#include <cstdint>
#include <cstdio>

#include <cuda_runtime.h>

// Syntax highlighting
#ifdef __INTELLISENSE__
#define __CUDACC__
#endif // __INTELLISENSE__

#include <device_launch_parameters.h>

#ifdef __INTELLISENSE__
#undef __CUDACC__
#endif // __INTELLISENSE__

#include "linalg.hpp"
#include "matrix.hpp"
#include "moment_problem.hpp"

#include "kernel_flags.hpp"

///////////////////
// Primal pass for moment bound computation
///////////////////

// Kernel prototype, instantiated below
template<typename Float, unsigned int N, uint32_t Flags>
__device__ void k_compute_moment_bounds(unsigned int size, Float const* __restrict__ moments, Float const* __restrict__ etas, dm::MomentBoundParams<Float> params, Float* __restrict__ bounds, Float* __restrict__ roots, Float* __restrict__ weights)
{
	constexpr unsigned int NumMoments = 2 * N + 1;

	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= size)
		return;

	// Load the moments and eta from global memory and compute the bound
	Float m[NumMoments];
	for (unsigned int j = 0; j < NumMoments; ++j)
	{
		m[j] = moments[j * size + i];
	}

	Float eta = etas[i];

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

		result = dm::compute_moment_bound<N, Float>(params, m, eta, &bound, nullptr, nullptr, r, w_ptr);

		for (unsigned int j = 0; j < N + 1; ++j)
		{
			roots[j * size + i] = r[j];
			if constexpr (Flags & +ComputeMomentBoundsFlags::RetainWeights)
				weights[j * size + i] = w[j];
		}
	}
	else
		result = dm::compute_moment_bound<N, Float>(params, m, eta, &bound);

	if (result != dm::MomentBoundResult::Success) [[unlikely]]
	{
		// If an error occurred, encode the result code inside a NaN.
		// To the outside, this simply looks like an undefined result,
		// but we can decode the error later in `k_detect_bound_errors`
		// to supply more information.
		bound = dm::encode_bits<Float>(+result);
	}

	bounds[i] = bound;
}

// Instantiate the `k_compute_moment_bounds` kernel for different template parameters, assigning each a unique name.
// For example, `k_compute_moment_bounds_double_1_0` for Float=double, N=1 and Flags=0 (= ComputeMomentBoundsFlags::None).

#define IMPLEMENT_COMPUTE_MOMENT_BOUNDS_KERNEL(Float, N, Flags) \
extern "C" \
__global__ void \
/*__launch_bounds__(128) \*/ \
k_compute_moment_bounds_##Float##_##N##_##Flags (unsigned int size, Float const* __restrict__ moments, Float const* __restrict__ etas, dm::MomentBoundParams<Float> params, Float* __restrict__ bounds, Float* __restrict__ roots, Float* __restrict__ weights) \
{ \
	k_compute_moment_bounds<Float, N, Flags>(size, moments, etas, params, bounds, roots, weights); \
}

#define IMPLEMENT_COMPUTE_MOMENT_BOUNDS_KERNEL_FLAGS(Float, N) \
IMPLEMENT_COMPUTE_MOMENT_BOUNDS_KERNEL(Float, N, 0 /*= ComputeMomentBoundsFlags::None*/); \
IMPLEMENT_COMPUTE_MOMENT_BOUNDS_KERNEL(Float, N, 1 /*= ComputeMomentBoundsFlags::RetainRoots*/); \
IMPLEMENT_COMPUTE_MOMENT_BOUNDS_KERNEL(Float, N, 3 /*= ComputeMomentBoundsFlags::RetainRoots & ComputeMomentBoundsFlags::RetainWeights*/);

#define IMPLEMENT_COMPUTE_MOMENT_BOUNDS_KERNEL_N(Float) \
IMPLEMENT_COMPUTE_MOMENT_BOUNDS_KERNEL_FLAGS(Float, 1); \
IMPLEMENT_COMPUTE_MOMENT_BOUNDS_KERNEL_FLAGS(Float, 2); \
IMPLEMENT_COMPUTE_MOMENT_BOUNDS_KERNEL_FLAGS(Float, 3); \
IMPLEMENT_COMPUTE_MOMENT_BOUNDS_KERNEL_FLAGS(Float, 4); \
IMPLEMENT_COMPUTE_MOMENT_BOUNDS_KERNEL_FLAGS(Float, 5); \
IMPLEMENT_COMPUTE_MOMENT_BOUNDS_KERNEL_FLAGS(Float, 6); 
// The maximum supported N is defined by `MaxN` in `dispatch.hpp`

IMPLEMENT_COMPUTE_MOMENT_BOUNDS_KERNEL_N(float);
IMPLEMENT_COMPUTE_MOMENT_BOUNDS_KERNEL_N(double);


///////////////////
// Backward pass for moment bound computation
///////////////////

// Kernel prototype, instantiated below
template<typename Float, unsigned int N, uint32_t Flags>
__device__ void k_compute_moment_bounds_backward(unsigned int size, Float const* __restrict__ moments, Float const* __restrict__ etas, dm::MomentBoundParams<Float> params, Float const* __restrict__ bounds, Float const* __restrict__ roots_in,
								 Float*__restrict__ dmoments, Float* __restrict__ detas, Float const* __restrict__ dbounds)
{
	constexpr unsigned int NumMoments = 2 * N + 1;
	constexpr bool UseExternalRoots = Flags & +ComputeMomentBoundsFlags::RetainRoots;
	// TODO: RetainWeights has currently no effect on the backward pass

	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= size)
		return;

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
	Float roots[N + 1];
	Float weights[N + 1];
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
}

#define IMPLEMENT_COMPUTE_MOMENT_BOUNDS_BACKWARD_KERNEL(Float, N, Flags) \
extern "C" \
__global__ void k_compute_moment_bounds_backward_##Float##_##N##_##Flags (unsigned int size, Float const* __restrict__ moments, Float const* __restrict__ etas, dm::MomentBoundParams<Float> params, Float const* __restrict__ bounds, Float const* __restrict__ roots_in, \
																Float*__restrict__ dmoments, Float* __restrict__ detas, Float const* __restrict__ dbounds) \
{ \
	k_compute_moment_bounds_backward<Float, N, Flags>(size, moments, etas, params, bounds, roots_in, dmoments, detas, dbounds); \
}

#define IMPLEMENT_COMPUTE_MOMENT_BOUNDS_BACKWARD_KERNEL_FLAGS(Float, N) \
IMPLEMENT_COMPUTE_MOMENT_BOUNDS_BACKWARD_KERNEL(Float, N, 0 /*= ComputeMomentBoundsFlags::None*/); \
IMPLEMENT_COMPUTE_MOMENT_BOUNDS_BACKWARD_KERNEL(Float, N, 1 /*= ComputeMomentBoundsFlags::RetainRoots*/); \
IMPLEMENT_COMPUTE_MOMENT_BOUNDS_BACKWARD_KERNEL(Float, N, 3 /*= ComputeMomentBoundsFlags::RetainRoots & ComputeMomentBoundsFlags::RetainWeights*/);


#define IMPLEMENT_COMPUTE_MOMENT_BOUNDS_BACKWARD_KERNEL_N(Float) \
IMPLEMENT_COMPUTE_MOMENT_BOUNDS_BACKWARD_KERNEL_FLAGS(Float, 1); \
IMPLEMENT_COMPUTE_MOMENT_BOUNDS_BACKWARD_KERNEL_FLAGS(Float, 2); \
IMPLEMENT_COMPUTE_MOMENT_BOUNDS_BACKWARD_KERNEL_FLAGS(Float, 3); \
IMPLEMENT_COMPUTE_MOMENT_BOUNDS_BACKWARD_KERNEL_FLAGS(Float, 4); \
IMPLEMENT_COMPUTE_MOMENT_BOUNDS_BACKWARD_KERNEL_FLAGS(Float, 5); \
IMPLEMENT_COMPUTE_MOMENT_BOUNDS_BACKWARD_KERNEL_FLAGS(Float, 6);
// The maximum supported N is defined by `MaxN` in `dispatch.hpp`

IMPLEMENT_COMPUTE_MOMENT_BOUNDS_BACKWARD_KERNEL_N(float);
IMPLEMENT_COMPUTE_MOMENT_BOUNDS_BACKWARD_KERNEL_N(double);


///////////////////
// Error detection
///////////////////

__device__ bool try_write_error(unsigned int idx, uint32_t result, int* error_info)
{
	constexpr int no_error_flag = static_cast<int>(dm::MomentBoundResult::Success);

	int error_flag_previous = atomicCAS(&error_info[0], no_error_flag /*==no error set*/, static_cast<int>(result));
	if (error_flag_previous != no_error_flag) [[likely]]
		return false;
	
	// This thread has set the error flag, so note it's index
	error_info[1] = idx;

	return true;
}

// Kernel prototype, instantiated below
template<typename Float>
__device__ void k_detect_bound_errors(unsigned int size, Float const* bounds, int* error_info)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= size)
		return;

	Float bound = bounds[i];

	// Bound is a finite value, so probably ok
	if (isfinite(bound))
		return;

	// Infinity is an error, but this is not tied to a specific result code.
	if (isinf(bound))
	{
		try_write_error(i, +dm::MomentBoundResult::Unknown, error_info);
		return;
	}

	// The bound is NaN, so it should contain the encoded result.
	uint32_t result = dm::decode_bits(bound);

	if (result > +dm::MomentBoundResult::Success && 
		result < +dm::MomentBoundResult::Count)
	{
		// The NaN contains a result code, so write it to the error info
		try_write_error(i, result, error_info);
	}
	else
	{
		// The NaN doesn't seem to contain any information.
		// FIXME: This indicates that not all errors are considered when computing the bound.
		try_write_error(i, +dm::MomentBoundResult::Unknown, error_info);
	}
}

#define IMPLEMENT_DETECT_BOUND_ERRORS_KERNEL(Float) \
extern "C" \
__global__ void k_detect_bound_errors_##Float (unsigned int size, Float const* __restrict__ bounds, int* __restrict__ error_info) \
{ \
	k_detect_bound_errors<Float>(size, bounds, error_info); \
}

IMPLEMENT_DETECT_BOUND_ERRORS_KERNEL(float)
IMPLEMENT_DETECT_BOUND_ERRORS_KERNEL(double)


template<typename Float, unsigned int N>
__device__ void k_compute_singularities(unsigned int size, Float const* __restrict__ moments, dm::MomentBoundParams<Float> params, Float* __restrict__ singularities)
{
	constexpr unsigned int NumMoments = 2 * N + 1;

	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= size)
		return;

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
}

#define IMPLEMENT_COMPUTE_SINGULARITIES_KERNEL(Float, N) \
extern "C" \
__global__ void k_compute_singularities_##Float##_##N (unsigned int size, Float const* __restrict__ moments, dm::MomentBoundParams<Float> params, Float* __restrict__ singularities) \
{ \
	k_compute_singularities<Float, N>(size, moments, params, singularities); \
}

#define IMPLEMENT_COMPUTE_SINGULARITIES_KERNEL_N(Float) \
IMPLEMENT_COMPUTE_SINGULARITIES_KERNEL(Float, 1); \
IMPLEMENT_COMPUTE_SINGULARITIES_KERNEL(Float, 2); \
IMPLEMENT_COMPUTE_SINGULARITIES_KERNEL(Float, 3); \
IMPLEMENT_COMPUTE_SINGULARITIES_KERNEL(Float, 4); \
IMPLEMENT_COMPUTE_SINGULARITIES_KERNEL(Float, 5); \
IMPLEMENT_COMPUTE_SINGULARITIES_KERNEL(Float, 6); 
// The maximum supported N is defined by `MaxN` in `dispatch.hpp`

IMPLEMENT_COMPUTE_SINGULARITIES_KERNEL_N(float);
IMPLEMENT_COMPUTE_SINGULARITIES_KERNEL_N(double);