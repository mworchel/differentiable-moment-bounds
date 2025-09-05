#pragma once

#include <cstdint>
#include <cstdio>
#include <iterator>
#include <type_traits>

enum class Kernels : uint32_t
{
	ComputeMomentBounds = 0,
	ComputeMomentBoundsBackward,
	DetectBoundErrors,
	ComputeSingularities,
	NumKernels
};

template<typename Float>
constexpr char const* get_float_name()
{
	if constexpr (std::is_same_v<float, Float>)
		return "float";
	else
		return "double";
}

template<Kernels Kernel>
constexpr char const* get_generic_kernel_name()
{
	constexpr char const* names[] = {
		"k_compute_moment_bounds",
		"k_compute_moment_bounds_backward",
		"k_detect_bound_errors",
		"k_compute_singularities",
	};
	static_assert(std::size(names) == static_cast<uint32_t>(Kernels::NumKernels));
	static_assert(static_cast<uint32_t>(Kernel) < static_cast<uint32_t>(Kernels::NumKernels));
	return names[static_cast<uint32_t>(Kernel)];
}

template<Kernels Kernel, typename Float, unsigned int N, uint32_t Flags>
char const* get_kernel_name()
{
	// TODO: This buffer is created per function and not thread-safe
	static char buffer[256];
	snprintf(buffer, std::size(buffer), "%s_%s_%d_%d", get_generic_kernel_name<Kernel>(), get_float_name<Float>(), N, Flags);
	return buffer;
}

template<Kernels Kernel, typename Float, unsigned int N>
char const* get_kernel_name()
{
	// TODO: This buffer is created per function and not thread-safe
	static char buffer[256];
	snprintf(buffer, std::size(buffer), "%s_%s_%d", get_generic_kernel_name<Kernel>(), get_float_name<Float>(), N);
	return buffer;
}

template<Kernels Kernel, typename Float>
char const* get_kernel_name()
{
	// TODO: This buffer is created per function and not thread-safe
	static char buffer[256]; 
	snprintf(buffer, std::size(buffer), "%s_%s", get_generic_kernel_name<Kernel>(), get_float_name<Float>());
	return buffer;
}