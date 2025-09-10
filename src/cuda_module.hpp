#pragma once

#include <cassert>
#include <cstdint>
#include <type_traits>

#include "cuda_driver.hpp"
#include "kernels.hpp"

extern CUmodule cu_module;

template<Kernels Kernel, typename Float, unsigned int N, uint32_t Flags>
CUfunction get_kernel(int device_id)
{
	// Currently only supports device 0
	assert(device_id == 0); 

	// Load the function from the module
	CUfunction kernel = nullptr;
	char const* kernel_name = get_kernel_name<Kernel, Float, N, Flags>();
	cuda_check(cuModuleGetFunction(&kernel, cu_module, kernel_name));

	return kernel;
}

template<Kernels Kernel, typename Float, unsigned int N>
CUfunction get_kernel(int device_id)
{
	// Currently only supports device 0
	assert(device_id == 0);

	// Load the function from the module
	CUfunction kernel = nullptr;
	char const* kernel_name = get_kernel_name<Kernel, Float, N>();
	cuda_check(cuModuleGetFunction(&kernel, cu_module, kernel_name));

	return kernel;
}

template<Kernels Kernel, typename Float>
CUfunction get_kernel(int device_id)
{
	// Currently only supports device 0
	assert(device_id == 0);

	// Load the function from the module
	CUfunction kernel = nullptr;
	char const* kernel_name = get_kernel_name<Kernel, Float>();
	cuda_check(cuModuleGetFunction(&kernel, cu_module, kernel_name));

	return kernel;
}

void load_module();

void unload_module();