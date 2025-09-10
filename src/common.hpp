#pragma once

/**
 * The same code is compiled for the CPU and the GPU,
 * so provide some defines to distinguish between
 * the versions during compilation.
 */
#ifdef __CUDACC__
#define IS_CUDA 1
#define DEVICE __device__
#define CONSTANT __constant__
#define MAYBE_STD(func) ::func
#define MAYBE_CUDA(func) ::func
#define IF_NOT_CUDA(expr)
#define FORCE_INLINE __forceinline__ 
#else
#define IS_CUDA 0
#define DEVICE
#define CONSTANT
// Mark a function that, on the CPU, lives in the `std` namespace
#define MAYBE_STD(func) std::func
// Mark a function that is available only in CUDA
#define MAYBE_CUDA(func) ::cuda_replacement::func
#define IF_NOT_CUDA(expr) expr
#define FORCE_INLINE

#if !IS_CUDA
#include <bit>
#endif
#include <cmath>
#include <cstdint>
#include <type_traits>

namespace cuda_replacement
{
	template<typename Float>
	void sincos(Float v, Float* s, Float* c)
	{
		*s = ::std::sin(v);
		*c = ::std::cos(v);
	}
}
#endif

namespace dm
{

/**
 * Store bits inside an IEEE 754-1985 floating point NaN.
 * - 22 bits for 32-bit float
 * - 32 bits for 64-bit double
 * (just generally use this with care...)
 */
template<typename Float>
DEVICE Float encode_bits(uint32_t bits)
{
    static_assert(std::is_same_v<Float, float> || std::is_same_v<Float, double>, 
                  "encode_bits only supports float and double");

	if constexpr (std::is_same_v<Float, float>)
	{
		// Encode bits in the mantissa of a quiet NaN
		//                               exponent |      payload       |
		constexpr uint32_t nan_base = 0b01111111110000000000000000000000;
		//                              |        |       
		//                            sign     quiet        

        // Ensure we don't overflow the available 22 payload bits
        constexpr uint32_t max_bits = (1u << 22) - 1;
        bits = bits & max_bits;

        uint32_t nan_flagged = nan_base | bits;

#if IS_CUDA
		return __uint_as_float(nan_flagged);
#else
		return std::bit_cast<Float>(nan_flagged);
#endif
	}
	else if constexpr (std::is_same_v<Float, double>)
	{
		// Encode bits in the mantissa of a quiet NaN
		//                               |exponent | |                     payload                     |
		constexpr uint64_t nan_base = 0b0111111111111000000000000000000000000000000000000000000000000000;
		//                              |           |       
		//                            sign        quiet  

		// For double, all 32 bits are available

		uint64_t nan_flagged = nan_base | bits;

#if IS_CUDA
		// Static cast to (signed) long long is fine because nan_flagged will be < 2^63 - 1
		return __longlong_as_double(static_cast<long long>(nan_flagged));
#else
		return std::bit_cast<Float>(nan_flagged);
#endif
	}
}

// Retrieve bits stored inside an IEEE 754-1985 floating point NaN.
template<typename Float>
DEVICE uint32_t decode_bits(Float value)
{
	if constexpr (std::is_same_v<Float, float>)
	{
		uint32_t bits = 
#if IS_CUDA
			__float_as_uint(value);
#else
			std::bit_cast<uint32_t>(value);
#endif

		// Mask the payload bits
		constexpr uint32_t mask = (1u << 22) - 1;

		return bits & mask;
	}
	else if constexpr (std::is_same_v<Float, double>)
	{
		uint64_t bits =
#if IS_CUDA
			// Static cast to unsigned preserves the bit pattern
			static_cast<uint64_t>(__double_as_longlong(value));
#else
			std::bit_cast<uint64_t>(value);
#endif

		// Mask the payload bits
		constexpr uint64_t mask = (1ull << 32) - 1;

		return static_cast<uint32_t>(bits & mask);
	}
}

template<typename T>
DEVICE inline void constexpr swap(T& v1, T& v2)
{
	T tmp(v1);
	v1 = v2;
	v2 = tmp;
}

// Compute a*b - c*d using Kahan's algorithm
template<typename Float>
DEVICE Float kahan(Float a, Float b, Float c, Float d)
{
	Float cd = c * d;
	Float error = MAYBE_STD(fma)(c, d, -cd);
	return MAYBE_STD(fma)(a, b, -cd) - error;
}

// Copy a constant number of elements from src to dst (really just to replace memcpy)
template<unsigned int N, typename Float> 
DEVICE void fixed_copy(Float* dst, Float const* src)
{
	for (unsigned int i = 0; i < N; ++i)
		dst[i] = src[i];
}

template<typename Float>
DEVICE Float clamp_absmin(Float const v, Float const v_min)
{
	return MAYBE_STD(abs)(v) < v_min ? MAYBE_STD(copysign)(v_min, v) : v;
}

template<typename Float>
DEVICE Float clamp_absmin_backward(Float const v, Float const v_min, Float const dout)
{
	return MAYBE_STD(abs)(v) < v_min ? Float(0) : dout;
}

} // namespace dm