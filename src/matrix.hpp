#pragma once

#include <math.h>

#include "common.hpp"

namespace dm
{

/** 
 * Non-owning wrapper around compact storage that represents a symmetric matrix.
 * 
 *          [a b c] 
 * A matrix [b d e] is assumed to be layed out in memory as [a b d c e f]
 *          [c e f]
 */
template<typename Float_, unsigned int N_>
struct SymmetricMatrix
{
    using Float = Float_;

	static constexpr unsigned int N   = N_;
	static constexpr size_t data_size = N_ * (N_ + 1) / 2;

	struct Storage
	{
		Float_ data[SymmetricMatrix<Float_, N_>::data_size] = { 0 };
	};

	DEVICE constexpr Float& operator() (unsigned int i, unsigned int j)
	{
		return data[i * (i + 1) / 2 + j];
	}

	DEVICE constexpr Float const& get_full (unsigned int i, unsigned int j) const
	{
		if (i < j)
		{
			unsigned int i_ = j; j = i; i = i_;
		}

		return data[i * (i + 1) / 2 + j];
	}

	Float* const data;
};

/**
 * Helper for looping over a compactly-stored symmetric matrix
 * (loops only over the lower triangular part)
 * 
 * Example usage:
 * 
 * SymmetricMatrix<Float, 3> m = ...
 * SYMMETRIC_MATRIX_LOOP(3,
 *    // do something with element (i, j)
 *    m(i, j)
 * )
 */
#define SYMMETRIC_MATRIX_LOOP(N, expr) \
for (unsigned int i = 0; i < N; ++i) \
{\
	for (unsigned int j = 0; j <= i; ++j)\
	{\
		expr;\
	}\
}

/**
 * Non - owning wrapper around compact storage that represents a lower triangular matrix.
 * 
 *          [a 0 0]
 * A matrix [b d 0] is assumed to be layed out in memory as [a b d c e f]
 *          [c e f]
 */
template<typename Float_, unsigned int N_>
struct LowerTriangularMatrix
{
    using Float = Float_;

    static constexpr unsigned int N = N_;
    static constexpr size_t data_size = N_ * (N_ + 1) / 2;

	struct Storage
	{
		Float_ data[LowerTriangularMatrix<Float_, N_>::data_size] = { 0 };
	};

	DEVICE constexpr Float& operator() (unsigned int i, unsigned int j)
	{
		return data[i * (i + 1) / 2 + j];
	}

	DEVICE constexpr Float get_full(unsigned int i, unsigned int j) const
	{
		if (i < j)
		{
			return 0;
		}

		return data[i * (i + 1) / 2 + j];
	}

	Float* data;
};

/**
 * Non - owning wrapper around compact storage that represents an upper triangular matrix.
 *
 *          [a b c]
 * A matrix [0 d e] is assumed to be layed out in memory as [a b d c e f]
 *          [0 0 f]
 */
template<typename Float_, unsigned int N_>
struct UpperTriangularMatrix
{
	using Float = Float_;

	static constexpr unsigned int N = N_;
	static constexpr size_t data_size = LowerTriangularMatrix<Float_, N_>::data_size;

	DEVICE constexpr Float& operator() (unsigned int i, unsigned int j)
	{
		LowerTriangularMatrix<Float_, N_> lower{ .data = data };
		return lower(j, i);
	}

	DEVICE constexpr Float get_full(unsigned int i, unsigned int j) const
	{
		LowerTriangularMatrix<Float_, N_> lower{ .data = data };
		return lower.get_full(j, i);
	}

	Float* data;
};

template<typename Float, unsigned int N>
DEVICE SymmetricMatrix<Float const, N> constant(SymmetricMatrix<Float, N> mat)
{
	return SymmetricMatrix<Float const, N> {.data = mat.data };
}

template<typename Float, unsigned int N>
DEVICE LowerTriangularMatrix<Float const, N> constant(LowerTriangularMatrix<Float, N> mat)
{
	return LowerTriangularMatrix<Float const, N> {.data = mat.data };
}

template<typename Float, unsigned int N>
DEVICE UpperTriangularMatrix<Float const, N> constant(UpperTriangularMatrix<Float, N> mat)
{
	return UpperTriangularMatrix<Float const, N> {.data = mat.data };
}

template<typename Float, unsigned int N>
DEVICE SymmetricMatrix<Float, N> transpose(SymmetricMatrix<Float, N> mat)
{
	return SymmetricMatrix<Float, N>{.data = mat.data};
}

template<typename Float, unsigned int N>
DEVICE UpperTriangularMatrix<Float, N> transpose(LowerTriangularMatrix<Float, N> mat)
{
	return UpperTriangularMatrix<Float, N>{mat.data};
}

template<typename Float, unsigned int N>
DEVICE LowerTriangularMatrix<Float, N> transpose(UpperTriangularMatrix<Float, N> mat)
{
	return LowerTriangularMatrix<Float, N>{.data = mat.data};
}

}