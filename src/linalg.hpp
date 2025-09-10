#pragma once

#ifndef __CUDACC__
	#include <algorithm>
	#include <cmath>
#endif

#include "common.hpp"
#include "matrix.hpp"
#include "polynomial.hpp"

#define DM_CHOLESKY_ALTERNATIVE 1

namespace dm
{
	// Perform the Cholesky factorization of a symmetric, positive-definite matrix A
	template<typename Float, unsigned int N>
	/*__forceinline__*/ DEVICE bool cholesky(SymmetricMatrix<Float const, N> A, LowerTriangularMatrix<Float, N> L)
	{
#if IS_CUDA
#pragma unroll
#endif
		for (unsigned int i = 0; i < N; ++i)
		{
#if !DM_CHOLESKY_ALTERNATIVE
			Float row_sum(0);
			for (unsigned int j = 0; j < i; j++)
			{
				Float sum(0);
				for (unsigned int k = 0; k < j; ++k)
				{
					sum += L(i, k) * L(j, k);
				}
				Float a = (A(i, j) - sum) / L(j, j);

				L(i, j) = a;
				row_sum += a * a;
			}

			Float a = A(i, i) - row_sum;
			// Check if the matrix can be factorized
			if (a <= Float(0)) [[unlikely]]
				return false;

			L(i, i) = MAYBE_STD(sqrt)(a);
#else
			Float row_sum(A(i, i));
			for (unsigned int j = 0; j < i; j++)
			{
				Float sum(A(i, j));
				for (unsigned int k = 0; k < j; ++k)
				{
					sum -= L(i, k) * L(j, k);
				}
				Float a = sum / L(j, j);

				L(i, j) = a;
				row_sum -= a*a;
			}

			Float a = row_sum;

			// Check if the matrix can be factorized
			if (a <= Float(0)) [[unlikely]]
				return false;

			L(i, i) = MAYBE_STD(sqrt)(a);
#endif
		}

		return true;
	}

	template<typename Float, unsigned int N>
	DEVICE void forward_substitution(LowerTriangularMatrix<Float const, N> L, Float const* b, Float* x)
	{
		for (unsigned int i = 0; i < N; ++i)
		{
			Float sum(0);
			for (unsigned int j = 0; j < i; ++j)
			{
				sum += x[j] * L(i, j);
			}

			x[i] = (b[i] - sum) / L(i, i);
		}
	}

	template<typename Float, unsigned int N>
	DEVICE void backward_substitution(UpperTriangularMatrix<Float const, N> L, Float const* b, Float* x)
	{
		for (unsigned int i_ = 0; i_ < N; ++i_)
		{
			unsigned int i = (N - 1) - i_;

			Float sum(0);
			for (unsigned int j = i + 1; j < N; ++j)
			{
				sum += x[j] * L(i, j);
			}

			x[i] = (b[i] - sum) / L(i, i);
		}
	}

	template<typename Float, unsigned int N>
	DEVICE void cholesky_solve(LowerTriangularMatrix<Float const, N> L, Float const* b, Float* x)
	{
		Float y[N] = {0};
		forward_substitution<Float, N>(L, b, y);
		backward_substitution<Float, N>(dm::transpose(L), y, x); // Really a good idea? haha
	}

	/**
	 * Solve a Vandermonde linear system using the Bjoerck-Pereyra Algorithm
	 * 
	 * [  1   ...   1  ] [x_1]   [b_1]
     * [a_1   ... a_n  ] [x_2]   [b_2]
     * [a_1^2 ... a_n^2] [x_3] = [b_3]
     * [ :     :   :   ] [ : ]   [ : ]
     * [a_1^n ... a_n^n] [x_n]   [b_n]
	 */
	template<typename Float, unsigned int N, bool Sort = true>
	DEVICE void bjoerck_pereyra_solve(Float const* a, Float const* b, Float* x)
	{
		// Without sorting, these are unused variables (and hopefully optimized away)
		unsigned int a_perm[N];
		Float        x_sorted[N];
		Float        a_sorted[N];
		Float*       x_output = x;

		if constexpr (Sort)
		{
			// Sort the rows by magnitude using Bubble sort
			for (unsigned int i = 0; i < N; ++i)
			{
				a_perm[i] = i;
				a_sorted[i] = a[i];
			}

			for (unsigned int i = 0; i < N; ++i)
			{
				for (unsigned int j = i + 1; j < N; ++j)
				{
					if (MAYBE_STD(abs)(a_sorted[i]) > MAYBE_STD(abs)(a_sorted[j]))
					{
						swap(a_sorted[i], a_sorted[j]);
						swap(a_perm[i], a_perm[j]);
					}
				}
			}

			// Overwrite the pointers, so the main body uses the standard variables
			x = x_sorted;
			a = a_sorted;
		}

		// Main body of the Bjoerck-Pereyra algorithm
		for (int i = 0; i < static_cast<int>(N); ++i)
		{
			x[i] = b[i];
		}

		constexpr int const n = static_cast<int>(N - 1);

		for (int k = 0; k < n; ++k)
		{
			for (int j = n; j >= k + 1; --j)
			{
				x[j] = MAYBE_STD(fma)(-a[k], x[j - 1], x[j]);
			}
		}

		for (int k = n - 1; k >= 0; --k)
		{
			for (int j = k + 1; j <= n; ++j)
			{
				x[j] /= a[j] - a[j - k - 1];
				x[j - 1] -= x[j];
			}
		}

		if constexpr (Sort)
		{
			// Permute the weights (avoid dynamic indexing == register spilling)
			for (int i = 0; i < static_cast<int>(N); ++i)
			{
				for (int j = 0; j < static_cast<int>(N); j++)
				{
					if (a_perm[i] == j)
						x_output[j] = x_sorted[i];
				}
			}
		}
	}

	template<typename Float, unsigned int N, bool Sort = true>
	DEVICE void bjoerck_pereyra_dual_solve(Float const* a, Float const* f, Float* x)
	{
		Float a_sorted[N];
		Float f_sorted[N];
		
		if constexpr (Sort)
		{
			for (unsigned int i = 0; i < N; ++i)
			{
				a_sorted[i] = a[i];
				f_sorted[i] = f[i];
			}

			// Sort the rows by magnitude of the evaluation point using Bubble sort
			for (unsigned int i = 0; i < N; ++i)
			{
				for (unsigned int j = i + 1; j < N; ++j)
				{
					if (MAYBE_STD(abs)(a_sorted[i]) > MAYBE_STD(abs)(a_sorted[j]))
					{
						swap(a_sorted[i], a_sorted[j]);
						swap(f_sorted[i], f_sorted[j]);
					}
				}
			}

			a = a_sorted;
			f = f_sorted;
		}

		for (int i = 0; i < static_cast<int>(N); ++i)
		{
			x[i] = f[i]; // c^0 = f
		}

		constexpr int const n = static_cast<int>(N - 1);

		for (int k = 0; k < n; ++k)
		{
			for (int j = n; j >= k + 1; --j)
			{
				x[j] = (x[j] - x[j - 1]) / (a[j] - a[j - k - 1]); // c^{k+1}_j = c^{k}_j
			}
		}

		for (int k = n - 1; k >= 0; --k)
		{
			for (int j = k; j <= n - 1; ++j)
			{
				x[j] = MAYBE_STD(fma)(- a[k], x[j + 1], x[j]);
			}
		}
	}

	template<typename Float, unsigned int N>
	DEVICE void vandermonde_solve(Float const* a, Float const* b, Float* x)
	{
		constexpr int N_ = static_cast<int>(N);

		if constexpr (N_ == 1)
		{
			x[0] = b[0];
			return;
		}

		Float c[N_] = { 0 };
		c[N_ - 1] = -a[0];

		for (int i = 1; i < N_; ++i)
		{
			Float aa = -a[i];
			for (int j = N_ - (i+1); j < N_ - 1; ++j)
			{
				c[j] += aa * c[j + 1];
			}
			c[N_ - 1] += aa;
		}

		for (int i = 0; i < N_; ++i)
		{
			Float aa = a[i];
			Float t(1);
			Float u(1); // b = u
			Float s = b[N_ - 1];
			for (int k = N - 1; k >= 1; --k)
			{
				u = MAYBE_STD(fma)(aa, u, c[k]);
				s += b[k - 1] * u;
				t = MAYBE_STD(fma)(aa, t, u);
			}
			x[i] = s / t;
		}
	}

	template<typename Float, unsigned int N>
	DEVICE void cholesky_solve_backward(LowerTriangularMatrix<Float const, N> L, Float const* b, Float const* x, SymmetricMatrix<Float, N> dA, Float* db, Float const* dx)
	{
		// Use the adjoints 

		//  A^T db = dx
		cholesky_solve(L, dx, db);

		// dA = -x * db^T
		for (unsigned int i = 0; i < N; ++i)
		{
			for (unsigned int j = 0; j <= i; ++j)
			{
				dA(i, j) = - x[i] * db[j];
				
				if (i != j)
					dA(i, j) = Float(0.5) * (dA(i, j) - x[j] * db[i]); // dA is symmetric
			}
		}
	}

	template<typename Float, unsigned int N>
	DEVICE void cholesky_solve_forward(LowerTriangularMatrix<Float const, N> L, Float const* b, Float const* x, LowerTriangularMatrix<Float const, N> dA, Float const* db, Float* dx)
	{
		// TODO
	}

	template<typename Float, unsigned int N>
	DEVICE void bjoerck_pereyra_solve_backward(Float const* a, Float const* b, Float const* x, Float* da, Float* db, Float const* dx)
	{
		constexpr int iN = static_cast<int>(N);

		// Solve the system V^T db = dx
		bjoerck_pereyra_dual_solve<Float, N>(a, dx, db);

		// dV = -x * db^T
		for (int i = 0; i < iN; ++i)
		{
			// We start with j=1 because the first coefficient doesn't matter..
			Float c[N];
			for (int j = 1; j < iN; ++j)
			{
				c[j] = x[i] * db[j];
			}

			// Attention: if `a[i]` is +-inf, then x[i] will be zero, so all c should be zero
			// However, evaluating the polynomial below will return NaN, so da[i] is going to be NaN.

			// N-1, this is the DEGREE and the DEGREE is N-1!
			// FIXME: This can be NaN but a[i] != inf! (fp accuracy)
			da[i] = -evaluate_polynomial_derivative<Float, N-1>(c, a[i]);
		}
	}

	template<unsigned int N, typename Float>
	/*__forceinline__*/ DEVICE Float dot(Float const* v, Float const* w)
	{
		Float value(0);
		for (unsigned int i = 0; i < N; ++i)
		{
			value += v[i] * w[i];
		}
		return value;
	}

	template<unsigned int N, typename Float>
	DEVICE void add(Float const value, LowerTriangularMatrix<Float, N> M)
	{
		for (unsigned int i = 0; i < N; ++i)
		{
			for (unsigned int j = 0; j <= i; ++j)
			{
				M(i, j) += value;
			}
		}
	}

	template<unsigned int N, typename Float>
	DEVICE void add(LowerTriangularMatrix<Float const, N> A, LowerTriangularMatrix<Float, N> B)
	{
		for (unsigned int i = 0; i < N; ++i)
		{
			for (unsigned int j = 0; j <= i; ++j)
			{
				B(i, j) += A(i, j);
			}
		}
	}

	template<unsigned int N, typename Float>
	DEVICE void add(SymmetricMatrix<Float const, N> in1, SymmetricMatrix<Float const, N> in2, SymmetricMatrix<Float, N> out)
	{
		SYMMETRIC_MATRIX_LOOP(N,
			out(i, j) = in1(i, j) + in2(i, j);
		);
	}

	template<unsigned int N, typename Float>
	DEVICE void cmul(SymmetricMatrix<Float const, N> in1, SymmetricMatrix<Float const, N> in2, SymmetricMatrix<Float, N> out)
	{
		SYMMETRIC_MATRIX_LOOP(N,
			out(i, j) = in1(i, j) * in2(i, j);
		);
	}

	template<unsigned int N, typename Float>
	DEVICE void mul(SymmetricMatrix<Float const, N> in_mat, Float const in_val, SymmetricMatrix<Float, N> out_mat)
	{
		SYMMETRIC_MATRIX_LOOP(N,
			out_mat(i, j) = in_val * in_mat(i, j);
		);
	}

	template<unsigned int N, typename Float>
	DEVICE Float sum(SymmetricMatrix<Float const, N> mat)
	{
		Float sum_(0);
		SYMMETRIC_MATRIX_LOOP(N,
			sum_ += (i == j) ? mat(i, j) : 2 * mat(i, j);
		);
		return sum_;
	}

	template<unsigned int N, typename Float>
	DEVICE Float abssum(Float const* vector)
	{
		Float sum_(0);
		for (unsigned int i = 0; i < N; i++)
		{
			sum_ += MAYBE_STD(abs)(vector[i]);
		}
		return sum_;
	}

	template<unsigned int N, typename Float>
	DEVICE Float abssum(SymmetricMatrix<Float const, N> mat)
	{
		Float sum_(0);
		SYMMETRIC_MATRIX_LOOP(N,
			sum_ += (i == j) ? MAYBE_STD(abs)(mat(i, j)) : 2 * MAYBE_STD(abs)(mat(i, j));
		);
		return sum_;
	}

	template<unsigned int N, typename Float>
	/*__forceinline__*/ DEVICE void add(Float const* a, Float const* b, Float* out)
	{
		for (unsigned int i = 0; i < N; ++i)
		{
			out[i] = a[i] + b[i];
		}
	}

	template<unsigned int N, typename Float>
	/*__forceinline__*/ DEVICE void mul(Float const* a, Float const value, Float* out)
	{
		for (unsigned int i = 0; i < N; ++i)
		{
			out[i] = a[i] * value;
		}
	}

	template<unsigned int N, typename Float>
	/*__forceinline__*/ DEVICE void sub(Float const* a, Float const* b, Float* out)
	{
		for (unsigned int i = 0; i < N; ++i)
		{
			out[i] = a[i] - b[i];
		}
	}

	template<unsigned int N, typename Float>
	/*__forceinline__*/ DEVICE void fma(Float const a, Float const* b, Float const* c, Float* out)
	{
		for (unsigned int i = 0; i < N; ++i)
		{
			out[i] = MAYBE_STD(fma)(a, b[i], c[i]);
		}
	}

	template<unsigned int N, typename Float>
	/*__forceinline__*/ DEVICE void lerp(Float const* v1, Float const* v2, Float const t, Float* out)
	{
		// See https://developer.nvidia.com/blog/lerp-faster-cuda/
		for (unsigned int i = 0; i < N; ++i)
		{
			out[i] = MAYBE_STD(fma)(t, v2[i], MAYBE_STD(fma)(-t, v1[i], v1[i]));
		}
	}

	template<unsigned int N, typename Float>
	DEVICE void cmul(Float const value, LowerTriangularMatrix<Float, N> matrix)
	{
		for (unsigned int i = 0; i < N; ++i)
		{
			for (unsigned int j = 0; j <= i; ++j)
			{
				matrix(i, j) *= value;
			}
		}
	}

	template<unsigned int N, typename Float>
	DEVICE void mul(LowerTriangularMatrix<Float const, N> M, Float const* v, Float* out)
	{
		for (unsigned int i = 0; i < N; ++i)
		{
			out[i] = Float(0);
			for (unsigned int j = 0; j <= i; ++j)
			{
				out[i] += M(i, j) * v[j];
			}
		}
	}

	template<unsigned int N, typename Float>
	DEVICE void cmul(LowerTriangularMatrix<Float const, N> mat_in_1, LowerTriangularMatrix<Float const, N> mat_in_2, LowerTriangularMatrix<Float, N> mat_out)
	{
		for (unsigned int i = 0; i < N; ++i)
		{
			for (unsigned int j = 0; j <= i; ++j)
			{
				mat_out(i, j) = mat_in_1(i, j) * mat_in_2(i, j);
			}
		}
	}

	template<unsigned int N, typename Float>
	DEVICE void mul(LowerTriangularMatrix<Float const, N> L, UpperTriangularMatrix<Float const, N> U, SymmetricMatrix<Float, N> M)
	{
		for (unsigned int i = 0; i < N; ++i)
		{
			for (unsigned int j = 0; j <= i; ++j)
			{
				Float sum(0);
				for (unsigned int k = 0; k <= MAYBE_STD(min)(i, j); ++k)
				{
					sum += L(i, k) * U(k, j);
				}
				M(i, j) = sum;
			}
		}
	}

	template<unsigned int N, typename Float>
	DEVICE Float sum(LowerTriangularMatrix<Float const, N> mat)
	{
		Float sum_(0);
		for (unsigned int i = 0; i < N; ++i)
		{
			for (unsigned int j = 0; j <= i; ++j)
			{
				sum_ += (i == j) ? mat(i, j) : 2*mat(i, j);
			}
		}
		return sum_;
	}

	template<unsigned int N, typename Float>
	DEVICE void mul(UpperTriangularMatrix<Float const, N> M, Float const* v, Float* out)
	{
		for (unsigned int i = 0; i < N; ++i)
		{
			out[i] = Float(0);
			for (unsigned int j = i; j < N; ++j)
			{
				out[i] += M(i, j) * v[j];
			}
		}
	}

	template<typename Float, unsigned int N>
	DEVICE Float det(LowerTriangularMatrix<Float, N> mat)
	{
		std::decay_t<Float> result(1.f);
		for (unsigned int i = 0; i < N; ++i)
		{
			result *= mat(i, i);
		}
		return result;
	}

	template<typename Float, unsigned int N>
	DEVICE Float det(UpperTriangularMatrix<Float, N> mat)
	{
		return det(transpose(mat));
	}
}