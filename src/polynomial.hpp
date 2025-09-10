#pragma once

#include "common.hpp"

#define DM_UNROLL_NEWTON_CALLS 1

namespace dm
{

template<typename Float>
DEVICE inline Float sign(Float v)
{
	return v < 0 ? Float(-1) : Float(1);
}

template<typename Float, unsigned int Degree>
DEVICE Float evaluate_polynomial(Float const* coeffs, Float const x)
{
	Float b = coeffs[Degree];
	for (int i = static_cast<int>(Degree) - 1; i >= 0; --i)
	{
		b = MAYBE_STD(fma)(b, x, coeffs[i]);
	}
	return b;
}

template<typename Float, unsigned int Degree>
DEVICE Float evaluate_polynomial_at_infinity(Float const* coeffs, Float const x)
{
	Float cn           = coeffs[Degree];
	bool is_odd_degree = Degree & 1;

	// Handle vanishing leading coefficient (the exact comparison to zero is intended)
	if (cn == Float(0))
	{
		cn = coeffs[Degree-1];
		is_odd_degree = !is_odd_degree;
	}

	if (x < 0)
		return sign(cn) * (is_odd_degree ? Float(-1) : Float(1)); // Get the sign at f(-inf)
	else
		return sign(cn); // Get the sign at f(inf)
}

template<typename Float, unsigned int Degree>
DEVICE Float evaluate_polynomial_derivative(Float const* coeffs, Float const x)
{
	// Polynomial: [  c0,   c1,  c2, ..., cn-1, cn]
	// Derivative: [1*c1, 2*c2,3*c3, ..., n*cn,  0]

	constexpr int Degree_ = static_cast<int>(Degree);

	Float b = Float(Degree_) * coeffs[Degree_];
	for (int i = Degree_ - 1; i > 0; --i)
	{
		b = MAYBE_STD(fma)(b, x, Float(i) * coeffs[i]);
	}
	return b;
}

// Combined method to evaluate a polynomial and its derivative
template<typename Float, unsigned int Degree>
DEVICE void evaluate_polynomial_and_derivative(Float const* coeffs, Float const x, Float& f, Float& df)
{
	constexpr int Degree_ = static_cast<int>(Degree);

	Float b  = coeffs[Degree_];
	Float db(0);
	for (int i = Degree_ - 1; i >= 0; --i)
	{
		db = MAYBE_STD(fma)(db, x, b);
		b  = MAYBE_STD(fma)(b, x, coeffs[i]);
	}
	
	f  = b;
	df = db;
}


/**
 * Expand a polynomial that is given in terms of its roots to a coefficient representation.
 */
template<typename Float, unsigned int Degree>
DEVICE void expand_polynomial(Float const* roots, Float* coeffs)
{
	// Convention: c0 * x^0 + c1 * x^1 + c2 * x^2 + ...
	// General idea:
	// Start with
	// [1, 0, 0, 0, 0, 0, 0] 
	// then compute in each iteration:
	// [    0,    c0,    c1, ..., c(n-1), cn, 0, 0] + 
	// [-r*c0, -r*c1, -r*c2, ...,  -r*cn,  0, 0, 0]
	
	coeffs[0] = Float(1);
	for (unsigned int i = 1; i < Degree + 1; ++i)
		coeffs[i] = Float(0);

	for (unsigned int i = 0; i < Degree; ++i)
	{
		Float root = roots[i];

		// Roots at infinite reduce the degree
		if (MAYBE_STD(isinf(root)))
			continue;

		for (unsigned int k = i + 1; k > 0; --k)
		{
			coeffs[k] = MAYBE_STD(fma)(coeffs[k], -root, coeffs[k - 1]);
		}
		coeffs[0] *= -root;
	}
}

template<typename Float, unsigned int Degree>
DEVICE Float upper_bound_polynomial_roots(Float const* coeffs)
{
	// Compute Cauchy's bound for the roots' absolute value
	Float cn        = coeffs[Degree];
	Float ratio_max = MAYBE_STD(abs)(coeffs[0] / cn);
	for (unsigned int i = 1; i < Degree; ++i)
	{
		Float ratio = MAYBE_STD(abs)(coeffs[i] / cn);
		if (ratio > ratio_max)
			ratio_max = ratio;
	}
	return 1 + ratio_max;
}

enum class IntervalType
{
	Closed,
	OpenLower,
	OpenUpper,
};

enum class NewtonResult
{
	Converged,
	MaxIteration,
	NoRoot,
};

template<typename Float>
DEVICE constexpr Float newton_initial_guess(IntervalType interval, Float lower, Float upper)
{
	if (interval == IntervalType::Closed)
	{
		return Float(0.5) * (upper + lower);
	}
	else if (interval == IntervalType::OpenLower)
	{
		return upper - Float(2);
	}
	else // OpenUpper
	{
		return lower + Float(2);
	}
}

template<typename Float>
DEVICE constexpr Float newton_next_bisect(IntervalType interval, Float lower, Float upper)
{
	if (interval == IntervalType::Closed)
	{
		return Float(0.5) * (upper + lower);
	}
	else if (interval == IntervalType::OpenLower)
	{
		return upper - Float(2);
	}
	else // OpenUpper
	{
		return lower + Float(2);
	}
}

// Root-finding method by Yuksel (2022) "High-Performance Polynomial Root Finding for Graphics"
template<typename Float, unsigned int Degree>
DEVICE NewtonResult newton_bisection(Float const* coeffs, Float lower, Float upper, Float tolerance, unsigned int max_iterations, Float& root)
{
	Float lower_value = evaluate_polynomial<Float, Degree>(coeffs, lower);
	Float upper_value = evaluate_polynomial<Float, Degree>(coeffs, upper);

	IntervalType interval = IntervalType::Closed;
	if (MAYBE_STD(isinf)(lower))
	{
		lower_value = evaluate_polynomial_at_infinity<Float, Degree>(coeffs, lower);
		interval = IntervalType::OpenLower;
	}
	if (MAYBE_STD(isinf)(upper))
	{
		upper_value = evaluate_polynomial_at_infinity<Float, Degree>(coeffs, upper);
		interval = IntervalType::OpenUpper;
	}

	if (lower_value * upper_value > Float(0))
		return NewtonResult::NoRoot;

	Float current = newton_initial_guess(interval, lower, upper);
	for (unsigned int i = 0; i < max_iterations; ++i)
	{
		Float current_value;
		Float current_deriv;
		evaluate_polynomial_and_derivative<Float, Degree>(coeffs, current, current_value, current_deriv);

		// Replace the lower bound with current if they have the same sign
		bool replace_lower = current_value * lower_value > Float(0);
		lower = replace_lower ? current : lower;
		upper = replace_lower ? upper : current;

		// If the lower/upper bound is replaced, they're not open anymore
		if (interval == IntervalType::OpenLower && replace_lower)
			interval = IntervalType::Closed;

		if (interval == IntervalType::OpenUpper && !replace_lower)
			interval = IntervalType::Closed;

		// Choose between either a newton step or bisection for the next position
		// NOTE: Newton estimate can be NaN due to inf value/derivative (float overflow)
		Float next_newton = current - current_value / current_deriv;
		Float next_bisect = newton_next_bisect(interval, lower, upper);
		Float next = (next_newton <= lower || next_newton >= upper || !MAYBE_STD(isfinite)(next_newton)) ? next_bisect : next_newton;

		// NOTE: If current == 0, then rel_difference will be inf (never passing the tolerance)
		Float rel_difference = MAYBE_STD(abs)(current - next) / MAYBE_STD(abs)(current);
		if (rel_difference < tolerance)
		{
			root = current;
			return NewtonResult::Converged;
		}

		current = next;
	}

	root = current;
	return NewtonResult::MaxIteration;
}

enum class RootFindingResult : uint8_t
{
	Success = 0,
	Failed,
	Unknown
};

// Find roots of a linear polynomial p(x) = c0 + c1*x
template<typename Float>
DEVICE RootFindingResult find_real_linear_polynomial_root(Float const c0, Float const c1, Float& root)
{
	// This intentionally evalutes to +-inf for c1 == 0
	root = -c0 / c1;

	return RootFindingResult::Success;
}

// Find roots of a quadratic polynomial p(x) = c0 + c1*x + c2*x^2
template<typename Float>
DEVICE RootFindingResult find_real_quadratic_polynomial_roots(Float const c0, Float const c1, Float const c2, Float& root1, Float& root2)
{
	// Float d = c1 * c1 - 4 * c2 * c0;
	Float d = kahan(c1, c1, 4 * c2, c0);
	if (d < Float(0)) [[unlikely]]
		return RootFindingResult::Failed;

	// The second root intentionally evalutes to +-inf for c2 == 0
	Float r = -Float(0.5) * (MAYBE_STD(copysign)(MAYBE_STD(sqrt)(d), c1) + c1);
	root1 = c0 / r;
	root2 = r / c2;

	// Sort the roots ascending
	if (root1 > root2)
		dm::swap(root1, root2);

	return RootFindingResult::Success;
}

/**
 * Find roots of an n-th degree polynomial using a bracketed Newton approach
 * This is an implementation of the paper Yuksel (2022) "High-Performance Polynomial Root Finding for Graphics".
 * The construction of the (scaled) derivative polynomials follows this blog post by Christoph Peters: 
 * https://momentsingraphics.de/GPUPolynomialRoots.html
 */
template<typename Float, unsigned int Degree, unsigned int EndDegree>
DEVICE RootFindingResult find_real_ndegree_polynomial_roots(Float const* coeffs, Float const lower, Float const upper, Float* derivative, Float* roots, Float tolerance, unsigned int max_iterations)
{
	if constexpr (Degree == 2)
	{
		// Construct the quadratic derivative polynomial from the original coefficients
		derivative[0] = coeffs[EndDegree - 2];
		derivative[1] = Float(EndDegree - 1) * coeffs[EndDegree - 1];
		derivative[2] = (0.5 * Float((EndDegree - 1) * EndDegree)) * coeffs[EndDegree - 0];

		return find_real_quadratic_polynomial_roots(derivative[0], derivative[1], derivative[2], roots[0], roots[1]);
	}
	else
#if DM_UNROLL_NEWTON_CALLS
		if constexpr (Degree <= EndDegree)
#endif
	{
		// Compute the derivative of degree `Degree` by integrating the previous derivative of degree `Degree-1`
		constexpr Float prev_derivative_order = static_cast<Float>(EndDegree + 1 - Degree);
#if IS_CUDA
#pragma unroll
#endif
		for (unsigned int i = Degree; i > 0; --i)
		{
			derivative[i] = derivative[i - 1] * (prev_derivative_order / Float(i));
		}
		derivative[0] = coeffs[EndDegree - Degree];

		// Determine all roots of the derivative polynomial
		RootFindingResult result = RootFindingResult::Success;
#if IS_CUDA
#pragma unroll
#endif
		for (int k = static_cast<int>(Degree) - 1; k >= 0; --k)
		{
			bool is_first = (k == 0);
			bool is_last = (k == static_cast<int>(Degree) - 1);

			Float local_lower = is_first ? lower : roots[k - 1];
			Float local_upper = is_last ? upper : roots[k];
			NewtonResult newton_result = newton_bisection<Float, Degree>(derivative, local_lower, local_upper, tolerance, max_iterations, roots[k]);

			if (newton_result == NewtonResult::NoRoot) [[unlikely]]
			{
				// No root in the first/last interval is fine *if* both bounds are infinite (means one root is at infinity)
				bool can_have_no_root = (is_first || is_last) && MAYBE_STD(isinf)(local_lower) && MAYBE_STD(isinf)(local_upper);
				if (can_have_no_root)
				{
					roots[k] = MAYBE_STD(copysign)(Float(INFINITY), is_last ? local_upper : local_lower);
				}
				else
				{
					result = RootFindingResult::Failed;
#if !IS_CUDA
					// Stop on the CPU and continue on CUDA to avoid a dynamic loop count,
					// and therefore dynamic indexing and register spilling.
					break;
#endif
				}
			}
		}

		return result;
	}

#if !DM_UNROLL_NEWTON_CALLS
	if constexpr (Degree < EndDegree)
	{
		find_real_ndegree_polynomial_roots<Float, Degree + 1, EndDegree>(coeffs, lower, upper, derivative, roots, tolerance, max_iterations);
	}
#endif

	return RootFindingResult::Success;
}

enum RootFindingFlags : uint32_t
{
	None            = 0,
	CubicFastPathCP = 1,
	CubicFastPathMA = 2, // Removed
};

/**
 * Requirements:
 * - all roots are real
 * - all roots have multiplicity 1
 */
template<unsigned int Degree, typename Float, uint32_t Flags = RootFindingFlags::None>
DEVICE RootFindingResult find_real_polynomial_roots(Float const coeffs[Degree + 1], Float roots[Degree], Float tolerance, unsigned int max_iterations = 100u)
{
	if constexpr (Degree == 1)
	{
		return find_real_linear_polynomial_root<Float>(coeffs[0], coeffs[1], roots[0]);
	}
	else if constexpr (Degree == 2)
	{
		return find_real_quadratic_polynomial_roots<Float>(coeffs[0], coeffs[1], coeffs[2], roots[0], roots[1]);
	}
	else if constexpr (Degree == 3 && (Flags & RootFindingFlags::CubicFastPathCP))
	{
		// See: https://momentsingraphics.de/CubicRoots.html

		// Normalize the polynomial
		// and divide middle coefficients by three
		Float w = coeffs[3];
		Float x = coeffs[0] / w;
		Float y = (coeffs[1] / w) / Float(3);
		Float z = (coeffs[2] / w) / Float(3);

		// Compute the Hessian and the discrimant
		Float delta[3] = {
			MAYBE_STD(fma)(-z, z, y),
			MAYBE_STD(fma)(-y, z, x),
			MAYBE_STD(fma)(z, x, - y * y)
		};

		// NOTE: This multiplication can overflow for small w (large x, y, z) and result in inf-inf = nan
		Float discriminant = (Float(4.0) * delta[0]) * delta[2] - delta[1] * delta[1];
		if (discriminant < 0) [[unlikely]]
			return RootFindingResult::Failed;

		// Compute coefficients of the depressed cubic 
		// (third is zero, fourth is one)
		Float depressed[2] = {
			MAYBE_STD(fma)(-Float(2) * z, delta[0], delta[1]),
			delta[0]
		};

		// Take the cubic root of a normalized complex number
		Float theta = MAYBE_STD(atan2)(MAYBE_STD(sqrt)(discriminant), -depressed[0]) / Float(3.0);
		Float cubic_root[2];
		MAYBE_CUDA(sincos)(theta, &cubic_root[1], &cubic_root[0]);

		// Compute the three roots, scale appropriately and revert the depression transform
		Float root[3] = {
			cubic_root[0],
			MAYBE_STD(fma)(-Float(0.5), cubic_root[0], - 0.5f * MAYBE_STD(sqrt)(3.0f) * cubic_root[1]),
			MAYBE_STD(fma)(-Float(0.5), cubic_root[0], + 0.5f * MAYBE_STD(sqrt)(3.0f) * cubic_root[1])
		};

		roots[0] = MAYBE_STD(fma)(Float(2) * MAYBE_STD(sqrt)(-depressed[1]), root[0], - z);
		roots[1] = MAYBE_STD(fma)(Float(2) * MAYBE_STD(sqrt)(-depressed[1]), root[1], - z);
		roots[2] = MAYBE_STD(fma)(Float(2) * MAYBE_STD(sqrt)(-depressed[1]), root[2], - z);

		return RootFindingResult::Success;
	}
	else
	{
#define CALL_AND_SET_RESULT_IF_ACTIVE(call) \
do{ \
RootFindingResult result_current = call;\
if (result == RootFindingResult::Success)\
	result = result_current; \
}while(0)

		Float derivative[Degree + 1] = { 0 };
		//Float upper = upper_bound_polynomial_roots<Float, Degree>(coeffs);
		Float upper = INFINITY;

		// Manually unroll the recursive call (compiles to no-op if the passed degree is >= Degree)
		// This reduces stack frame memory to 0 in CUDA.
		RootFindingResult result = find_real_ndegree_polynomial_roots<Float,  2, Degree>(coeffs, -upper, upper, derivative, roots, tolerance, max_iterations);
#if DM_UNROLL_NEWTON_CALLS
		static_assert(Degree <= 10);
		CALL_AND_SET_RESULT_IF_ACTIVE( (find_real_ndegree_polynomial_roots<Float,3,Degree>(coeffs, -upper, upper, derivative, roots, tolerance, max_iterations)) );
		CALL_AND_SET_RESULT_IF_ACTIVE( (find_real_ndegree_polynomial_roots<Float,4,Degree>(coeffs, -upper, upper, derivative, roots, tolerance, max_iterations)) );
		CALL_AND_SET_RESULT_IF_ACTIVE( (find_real_ndegree_polynomial_roots<Float,5,Degree>(coeffs, -upper, upper, derivative, roots, tolerance, max_iterations)) );
		CALL_AND_SET_RESULT_IF_ACTIVE( (find_real_ndegree_polynomial_roots<Float,6,Degree>(coeffs, -upper, upper, derivative, roots, tolerance, max_iterations)) );
		CALL_AND_SET_RESULT_IF_ACTIVE( (find_real_ndegree_polynomial_roots<Float,7,Degree>(coeffs, -upper, upper, derivative, roots, tolerance, max_iterations)) );
		CALL_AND_SET_RESULT_IF_ACTIVE( (find_real_ndegree_polynomial_roots<Float,8,Degree>(coeffs, -upper, upper, derivative, roots, tolerance, max_iterations)) );
		CALL_AND_SET_RESULT_IF_ACTIVE( (find_real_ndegree_polynomial_roots<Float,9,Degree>(coeffs, -upper, upper, derivative, roots, tolerance, max_iterations)) );
#endif
		return result;
	}
}

template<unsigned int Degree, typename Float>
DEVICE void find_real_polynomial_roots_backward(Float const coeffs[Degree + 1], Float const roots[Degree], Float* dcoeffs, Float const* droots, bool* has_overflow = nullptr)
{
	for (unsigned int root_idx = 0; root_idx < Degree; ++root_idx)
	{
		Float root = roots[root_idx];

		// Precompute droot/dfdx
		// TODO: how reliable is the computation of droot_inv_dfdx if the derivative should remain zero?
		Float droot_inv_dfdx = -droots[root_idx] / evaluate_polynomial_derivative<Float, Degree>(coeffs, root);

		Float contribution = droot_inv_dfdx;
		for (unsigned int coeff_idx = 0; coeff_idx < Degree + 1; ++coeff_idx)
		{
			if (!MAYBE_STD(isfinite(contribution)))
				contribution = Float(0);

			dcoeffs[coeff_idx] += contribution;

			contribution *= root;
		}
	}
}

template<unsigned int Degree, typename Float>
DEVICE void find_real_polynomial_roots_forward(Float const coeffs[Degree + 1], Float const roots[Degree], Float const* dcoeffs, Float* droots)
{
	// TODO: Implement forward derivatives
}

}