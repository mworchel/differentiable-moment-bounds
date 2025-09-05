#pragma once

#if !IS_CUDA
#include <iterator>
#endif

#include "common.hpp"
#include "linalg.hpp"
#include "polynomial.hpp"

namespace dm
{

template<typename Float>
struct MomentBoundParams
{
	Float bias; // Bias [0, 1]
	Float overestimation_weight;
	Float newton_tolerance;
	unsigned int newton_max_iterations = { 100 };
};

enum class MomentBoundResult : uint32_t
{
	Success = 0,
	NotPositive,
	NoRoots,
	Unknown,
	Count
};

DEVICE constexpr uint32_t operator+(MomentBoundResult result)
{
	return static_cast<uint32_t>(result);
}

inline char const* get_result_name(MomentBoundResult result)
{
	constexpr char const* names[] = {
		"Success",
		"NotPositive",
		"NoRoots",
		"Unknown"
	};
#if !IS_CUDA
	static_assert(std::size(names) == +MomentBoundResult::Count, "Incorrect number of result names");
#endif
	return names[+result];
}

inline char const* get_result_string(MomentBoundResult result)
{
	constexpr char const* strings[] = {
		"Moment bounds were computed successfully",
		"The Cholesky decomposition of the Hankel matrix failed because it is not positive definite (the moment sequence is not positive)",
		"The roots of the canonical representation could not be determined",
		"An unknown error occured"
	};
#if !IS_CUDA
	static_assert(std::size(strings) == +MomentBoundResult::Count, "Incorrect number of result strings");
#endif
	return strings[+result];
}

// All bias vectors lie in the affine hyperplane (1, 0, ..., 0)^T x = 1,
// and they are less effective if the moment vector lies outside of it (i.e., if the 0-th moment is != 1).

template<unsigned int N, typename Float>
CONSTANT IF_NOT_CUDA(constexpr) Float moment_bias_vector[] = { 0.f };

template<typename Float>
CONSTANT IF_NOT_CUDA(constexpr) Float moment_bias_vector<1, Float>[3] = { 1.f, 0.f, 1.f };

// For N = 2,...,4 take bias vectors from the supplementary material of
// Münstermann et al. (2018) Moment-Based Order-Independent Transparency
template<typename Float>
CONSTANT IF_NOT_CUDA(constexpr) Float moment_bias_vector<2, Float>[5] = {1, 0, 0.375, 0, 0.375};

template<typename Float>
CONSTANT IF_NOT_CUDA(constexpr) Float moment_bias_vector<3, Float>[7] = { 1, 0, 0.48, 0, 0.451, 0, 0.45 };

template<typename Float>
CONSTANT IF_NOT_CUDA(constexpr) Float moment_bias_vector<4, Float>[9] = { 1, 0, 0.75, 0, 0.676666667, 0, 0.63, 0, 0.60030303 };

template<typename Float>
CONSTANT IF_NOT_CUDA(constexpr) Float moment_bias_vector<5, Float>[11] = { 1.0, 0.0, 0.46666669845581055, 1.4901161193847656e-08, 0.3770667016506195, 0.0, 0.3489066958427429, 0.0, 0.33893293142318726, 0.0, 0.335348904132843 };

template<typename Float>
CONSTANT IF_NOT_CUDA(constexpr) Float moment_bias_vector<6, Float>[13] = { 1.0, -1.4901161193847656e-08, 0.4444444179534912, -1.4901161193847656e-08, 0.34567901492118835, 0.0, 0.3111895024776459, 0.0, 0.2969059646129608, 0.0, 0.2906738519668579, 0.0, 0.2879169285297394 };

template<typename Float>
CONSTANT IF_NOT_CUDA(constexpr) Float moment_bias_vector<7, Float>[15] = { 1.0, 1.4901161193847656e-08, 0.4285714030265808, 0.0, 0.3236151933670044, 0.0, 0.28475379943847656, 0.0, 0.26722463965415955, 0.0, 0.25869518518447876, 7.450580596923828e-09, 0.2544192373752594, 0.0, 0.2522515654563904 };

template<typename Float>
CONSTANT IF_NOT_CUDA(constexpr) Float moment_bias_vector<8, Float>[17] = { 1.0, 0.0, 0.4166666567325592, 7.450580596923828e-09, 0.3072916567325592, 7.450580596923828e-09, 0.2652994394302368, 7.450580596923828e-09, 0.2453409731388092, 0.0, 0.23495356738567352, 0.0, 0.22931568324565887, 0.0, 0.2261953353881836, 0.0, 0.22445286810398102 };

template<typename Float>
CONSTANT IF_NOT_CUDA(constexpr) Float moment_bias_vector<9, Float>[19] = { 1.0000001192092896, -1.4901161193847656e-08, 0.40740740299224854, 7.450580596923828e-09, 0.2947416603565216, 0.0, 0.25043046474456787, 0.0, 0.22862932085990906, 0.0, 0.21676616370677948, 0.0, 0.20997484028339386, 0.0, 0.2059827595949173, 0.0, 0.2036033570766449, 0.0, 0.20217493176460266 };

// Evaluate the moment curve u(t) = [t^0, t^1, t^2, ..., t^n], where n is `Degree`.
template<unsigned int Degree, typename Float>
DEVICE void moment_curve(Float const t, Float* u) // u[Degree + 1]
{
	Float t_product(1);
	for (unsigned int i = 0; i < Degree + 1; ++i, t_product *= t)
	{
		u[i] = t_product;
	}
}

// Reverse-mode derivative of `moment_curve()`.
template<unsigned int Degree, typename Float>
DEVICE void moment_curve_backward(Float const t, Float const* u, Float& dt, Float const* du)
{
	// The function `moment_curve` computes f(t) = [t^0, t^1, t^2, ..., t^n], where n is the degree.
	// Therefore f'(t) = [0, 1*t^0, 2*t^1, ..., n*t^(n-1)] and dt = du^T f'(t),
	// which is simply the derivative of a polynomial with coefficients du, evaluated at t.
	dt = dm::evaluate_polynomial_derivative<Float, Degree>(du, t);
}

/**
 * Compute the coefficients of the (orthogonal) polynomial P_n.
 * The coefficients are scaled by default but the roots remain the same as those of P_n.
 *
 * Details:
 * The last row/column of the inverse Hankel matrix H^{-1} are the coefficients of P_n,
 * and this function computes the scaled version by solving the linear system Hx = LL^Tx = (0, 0, ..., 0, s)
 *
 * If one solves with the right hand side (0, 0, ..., 0, 1), i.e. computes the last row/column of H^{-1}, 
 * then all the entries in x are a sum of terms with 1/L_{N,N}^2 in the denominator. By scaling the
 * right hand side with s = L_{N,N} it becomes (0, 0, ..., 0, L_{N,N}) and the entries
 * in x only have terms with 1/L_{N,N} in the denominator.
 */
template<unsigned int N, typename Float, bool Unscaled = false>
DEVICE void compute_Pn_coefficients(LowerTriangularMatrix<Float const, N + 1> L, Float* x)
{
	Float y[N + 1] = { 0 };
	if constexpr (Unscaled)
	{
		y[N] = Float(1) / L(N, N);
	}
	else
	{
		// The coefficients are scaled in this variant, but the roots remain the same.
		y[N] = Float(1);
	}
	backward_substitution<Float, N + 1>(dm::transpose(L), y, x);
}

template<unsigned int N, typename Float, bool UseExternalRoots = false, bool UseClosedFormSolution = false>
DEVICE MomentBoundResult compute_moment_bound(MomentBoundParams<Float> const& params, Float const* moments, Float const eta, Float* bound, 
											  Float* /*optional*/ L_out = nullptr, Float* /*optional*/ coeffs_out = nullptr,
											  Float* /*optional*/ roots_out = nullptr, Float* /*optional*/ weights_out = nullptr,
											  Float const* roots_in = nullptr)
{
	constexpr unsigned int Degree     = 2 * N;
	constexpr unsigned int NumMoments = Degree + 1;

	// Copy the moments to local memory 
	Float m[NumMoments];
	fixed_copy<NumMoments>(m, moments);

	// Bias the moments: m = (1-alpha)*m + alpha*m^\star (Proposition 6 in the paper)
	dm::lerp<NumMoments, Float>(m, dm::moment_bias_vector<N, Float>, params.bias, m);

	// A closed-form solution exists for N = 1.
	if constexpr (UseClosedFormSolution && (N == 1))
	{
		// `UseExternalRoots` is ignored because it's just too cheap to compute.

		Float m0 = m[0];
		Float m1 = m[1];
		Float m2 = m[2];

		// The root of the polynomial Pn is y = m1/m0;
		Float a = m0 * eta - m1;
		Float b = m1 * eta - m2;
		Float denom = eta * a - b;

		Float w0 = (m0 * m2 - m1 * m1) / denom;
		Float w1 = a * a / denom;

		Float bound_ = params.overestimation_weight * w0;

		// The other root of the canonical representation is
		// x1 = (m1 * eta - m2) / (m0 * eta - m1) = b / a
		// so w1 is added to the bound if 
		//        x1 < eta
		// <=> b / a < eta 
		// <=> b < eta * a
		// TODO: Could use the singularity m1/m0 for the comparison (similar to VSM)
		if (b < eta * a)
			bound_ += w1;

		*bound = bound_;

		if (roots_out)
		{
			roots_out[0] = eta;
			roots_out[1] = b / a;
		}

		if (weights_out)
		{
			weights_out[0] = w0;
			weights_out[1] = w1;
		}

		return MomentBoundResult::Success;
	}

	// Construct the (symmetric, p.d.) Hankel matrix H from moments in local memory
	typename dm::SymmetricMatrix<Float, N + 1>::Storage H_storage;
	dm::SymmetricMatrix<Float, N + 1> H{ .data = H_storage.data };
	SYMMETRIC_MATRIX_LOOP(N + 1,
		H(i, j) = m[i + j];
	);

	// Perform Cholesky decomposition of the Hankel matrix, in place
	static_assert(dm::LowerTriangularMatrix<Float, N + 1>::data_size == dm::SymmetricMatrix<Float, N + 1>::data_size);
	dm::LowerTriangularMatrix<Float, N + 1> L{ .data = H.data };
	if (!dm::cholesky<Float, N + 1>(dm::constant(H), L)) [[unlikely]]
	{
		return MomentBoundResult::NotPositive;
	}

	// Evaluate the reduced moment curve at eta (=t)
	Float u_eta[N + 1]; // [N + 1] ;
	dm::moment_curve<N>(eta, u_eta);

	// Solve H-^1 x_0 to get the polynomial coefficients
	Float coeffs[N + 1] = { 0 };
	dm::cholesky_solve<Float, N + 1>(dm::constant(L), u_eta, coeffs);

	// // Legacy (debug) code that determines if this evaluation might be 
	// // critical by checking if eta is close to a root of P_n
	// Float Pn_coeffs[N + 1];
	// dm::compute_Pn_coefficients<N, Float>(dm::constant(L), Pn_coeffs);
	// Float Pn_roots[N];
	// dm::find_real_polynomial_roots<N, Float>(Pn_coeffs, Pn_roots, params.newton_tolerance, params.newton_max_iterations);
	// bool is_critical_eta = false;
	// for (int i = 0; i < static_cast<int>(N); ++i)
	// {
	// 	is_critical_eta |= MAYBE_STD(abs)(Pn_roots[i] - eta) < 1e-7;
	// }

	Float roots[N + 1] = { 0 };
	if constexpr (UseExternalRoots)
	{
		// TODO: Check if roots != nullptr!
		fixed_copy<N + 1>(roots, roots_in);
	}
	else 
	{
		// Find the roots of the resulting polynomial -> points of the canonical representation
		if (dm::find_real_polynomial_roots<N, Float>(coeffs, &roots[1], params.newton_tolerance, params.newton_max_iterations) == RootFindingResult::Failed)
		{
			return MomentBoundResult::NoRoots;
		}
		roots[0] = eta;
	}

	// TODO: Consider sorting roots/pivoting here and only sort the largest root to the right.
	//       (because here, we know that we will, probably, only have one root that becomes large)

	// Compute the weights for each point of the canonical representation
	Float weights[N + 1] = { 0 };
	dm::bjoerck_pereyra_solve<Float, N + 1>(roots, m, weights);

	// TODO: Clamp? Weights can be (slightly) negative due to numerical imprecision 
	//for (unsigned int i = 0; i < N + 1; ++i)
	//{
	//	weights[i] = MAYBE_STD(max)(weights[i], Float(0));
	//}

	Float bound_(params.overestimation_weight * weights[0]);
	for (unsigned int i = 1; i < N + 1; ++i)
	{
		if (roots[i] < eta)
		{
			bound_ += weights[i];
		}
	}

	if (L_out)
		fixed_copy<L.data_size>(L_out, L.data);

	if (coeffs_out)
		fixed_copy<N + 1>(coeffs_out, coeffs);

	if (roots_out)
		fixed_copy<N + 1>(roots_out, roots);

	if (weights_out)
		fixed_copy<N + 1>(weights_out, weights);

	*bound = bound_;

	return MomentBoundResult::Success;
}

// Fused backward operation for the Vandermonde solve and root finding
template<typename Float, unsigned int N, unsigned int Degree = N>
DEVICE void bjoerck_pereyra_solve_and_roots_backward(Float const c[Degree + 1], Float const roots[Degree + 1], Float const* weights,
													 Float* dc, Float* deta, Float* dm, Float const* dweights)
{
	// Solve the system V^T dm = dw
	// Attention, this `dm` is missing the negative sign.
	bjoerck_pereyra_dual_solve<Float, N + 1>(roots, dweights, dm);

	// Handle eta and then continue with the remaining roots
	*deta -= weights[0] * evaluate_polynomial_derivative<Float, N>(dm, roots[0]);

	for (unsigned int i = 1; i < N + 1; ++i)
	{
		Float xi = roots[i];
		Float wi = weights[i];

		// Early exit if `xi` is infinity, as it does
		// not contribute to the derivative.
		if (!MAYBE_STD(isfinite)(xi))
			continue;

		// Theoretically, as `xi` approaches infinity, `a` attends some limit value:
		// the highest coefficients of `dm` and `c` vanish, so the limit is
		// dm[Degree-1]/c[Degree-1]. But, the closer one comes to the limit, the less
		// accurate are `den` and `num` because very small coefficients are multiplied by 
		// hugh numbers (`xi` to some power). So, `a` can be orders of magnitude off.
		// The following is a heuristic that works for 32-bit precision:
		// Check if the quadratic part of the denominator in Eq.X (=(1/wi)^2) overflows
		// and if so, null the contribution (this is effectively a magnitude check)
		// TODO: Think of a more principled way of handling this situation (e.g. use the known limit value of `a`).
		// NOTE: This is not the only contribution of xi because it's used to compute dm (and scattered to deta)
		Float num = evaluate_polynomial_derivative<Float, Degree>(dm, xi);
		Float den = evaluate_polynomial_derivative<Float, Degree>(c, xi);
		Float a = num / den;

		Float wi_inv_sqr = 1 / (wi * wi);

		if (!MAYBE_STD(isfinite)(wi_inv_sqr))
			continue;

		dc[0] += a * wi;
		dc[1] += a * wi * xi;
		Float wi_xi_k(wi * xi);
		for (unsigned int k = 2; k < Degree + 1; ++k)
		{
			// TODO: wi_xi_k could overflow
			wi_xi_k *= xi;

			dc[k] += a * wi_xi_k;
		}
	}
}

template<typename Float, unsigned int N, unsigned int Degree = N>
DEVICE void direct_solve_backward(dm::LowerTriangularMatrix<Float const, N + 1> L,
	                              Float const c[Degree + 1], Float const roots[Degree + 1], Float const* weights,
	                              Float* dc, Float* deta, Float* dm, Float const* dweights)
{
	//(xH^-1)x'
	Float dw0deta = dm::evaluate_polynomial_derivative<Float, N>(c, roots[0]);
	*deta -= Float(2) * dw0deta * weights[0] * weights[0];

	// Now iterate over the remaining roots
	for (unsigned int i = 1; i < N + 1; ++i)
	{
		Float xi = roots[i];
		Float wi = weights[i];

		if (!MAYBE_STD(isfinite)(xi))
			continue;

		// Evaluate xi^T H^-1 (xi)'
		Float u_xi[N + 1];
		dm::moment_curve<N>(xi, u_xi);
		Float ci[N + 1];
		cholesky_solve<Float, N + 1>(L, u_xi, ci);
		Float xHdx = dm::evaluate_polynomial_derivative<Float, N>(ci, xi);

		// Evaluate eta^T H^-1 (xi)' (derivative of Kernel at xi)
		Float K_deriv = evaluate_polynomial_derivative<Float, Degree>(c, xi);

		Float a = xHdx / K_deriv;
		Float dwidxi = (- Float(2) * a * wi * wi);

		Float contribution = dwidxi;
		for (unsigned int k = 0; k < Degree + 1; ++k)
		{
			if (MAYBE_STD(isfinite)(contribution))
				dc[k] += contribution;
			contribution *= xi;
		}

		// TODO: dm is missing!
	}
}

template<unsigned int N, typename Float>
DEVICE void compute_moment_bound_backward(MomentBoundParams<Float> const& params, Float const* moments, Float const eta, Float const bound, 
										  Float const* L, Float const* coeffs, Float const* roots, Float const* weights,
										  Float* dmoments, Float* deta, Float const dbound)
{
	constexpr unsigned int Degree = 2 * N;
	constexpr unsigned int NumMoments = Degree + 1;

	// Only u_eta is re-evaluated from the primal execution (because it's super simple)
	Float u_eta[N + 1]; // [N + 1] ;
	dm::moment_curve<N>(eta, u_eta);

	Float dweights[N + 1] = { 0 };
	dweights[0] = params.overestimation_weight * dbound;
	for (unsigned int i = 1; i < N + 1; ++i)
	{
		dweights[i] = (roots[i] < eta) * dbound;
	}

	// // Legacy code that backwards through the solver and root finding separately.
	// Float droots[N + 1]  = { 0 };
	// Float dm[NumMoments] = { 0 };
	// dm::bjoerck_pereyra_solve_backward<Float, N + 1>(roots, nullptr, weights, droots, dm, dweights);
	// *deta += droots[0];
	// Float dcoeffs[N + 1] = { 0 };
	// dm::find_real_polynomial_roots_backward<N, Float>(coeffs, &roots[1], dcoeffs, &droots[1], has_overflow);

	Float dcoeffs[N + 1] = { 0 };
	Float dm[NumMoments] = { 0 };
	dm::bjoerck_pereyra_solve_and_roots_backward<Float, N>(coeffs, roots, weights,
														   dcoeffs, deta, dm, dweights);

	// TODO: Clamp magnitude of leading coefficient?
	// dcoeffs[N] = clamp_absmin_backward(coeffs[N], Float(1e-8), dcoeffs[N]);

	dm::LowerTriangularMatrix<Float const, N + 1> L_mat{ .data = L };

	typename dm::SymmetricMatrix<Float, N + 1>::Storage dH_storage;
	dm::SymmetricMatrix<Float, N + 1> dH{ .data = dH_storage.data };

	Float du_eta[N + 1] = { 0 };
	dm::cholesky_solve_backward<Float, N + 1>(dm::constant(L_mat), u_eta, coeffs, dH, du_eta, dcoeffs);

	Float deta_ = 0;
	dm::moment_curve_backward<N>(eta, u_eta, deta_, du_eta);
	*deta += deta_;

	// dH is symmetric, so off-diagonal entries count twice.
	SYMMETRIC_MATRIX_LOOP(N + 1,
		dm[i + j] += (i == j) ? dH(i, j) : Float(2) * dH(i, j);
	);

	for (unsigned int i = 0; i < NumMoments; ++i)
		dmoments[i] = (1 - params.bias) * dm[i];
}

}