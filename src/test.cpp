#include <cassert>
#include <chrono>
#include <iostream>
#include <limits>
#include <numeric>
#include <cstdio>
#include <random>
#include <optional>

#include "linalg.hpp"
#include "matrix.hpp"
#include "moment_problem.hpp"
#include "polynomial.hpp"

// These are basic sanity checks for the functions in the core library.
// TODO: Update, because tests are outdated.

template<typename MatrixType>
void print_matrix(MatrixType mat)
{
	constexpr int N_ = static_cast<int>(MatrixType::N);

	for (int i = 0; i < N_; i++)
	{
		putchar((i == 0) ? '[' : ' ');

		for (int j = 0; j < N_; ++j)
		{
			printf("%f", mat.get_full(i, j));
			
			if (j < N_ - 1)
				putchar(' ');
		}

		printf((i < N_ - 1) ? " \n" : "]\n");
	}
}

template<unsigned int N, typename Float>
void print_vector(Float const* v)
{
	constexpr int N_ = static_cast<int>(N);

	putchar('[');
	for (unsigned int i = 0; i < N_; i++)
	{
		printf("%f", v[i]);

		if (i < (N_ - 1))
		{
			putchar(' ');
		}
	}
	printf("]\n");
}

template<typename Float>
Float relative_error_sym(Float v1, Float v2)
{
	if (std::isinf(v1) || std::isinf(v2))
		return std::isinf(v1) && std::isinf(v2) ? Float(0) : std::numeric_limits<Float>::infinity();

	Float denom = std::max(std::abs(v1), std::abs(v2));
	bool zero_denom = denom < std::numeric_limits<Float>::min();
	return zero_denom ? Float(0) : std::abs(v1 - v2) / denom;
}

template<typename Float>
constexpr Float get_solver_tolerance()
{
	// Compute suitable tolerance from machine precision
	if constexpr (std::is_same_v<Float, double>) // double
		return 1e4 * std::numeric_limits<Float>::epsilon();
	else if constexpr (std::is_same_v<Float, float>) // float
		return 1e2 * std::numeric_limits<Float>::epsilon();
	else
		return std::numeric_limits<Float>::epsilon();
}

template<typename Float, unsigned int Degree, int Flags = dm::RootFindingFlags::None, bool TestSpecialCases = false>
void test_find_real_polynomial_roots(Float tolerance = 1e-12, bool verbose = false);

template<typename Float, unsigned int Degree>
void test_find_real_polynomial_roots_backward(Float finite_difference_eps, Float tolerance = 1e-3, bool verbose = false);

template<typename Float, unsigned int N>
void test_cholesky(Float tolerance = 1e-12, bool verbose = false);

template<typename Float, unsigned int N>
void test_forward_substitution(bool verbose = false);

template<typename Float, unsigned int N>
void test_backward_substitution(bool verbose = false);

template<typename Float, unsigned int N>
void test_cholesky_solve(Float tolerance_factor = Float(1), bool verbose = false);

template<typename Float, unsigned int N>
void test_cholesky_solve_backward(Float tolerance = 5e-5, bool verbose = false);

template<typename Float, unsigned int N, bool TestSpecialCases = false>
void test_bjoerck_pereyra_solve(bool verbose = false);

template<typename Float, unsigned int N, bool TestSpecialCases = false>
void test_bjoerck_pereyra_dual_solve(Float tolerance_factor = Float(1), bool verbose = false);

template<typename Float, unsigned int N>
void test_compute_moment_bound(Float tolerance = 5e-9, bool verbose = false);

template<typename Float, unsigned int N>
void test_moment_curve_backward(Float tolerance = 6e-6, bool verbose = false);

template<typename Float, unsigned int N>
void test_bjoerck_pereyra_solve_backward(Float tolerance=5e-5, bool verbose = false);

template<typename Float, unsigned int N>
void test_compute_moment_bound_backward(Float tolerance = 5e-5, bool verbose = false);

// Switches to toggle different tests
#define TEST_FIND_REAL_POLYNOMIAL_ROOTS
#define TEST_FIND_REAL_POLYNOMIAL_ROOTS_SPECIAL_CASES
#define TEST_FIND_REAL_POLYNOMIAL_ROOTS_BACKWARD
#define TEST_CHOLESKY
#define TEST_FORWARD_SUBSTITUTION
#define TEST_BACKWARD_SUBSTITUTION
#define TEST_CHOLESKY_SOLVE
#define TEST_CHOLESKY_SOLVE_BACKWARD
#define TEST_BJOERCK_PEREYRA_SOLVE
#define TEST_BJOERCK_PEREYRA_SOLVE_SPECIAL_CASES
#define TEST_BJOERCK_PEREYRA_DUAL_SOLVE
#define TEST_BJOERCK_PEREYRA_DUAL_SOLVE_SPECIAL_CASES
#define TEST_COMPUTE_MOMENT_BOUND
#define TEST_MOMENT_CURVE_BACKWARD
#define TEST_BJOERCK_PEREYRA_SOLVE_BACKWARD
#define TEST_COMPUTE_MOMENT_BOUND_BACKWARD

int main()
{
#ifdef TEST_FIND_REAL_POLYNOMIAL_ROOTS
	printf("#############################################\n");
	printf("# find_real_polynomial_roots \n");
	printf("#############################################\n");
	test_find_real_polynomial_roots<float, 1>();
	test_find_real_polynomial_roots<float, 2>();
	test_find_real_polynomial_roots<float, 3, dm::RootFindingFlags::CubicFastPathCP>(3e-5);
	test_find_real_polynomial_roots<float, 3, dm::RootFindingFlags::CubicFastPathMA>(3e-5);
	test_find_real_polynomial_roots<float, 3>();
	test_find_real_polynomial_roots<float, 4>(3e-12);
	test_find_real_polynomial_roots<float, 5>(3e-12);
	test_find_real_polynomial_roots<float, 6>(6e-9);
	test_find_real_polynomial_roots<float, 7>(2e-10);
	test_find_real_polynomial_roots<double, 1>();
	test_find_real_polynomial_roots<double, 2>();
	test_find_real_polynomial_roots<double, 3, dm::RootFindingFlags::CubicFastPathCP>(3e-5);
	test_find_real_polynomial_roots<double, 3, dm::RootFindingFlags::CubicFastPathMA>(3e-5);
	test_find_real_polynomial_roots<double, 3>();
	test_find_real_polynomial_roots<double, 4>(3e-12);
	test_find_real_polynomial_roots<double, 5>(3e-12);
	test_find_real_polynomial_roots<double, 6>(6e-9);
	test_find_real_polynomial_roots<double, 7>(2e-10);
	printf("#############################################\n\n");
#endif

#ifdef TEST_FIND_REAL_POLYNOMIAL_ROOTS_SPECIAL_CASES
	printf("#############################################\n");
	printf("# find_real_polynomial_roots (special cases) \n");
	printf("#############################################\n");
	test_find_real_polynomial_roots<float, 1, dm::RootFindingFlags::None, true>();
	test_find_real_polynomial_roots<float, 2, dm::RootFindingFlags::None, true>(1e-12);
	test_find_real_polynomial_roots<float, 3, dm::RootFindingFlags::None, true>(3e-5, true);
	//test_find_real_polynomial_roots<float, 4, dm::RootFindingFlags::None, true>(3e-5);
	//test_find_real_polynomial_roots<double, 1, dm::RootFindingFlags::None, true>();
	//test_find_real_polynomial_roots<double, 2, dm::RootFindingFlags::None, true>();
	//test_find_real_polynomial_roots<double, 3, dm::RootFindingFlags::CubicFastPathCP, true>(3e-5);
	//test_find_real_polynomial_roots<double, 3, dm::RootFindingFlags::CubicFastPathMA, true>(3e-5);
	//test_find_real_polynomial_roots<double, 3, dm::RootFindingFlags::None, true>();
	//test_find_real_polynomial_roots<double, 4, dm::RootFindingFlags::None, true>();
	printf("#############################################\n\n");
#endif

#ifdef TEST_FIND_REAL_POLYNOMIAL_ROOTS_BACKWARD
	printf("#############################################\n");
	printf("# find_real_polynomial_roots_backward \n");
	printf("#############################################\n");
	test_find_real_polynomial_roots_backward<float, 1>(1e-2);
	test_find_real_polynomial_roots_backward<float, 2>(1e-2);
	test_find_real_polynomial_roots_backward<float, 3>(1e-2);
	test_find_real_polynomial_roots_backward<float, 4>(1e-2);
	test_find_real_polynomial_roots_backward<float, 5>(1e-2);
	test_find_real_polynomial_roots_backward<double, 1>(1e-7);
	test_find_real_polynomial_roots_backward<double, 2>(1e-7);
	test_find_real_polynomial_roots_backward<double, 3>(1e-7);
	test_find_real_polynomial_roots_backward<double, 4>(1e-7);
	test_find_real_polynomial_roots_backward<double, 5>(1e-7);
	printf("#############################################\n\n");
#endif

#ifdef TEST_CHOLESKY
	printf("#############################################\n");
	printf("# cholesky \n");
	printf("#############################################\n");
	test_cholesky<float, 2>(1e-6);
	test_cholesky<float, 3>(2.5e-6);
	test_cholesky<float, 4>(6.5e-6);
	test_cholesky<float, 5>(6e-5);
	test_cholesky<float, 6>(3e-5);
	test_cholesky<float, 7>(7e-5);
	test_cholesky<float, 8>(7e-5);
	test_cholesky<float, 9>(2e-4);
	test_cholesky<double, 2>();
	test_cholesky<double, 3>();
	test_cholesky<double, 4>();
	test_cholesky<double, 5>();
	test_cholesky<double, 6>();
	test_cholesky<double, 7>();
	test_cholesky<double, 8>();
	test_cholesky<double, 9>();
	printf("#############################################\n\n");
#endif

#ifdef TEST_FORWARD_SUBSTITUTION
	printf("#############################################\n");
	printf("# forward_substitution \n");
	printf("#############################################\n");
	test_forward_substitution<float, 2>();
	test_forward_substitution<float, 3>();
	test_forward_substitution<float, 4>();
	test_forward_substitution<float, 5>();
	test_forward_substitution<float, 6>();
	test_forward_substitution<float, 7>();
	test_forward_substitution<float, 8>();
	test_forward_substitution<float, 9>();
	test_forward_substitution<double, 2>();
	test_forward_substitution<double, 3>();
	test_forward_substitution<double, 4>();
	test_forward_substitution<double, 5>();
	test_forward_substitution<double, 6>();
	test_forward_substitution<double, 7>();
	test_forward_substitution<double, 8>();
	test_forward_substitution<double, 9>();
	printf("#############################################\n\n");
#endif

#ifdef TEST_BACKWARD_SUBSTITUTION
	printf("#############################################\n");
	printf("# backward_substitution \n");
	printf("#############################################\n");
	test_backward_substitution<float, 2>();
	test_backward_substitution<float, 3>();
	test_backward_substitution<float, 4>();
	test_backward_substitution<float, 5>();
	test_backward_substitution<float, 6>();
	test_backward_substitution<float, 7>();
	test_backward_substitution<float, 8>();
	test_backward_substitution<float, 9>();
	test_backward_substitution<double, 2>();
	test_backward_substitution<double, 3>();
	test_backward_substitution<double, 4>();
	test_backward_substitution<double, 5>();
	test_backward_substitution<double, 6>();
	test_backward_substitution<double, 7>();
	test_backward_substitution<double, 8>();
	test_backward_substitution<double, 9>();
	printf("#############################################\n\n");
#endif

#ifdef TEST_CHOLESKY_SOLVE
	printf("#############################################\n");
	printf("# cholesky_solve \n");
	printf("#############################################\n");
	test_cholesky_solve<double, 2>();
	test_cholesky_solve<double, 3>(2);
	test_cholesky_solve<double, 4>();
	test_cholesky_solve<double, 5>(30);
	test_cholesky_solve<double, 6>(20);
	test_cholesky_solve<double, 7>(20);
	test_cholesky_solve<double, 8>(4);
	test_cholesky_solve<double, 9>(30);
	printf("#############################################\n\n");
#endif


#ifdef TEST_CHOLESKY_SOLVE_BACKWARD
	printf("#############################################\n");
	printf("# cholesky_solve_backward \n");
	printf("#############################################\n");
	test_cholesky_solve_backward<double, 2>();
	test_cholesky_solve_backward<double, 3>();
	test_cholesky_solve_backward<double, 4>();
	test_cholesky_solve_backward<double, 5>();
	test_cholesky_solve_backward<double, 6>();
	test_cholesky_solve_backward<double, 7>();
	test_cholesky_solve_backward<double, 8>();
	test_cholesky_solve_backward<double, 9>();
	printf("#############################################\n\n");
#endif

#ifdef TEST_BJOERCK_PEREYRA_SOLVE
	printf("#############################################\n");
	printf("# bjoerck_pereyra_solve \n");
	printf("#############################################\n");
	test_bjoerck_pereyra_solve<float, 2>();
	test_bjoerck_pereyra_solve<float, 3>();
	test_bjoerck_pereyra_solve<float, 4>();
	test_bjoerck_pereyra_solve<double, 2>();
	test_bjoerck_pereyra_solve<double, 3>();
	test_bjoerck_pereyra_solve<double, 4>();
	test_bjoerck_pereyra_solve<double, 5>();
	test_bjoerck_pereyra_solve<double, 6>();
	test_bjoerck_pereyra_solve<double, 7>();
	test_bjoerck_pereyra_solve<double, 8>();
	test_bjoerck_pereyra_solve<double, 9>();
	printf("#############################################\n\n");
#endif

#ifdef TEST_BJOERCK_PEREYRA_SOLVE_SPECIAL_CASES
	printf("#############################################\n");
	printf("# bjoerck_pereyra_solve (special cases) \n");
	printf("#############################################\n");
	//test_bjoerck_pereyra_solve<float, 2, true>();
	test_bjoerck_pereyra_solve<float, 3, true>();
	//test_bjoerck_pereyra_solve<float, 4, true>();
	//test_bjoerck_pereyra_solve<double, 2, true>();
	//test_bjoerck_pereyra_solve<double, 3, true>();
	//test_bjoerck_pereyra_solve<double, 4, true>();
	//test_bjoerck_pereyra_solve<double, 5, true>();
	//test_bjoerck_pereyra_solve<double, 6, true>();
	//test_bjoerck_pereyra_solve<double, 7, true>();
	//test_bjoerck_pereyra_solve<double, 8, true>();
	//test_bjoerck_pereyra_solve<double, 9, true>();
	printf("#############################################\n\n");
#endif

#ifdef TEST_BJOERCK_PEREYRA_DUAL_SOLVE
	printf("#############################################\n");
	printf("# bjoerck_pereyra_dual_solve \n");
	printf("#############################################\n");
	test_bjoerck_pereyra_dual_solve<float, 2>();
	printf("#############################################\n\n");
#endif

#ifdef TEST_BJOERCK_PEREYRA_DUAL_SOLVE_SPECIAL_CASES
	printf("#############################################\n");
	printf("# bjoerck_pereyra_dual_solve (special cases) \n");
	printf("#############################################\n");
	test_bjoerck_pereyra_dual_solve<float, 2, true>();
	printf("#############################################\n\n");
#endif

#ifdef TEST_COMPUTE_MOMENT_BOUND
	printf("#############################################\n");
	printf("# compute_moment_bound \n");
	printf("#############################################\n");
	test_compute_moment_bound<float, 1>(1e-3);
	test_compute_moment_bound<float, 2>(1e-3);
	test_compute_moment_bound<float, 3>(1e-3);
	test_compute_moment_bound<float, 4>(1e-3);
	test_compute_moment_bound<float, 5>(1e-3);
	test_compute_moment_bound<float, 6>(1e-3);
	test_compute_moment_bound<double, 1>();
	test_compute_moment_bound<double, 2>();
	test_compute_moment_bound<double, 3>();
	test_compute_moment_bound<double, 4>();
	test_compute_moment_bound<double, 5>(6e-6);
	test_compute_moment_bound<double, 6>(3e-8);
	printf("#############################################\n\n");
#endif

#ifdef TEST_MOMENT_CURVE_BACKWARD
	printf("#############################################\n");
	printf("# moment_curve_backward \n");
	printf("#############################################\n");
	test_moment_curve_backward<double, 1>();
	test_moment_curve_backward<double, 2>();
	test_moment_curve_backward<double, 3>();
	test_moment_curve_backward<double, 4>();
	test_moment_curve_backward<double, 5>();
	test_moment_curve_backward<double, 6>();
	test_moment_curve_backward<double, 7>();
	test_moment_curve_backward<double, 8>();
	test_moment_curve_backward<double, 9>();
	printf("#############################################\n\n");
#endif

#ifdef TEST_BJOERCK_PEREYRA_SOLVE_BACKWARD
	printf("#############################################\n");
	printf("# bjoerck_pereyra_solve_backward \n");
	printf("#############################################\n");
	test_bjoerck_pereyra_solve_backward<double, 2>();
	test_bjoerck_pereyra_solve_backward<double, 3>();
	test_bjoerck_pereyra_solve_backward<double, 4>();
	test_bjoerck_pereyra_solve_backward<double, 5>();
	test_bjoerck_pereyra_solve_backward<double, 6>(9e-4);
	test_bjoerck_pereyra_solve_backward<double, 7>();
	test_bjoerck_pereyra_solve_backward<double, 8>(2e-4);
	test_bjoerck_pereyra_solve_backward<double, 9>(4e-3);
	printf("#############################################\n\n");
#endif

#ifdef TEST_COMPUTE_MOMENT_BOUND_BACKWARD
	printf("#############################################\n");
	printf("# compute_moment_bound_backward \n");
	printf("#############################################\n");
	test_compute_moment_bound_backward<double, 1>();
	test_compute_moment_bound_backward<double, 2>(5e-4);
	test_compute_moment_bound_backward<double, 3>(3e-4);
	test_compute_moment_bound_backward<double, 4>(3e-3);
	test_compute_moment_bound_backward<double, 5>(3e-3);
	//test_compute_moment_bound_backward<double, 6>(3e-8); // No bias vectors...
	//test_compute_moment_bound_backward<double, 7>(3e-8); // 
	//test_compute_moment_bound_backward<double, 8>(3e-8); // 
	//test_compute_moment_bound_backward<double, 9>(3e-8); // 
	printf("#############################################\n\n");
#endif
}

template<typename Float, unsigned int Degree, int Flags, bool TestSpecialCases>
void test_find_real_polynomial_roots(Float tolerance, bool verbose)
{
	constexpr unsigned int num_polynomials = 100000;

	constexpr Float root_finding_tolerance             = std::numeric_limits<Float>::epsilon();
	constexpr unsigned int root_finding_max_iterations = 100u;

	std::default_random_engine engine(Degree);
	std::normal_distribution<Float> distribution(Float(0), Float(10));
	std::uniform_real<Float> unit_distribution;

	Float error_max_mean(0);
	std::optional<Float> error_max_all;
	std::chrono::nanoseconds time(0);
	unsigned int num_no_roots = 0;
	unsigned int num_failed = 0;
	for (unsigned int idx = 0; idx < num_polynomials; ++idx)
	{
		// Generate random roots
		Float roots_ref[Degree];
		for (Float& r : roots_ref)
		{
			r = distribution(engine);
		}

		// Flip a coin if one of the roots is shifted (close) to +-inf or set to exactly 0
		if constexpr (TestSpecialCases)
		{
			Float p = unit_distribution(engine);
			if (p < Float(0.5))
			{
				roots_ref[0] = Float(0);
			}
			else
			{
				auto it = std::max_element(std::begin(roots_ref), std::end(roots_ref), [](Float l, Float r) { return abs(l) < abs(r); });
				if (p < 0.75)
					*it *= Float(100000000); // large
				else
					*it = std::copysign(std::numeric_limits<Float>::infinity(), *it); // infinity
			}
		}

		// Expand polynomial to get monomial coefficients
		Float coeffs[Degree + 1];
		dm::expand_polynomial<Float, Degree>(roots_ref, coeffs);

		// Find roots
		Float roots[Degree] = { 0 };
		auto time_start = std::chrono::high_resolution_clock::now();
		dm::RootFindingResult result = dm::find_real_polynomial_roots<Degree, Float, Flags>(coeffs, roots, root_finding_tolerance, root_finding_max_iterations);
		time += std::chrono::high_resolution_clock::now() - time_start;

		if (result != dm::RootFindingResult::Success)
		{
			++num_no_roots;
			continue;
		}

		// Roots at +-inf are equal
		for (Float& r : roots)
			if (std::isinf(r))
				r = std::abs(r);
		for (Float& r : roots_ref)
			if (std::isinf(r))
				r = std::abs(r);

		std::sort(std::begin(roots), std::end(roots));
		std::sort(std::begin(roots_ref), std::end(roots_ref));

		// Compute the maximum (relative) error for each root
		std::optional<Float> error_max;
		for (unsigned int i = 0; i < Degree; ++i)
		{
			Float error = relative_error_sym(roots[i], roots_ref[i]);
			if (!error_max || error > *error_max)
				error_max = error;
		}

		// Record the *global* maximum error across all polynomials
		if (!error_max_all || *error_max > *error_max_all)
			error_max_all = *error_max;

		error_max_mean += *error_max;

		if (*error_max > tolerance)
		{
			if (verbose)
			{
				printf("test_find_real_polynomial_roots<%s,%d>(): error for polynomial %d exceeds tolerance (%0.12f)\n", typeid(Float).name(), Degree, idx, *error_max);
				//printf("polynomial:\n");
				//print_vector<Degree + 1>(coeffs);
				//printf("Roots (ref):");
				//print_vector<Degree>(roots_ref);
				//printf("Roots (out):");
				//print_vector<Degree>(roots);
			}
			num_failed++;
		}
	}

	unsigned int num_polynomials_valid = num_polynomials - num_no_roots;

	if (num_polynomials_valid > 0)
		error_max_mean /= num_polynomials_valid;
		time /= num_polynomials;
	char const* status = num_failed > 0 ? "FAILED" : "passed";
	printf("test_find_real_polynomial_roots<%s,%d,%d, %d>(): %s (%06d/%06d polynomials, avg. time=%lld ns, mean error=%e, max error=%e, tolerance=%e)\n", typeid(Float).name(), Degree, Flags, TestSpecialCases, status, num_polynomials_valid - num_failed, num_polynomials_valid, time.count(), error_max_mean, *error_max_all, tolerance);
}

template<typename Float, unsigned int Degree>
void test_find_real_polynomial_roots_backward(Float fd_eps, Float tolerance, bool verbose)
{
	constexpr unsigned int num_polynomials = 100000;

	constexpr Float root_finding_tolerance             = std::numeric_limits<Float>::epsilon();
	constexpr unsigned int root_finding_max_iterations = 100u;

	std::default_random_engine engine(Degree);
	std::normal_distribution<Float> distribution(Float(0), Float(10));

	Float error_max_mean(0);
	std::optional<Float> error_max_all;
	std::chrono::nanoseconds time(0);
	unsigned int num_no_roots = 0;
	unsigned int num_no_fd_roots = 0;
	unsigned int num_failed = 0;
	for (unsigned int idx = 0; idx < num_polynomials; ++idx)
	{
		// Generate random roots
		Float roots_ref[Degree];
		for (Float& r : roots_ref)
		{
			r = distribution(engine);
		}

		// Expand polynomial to get monomial coefficients
		Float coeffs[Degree + 1];
		dm::expand_polynomial<Float, Degree>(roots_ref, coeffs);

		// Find roots
		Float roots[Degree] = { 0 };
		dm::RootFindingResult result = dm::find_real_polynomial_roots<Degree, Float>(coeffs, roots, root_finding_tolerance, root_finding_max_iterations);

		if (result != dm::RootFindingResult::Success)
		{
			++num_no_roots;
			continue;
		}

		// Test sensitivity for random directional derivative
		Float coeffs_dir[Degree + 1] = { 0 };
		for (Float& c : coeffs_dir)
		{
			c = distribution(engine);
		}

		Float coeffs_dir_sum = dm::abssum<Degree + 1>(coeffs_dir);
		for (Float& c : coeffs_dir)
		{
			c /= coeffs_dir_sum;
		}

		// Compute directional derivative using finite differences (reference)
		Float droots_fd[Degree] = { 0 };
		Float coeffs_adv[Degree + 1] = { 0 };
		Float coeffs_ret[Degree + 1] = { 0 };

		Float roots_adv[Degree] = { 0 };
		dm::fma<Degree + 1>(fd_eps, coeffs_dir, coeffs, coeffs_adv); // coeffs_adv = coeffs + fd_eps * coeffs_dir
		result = dm::find_real_polynomial_roots<Degree, Float>(coeffs_adv, roots_adv, root_finding_tolerance, root_finding_max_iterations);

		if (result != dm::RootFindingResult::Success)
		{
			++num_no_fd_roots;
			continue;
		}

		Float roots_ret[Degree] = { 0 };
		dm::fma<Degree + 1>(-fd_eps, coeffs_dir, coeffs, coeffs_ret); // coeffs_ret = coeffs - fd_eps * coeffs_dir
		result = dm::find_real_polynomial_roots<Degree, Float>(coeffs_ret, roots_ret, root_finding_tolerance, root_finding_max_iterations);

		if (result != dm::RootFindingResult::Success)
		{
			++num_no_fd_roots;
			continue;
		}

		dm::mul<Degree>(roots_ret, -1 / (2 * fd_eps), droots_fd);
		dm::fma<Degree>(1 / (2 * fd_eps), roots_adv, droots_fd, droots_fd);

		// Differentiate w.r.t. *one* root and compare to finite differences
		std::chrono::nanoseconds time_local(0);
		std::optional<Float> error_max;
		Float droots_bwd[Degree] = { 0 };
		for (unsigned int k = 0; k < Degree; ++k)
		{
			Float droots[Degree] = { 0 };
			droots[k] = Float(1);

			auto time_start = std::chrono::high_resolution_clock::now();
			Float dcoeffs[Degree + 1] = { 0 };
			dm::find_real_polynomial_roots_backward<Degree, Float>(coeffs, roots, dcoeffs, droots);
			time_local += std::chrono::high_resolution_clock::now() - time_start;
			
			// Compute directional derivative
			droots_bwd[k] = dm::dot<Degree+1, Float>(dcoeffs, coeffs_dir);

			Float error = relative_error_sym(droots_bwd[k], droots_fd[k]);
			if (!error_max || *error_max < error)
				error_max = error;
		}

		time += time_local / Degree;

		// Record the *global* maximum error across all polynomials
		if (!error_max_all || *error_max > *error_max_all)
			error_max_all = *error_max;

		error_max_mean += *error_max;

		if (*error_max > tolerance)
		{
			if (verbose)
			{
				printf("test_polynomial_root_finding_backward<%s,%d>(): error for polynomial %d exceeds tolerance (%0.12f)\n", typeid(Float).name(), Degree, idx, *error_max);
				printf("-- polynomial:\n");
				print_vector<Degree + 1>(coeffs);
				printf("-- roots (ref):");
				print_vector<Degree>(roots_ref);
				printf("-- roots (out):");
				print_vector<Degree>(roots);
				printf("-- direction (coeffs_dir):\n");
				print_vector<Degree + 1>(coeffs_dir);
				printf("-- directional Derivative (ref):\n");
				print_vector<Degree>(droots_fd);
				printf("-- Directional Derivative (bwd):\n");
				print_vector<Degree>(droots_bwd);
			}
			num_failed++;
		}
	}

	unsigned int num_polynomials_valid = num_polynomials - num_no_roots - num_no_fd_roots;
	
	error_max_mean /= num_polynomials_valid;
	time /= num_polynomials_valid; // <- here, we only perform backward for *valid* polynomials

	char const* status = num_failed > 0 ? "FAILED" : "passed";
	printf("test_polynomial_root_finding_backward<%s,%d>(): %s (%06d/%06d polynomials passed, avg. time=%lld ns, mean error=%e, max. error=%e, tolerance=%e)\n", typeid(Float).name(), Degree, status, num_polynomials_valid - num_failed, num_polynomials_valid, time.count(), error_max_mean, *error_max_all, tolerance);
}

template<typename Float, unsigned int N>
void test_cholesky(Float tolerance, bool verbose)
{
	constexpr unsigned int num_matrices = 100000;

	std::default_random_engine engine(N);
	std::normal_distribution<Float> distribution(0, 32);

	Float error_max_mean(0);
	std::optional<Float> error_max_all;
	std::chrono::nanoseconds time(0);
	unsigned int num_failed = 0;
	unsigned int num_no_decomposition = 0;
	for (unsigned int idx = 0; idx < num_matrices; ++idx)
	{
		// Generate a random lower triangular matrix (reference decomposition)
		typename dm::LowerTriangularMatrix<Float, N>::Storage L_storage[2] = { 0 };
		for (Float& m : L_storage[0].data)
		{
			m = distribution(engine);
		}

		// ...make sure L has positive Eigenvalues (diagonal > 0)
		dm::LowerTriangularMatrix<Float, N> L_ref_mut{ .data = L_storage[0].data };
		for (unsigned int i = 0; i < N; ++i)
		{
			L_ref_mut(i, i) = std::abs(L_ref_mut(i, i));
		}

		dm::LowerTriangularMatrix<Float const, N> L_ref{ .data = L_storage[0].data };

		// Generate p.d. symmetric matrix from L
		typename dm::SymmetricMatrix<Float, N>::Storage A_ref_storage = { 0 };
		dm::SymmetricMatrix<Float, N> A_ref{ .data = A_ref_storage.data };

		dm::mul(L_ref, dm::transpose(L_ref), A_ref);

		// Perform Cholesky decomposition
		dm::LowerTriangularMatrix<Float, N> L{ .data = L_storage[1].data };
		auto time_start = std::chrono::high_resolution_clock::now();
		bool success = dm::cholesky<Float, N>(dm::constant(A_ref), L);
		time += std::chrono::high_resolution_clock::now() - time_start;
		if (!success)
		{
			if (verbose)
				printf("test_cholesky<%s,%d>(): matrix %d is not positive definite (Cholesky failed)\n", typeid(Float).name(), N, idx);
			num_no_decomposition++;
			continue;
		}

		// Reconstruct A from the factors
		typename dm::SymmetricMatrix<Float, N>::Storage A_storage = { 0 };
		dm::SymmetricMatrix<Float, N> A{ .data = A_storage.data };

		dm::mul(dm::constant(L), dm::constant(dm::transpose(L)), A);

		// Compute maximum (relative) error for each matrix component
		std::optional<Float> error_max;
		for (unsigned int i = 0; i < N; ++i)
		{
			for (unsigned int j = 0; j <= i; ++j)
			{
				Float error = relative_error_sym(A(i, j), A_ref(i, j));
				if (!error_max || *error_max < error)
					error_max = error;
			}
		}

		// Record the *global* maximum error across all polynomials
		if (!error_max_all || *error_max > *error_max_all)
			error_max_all = *error_max;

		error_max_mean += *error_max;

		if (*error_max > tolerance)
		{
			if (verbose)
			{
				printf("test_cholesky<%s,%d>(): error for matrix %d exceeds tolerance (%0.12f)\n", typeid(Float).name(), N, idx, *error_max);
				printf("L_ref:\n");
				print_matrix(L_ref);
				printf("A_ref:\n");
				print_matrix(A_ref);
				printf("L:\n");
				print_matrix(L);
			}
			num_failed++;
		}
	}

	error_max_mean /= (num_matrices - num_no_decomposition);
	time /= num_matrices;
	char const* status = num_failed > 0 ? "FAILED" : "passed";
	printf("test_cholesky<%s,%d>(): %s (%06d/%06d matrices passed, no decomp=%d, over thresh.=%d, avg. time=%lld ns, mean error=%e, max. error=%e, tolerance=%e)\n", typeid(Float).name(), N, status, num_matrices - num_failed - num_no_decomposition, num_matrices, num_no_decomposition, num_failed, time.count(), error_max_mean, *error_max_all, tolerance);
}

template<typename Float, unsigned int N>
void test_forward_substitution(bool verbose)
{
	constexpr unsigned int num_matrices = 100;

	Float tolerance = get_solver_tolerance<Float>();

	std::default_random_engine engine(0);
	std::uniform_real_distribution<Float> distribution(-5, 5);

	std::optional<Float> error_max_all;
	unsigned int num_failed = 0;
	for (unsigned int idx = 0; idx < num_matrices; ++idx)
	{
		// Generate a random lower triangular matrix
		Float matrix_storage[dm::LowerTriangularMatrix<Float const, N>::data_size] = { 0.f };
		for (Float& m : matrix_storage)
		{
			m = distribution(engine);
		}
		dm::LowerTriangularMatrix<Float const, N> A{ .data = matrix_storage };

		// Generate a the solution vector
		Float solution[N] = {0};
		for (Float& s : solution)
		{
			s = distribution(engine);
		}

		// Generate the target vector `b`
		Float b[N] = {0};
		dm::mul(dm::constant(A), solution, b);

		// Solve system using forward substitution
		Float x[N] = {0};
		dm::forward_substitution(A, b, x);

		Float b_solved[N] = { 0 };
		dm::mul(dm::constant(A), x, b_solved);

		// Compute the maximum (relative) error for each component
		std::optional<Float> error_max;
		for (unsigned int i = 0; i < N; ++i)
		{
			Float error = relative_error_sym(b_solved[i], b[i]);
			if (!error_max || error > *error_max)
				error_max = error;
		}

		// Record the *global* maximum error across all polynomials
		if (!error_max_all || *error_max > *error_max_all)
			error_max_all = *error_max;

		if (*error_max > tolerance)
		{
			printf("test_forward_substitution<%s,%d>(): error for matrix %d exceeds tolerance (%0.12f)\n", typeid(Float).name(), N, idx, *error_max);
			if (verbose)
			{
				printf("Matrix:\n");
				print_matrix(A);
				printf("Solution (ref):");
				print_vector<N>(solution);
				printf("Solution (out):");
				print_vector<N>(x);
			}
			num_failed++;
		}
	}
	
	char const* status = num_failed > 0 ? "FAILED" : "passed";
	printf("test_forward_substitution<%s,%d>(): %s (%06d/%06d matrices passed, max. error=%e, tolerance=%e)\n", typeid(Float).name(), N, status, num_matrices - num_failed, num_matrices, *error_max_all, tolerance);
}

template<typename Float, unsigned int N>
void test_backward_substitution(bool verbose)
{
	constexpr unsigned int num_matrices = 100;

	Float tolerance = get_solver_tolerance<Float>();

	std::default_random_engine engine(0);
	std::uniform_real_distribution<Float> distribution(-5, 5);

	std::optional<Float> error_max_all;
	unsigned int num_failed = 0;
	for (unsigned int idx = 0; idx < num_matrices; ++idx)
	{
		// Generate a random lower triangular matrix
		Float matrix_storage[dm::UpperTriangularMatrix<Float const, N>::data_size] = { 0.f };
		for (Float& m : matrix_storage)
		{
			m = distribution(engine);
		}
		dm::UpperTriangularMatrix<Float const, N> A{ .data = matrix_storage };

		// Generate a the solution vector
		Float solution[N] = { 0 };
		for (Float& s : solution)
		{
			s = distribution(engine);
		}

		// Generate the target vector `b`
		Float b[N] = { 0 };
		dm::mul(A, solution, b);

		// Solve system using backward substitution
		Float x[N] = {};
		dm::backward_substitution(A, b, x);

		Float b_solved[N] = { 0 };
		dm::mul(dm::constant(A), x, b_solved);

		// Compute the maximum (relative) error for each component
		std::optional<Float> error_max;
		for (unsigned int i = 0; i < N; ++i)
		{
			Float error = relative_error_sym(b_solved[i], b[i]);
			if (!error_max || error > *error_max)
				error_max = error;
		}

		// Record the *global* maximum error across all polynomials
		if (!error_max_all || *error_max > *error_max_all)
			error_max_all = *error_max;

		if (*error_max > tolerance)
		{
			printf("test_backward_substitution<%s,%d>(): error for matrix %d exceeds tolerance (%0.12f)\n", typeid(Float).name(), N, idx, *error_max);
			if (verbose)
			{
				printf("Matrix:\n");
				print_matrix(A);
				printf("Solution (ref):");
				print_vector<N>(solution);
				printf("Solution (out):");
				print_vector<N>(x);
			}
			num_failed++;
		}
	}

	char const* status = num_failed > 0 ? "FAILED" : "passed";
	printf("test_backward_substitution<%s,%d>(): %s (%06d/%06d matrices passed, max. error=%e, tolerance=%e)\n", typeid(Float).name(), N, status, num_matrices - num_failed, num_matrices, *error_max_all, tolerance);
}

template<typename Float, unsigned int N>
void test_cholesky_solve(Float tolerance_factor, bool verbose)
{
	constexpr unsigned int num_matrices = 100000;

	Float tolerance = tolerance_factor * get_solver_tolerance<Float>();

	std::default_random_engine engine(N);
	std::normal_distribution<Float> distribution(0, 32);

	std::optional<Float> error_max_all;
	unsigned int num_failed = 0;
	for (unsigned int idx = 0; idx < num_matrices; ++idx)
	{
		// Generate a random lower triangular matrix
		Float matrix_storage[dm::LowerTriangularMatrix<Float const, N>::data_size] = { 0.f };
		for (Float& m : matrix_storage)
		{
			m = distribution(engine);
		}

		// ...make sure L has positive Eigenvalues (diagonal > 0)
		dm::LowerTriangularMatrix<Float, N> L_ref_mut{ .data = matrix_storage };
		for (unsigned int i = 0; i < N; ++i)
		{
			L_ref_mut(i, i) = std::abs(L_ref_mut(i, i));
		}

		dm::LowerTriangularMatrix<Float const, N> L{ .data = matrix_storage };

		// Generate p.d. symmetric matrix from L
		typename dm::SymmetricMatrix<Float, N>::Storage A_storage = { 0 };
		dm::SymmetricMatrix<Float, N> A{ .data = A_storage.data };

		dm::mul(L, dm::transpose(L), A);

		// Generate a the reference solution vector
		Float x_ref[N] = { 0 };
		for (Float& s : x_ref)
		{
			s = distribution(engine);
		}

		// Generate the target vector b = LL^T*x_ref = A*x_ref
		Float tmp[N] = { 0 };
		Float b[N] = { 0 };
		dm::mul(dm::transpose(dm::constant(L)), x_ref, tmp);
		dm::mul(dm::constant(L), tmp, b);

		Float x[N] = { 0 };
		dm::cholesky_solve(L, b, x);

		// Generate the our target vector from our solution b = LL^Tx = Ax
		Float b_solved[N] = { 0 };
		dm::mul(dm::transpose(dm::constant(L)), x, tmp);
		dm::mul(dm::constant(L), tmp, b_solved);

		// Compute the maximum (relative) error for each component
		std::optional<Float> error_max;
		for (unsigned int i = 0; i < N; ++i)
		{
			Float error = relative_error_sym(b_solved[i], b[i]);
			if (!error_max || error > *error_max)
				error_max = error;
		}

		// Record the *global* maximum error across all polynomials
		if (!error_max_all || *error_max > *error_max_all)
			error_max_all = *error_max;

		if (*error_max > tolerance)
		{
			if (verbose)
			{
				printf("test_cholesky_solve<%s,%d>(): error for matrix %d exceeds tolerance (%0.12f)\n", typeid(Float).name(), N, idx, *error_max);
				printf("Matrix (det(L): %0.6f):\n", dm::det(L));
				print_matrix(L);
				printf("b:");
				print_vector<N>(b);
				printf("Solution (ref):");
				print_vector<N>(x_ref);
				printf("Solution (out):");
				print_vector<N>(x);
			}
			num_failed++;
		}
	}

	char const* status = num_failed > 0 ? "FAILED" : "passed";
	printf("test_cholesky_solve<%s,%d>(): %s (%06d/%06d matrices passed, max. error=%e, tolerance=%e)\n", typeid(Float).name(), N, status, num_matrices - num_failed, num_matrices, *error_max_all,  tolerance);
}

template<typename Float, unsigned int N>
void test_cholesky_solve_backward(Float tolerance, bool verbose)
{
	constexpr unsigned int num_matrices = 100000;

	std::default_random_engine engine(N);
	std::normal_distribution<Float> distribution(0, 32);

	std::optional<Float> error_max_all;
	unsigned int num_no_decomposition = 0;
	unsigned int num_failed = 0;
	for (unsigned int idx = 0; idx < num_matrices; ++idx)
	{
		// Generate a random lower triangular matrix
		Float matrix_storage[dm::LowerTriangularMatrix<Float const, N>::data_size] = { 0.f };
		for (Float& m : matrix_storage)
		{
			m = distribution(engine);
		}

		// ...make sure L has positive Eigenvalues (diagonal > 0)
		dm::LowerTriangularMatrix<Float, N> L_ref_mut{ .data = matrix_storage };
		for (unsigned int i = 0; i < N; ++i)
		{
			L_ref_mut(i, i) = std::abs(L_ref_mut(i, i));
		}

		dm::LowerTriangularMatrix<Float const, N> L{ .data = matrix_storage};

		// Generate p.d. symmetric matrix from L
		typename dm::SymmetricMatrix<Float, N>::Storage A_storage = { 0 };
		dm::SymmetricMatrix<Float, N> A{ .data = A_storage.data };

		dm::mul(L, dm::transpose(L), A);

		// Generate a the reference solution vector
		Float x_ref[N] = { 0 };
		for (Float& s : x_ref)
		{
			s = distribution(engine);
		}

		// Generate the target vector b = LL^T*x_ref = A*x_ref
		Float tmp[N] = { 0 };
		Float b[N] = { 0 };
		dm::mul(dm::transpose(dm::constant(L)), x_ref, tmp);
		dm::mul(dm::constant(L), tmp, b);

		Float x[N] = { 0 };
		dm::cholesky_solve(L, b, x);

		typename dm::SymmetricMatrix<Float, N>::Storage A_storage_grad[4] = { 0.f };
		typename dm::LowerTriangularMatrix<Float, N>::Storage L_storage_grad[2] = { 0.f };

		// Test sensitivity for random directional derivative 
		
		// ... for the input matrix A
		dm::SymmetricMatrix<Float, N> A_dir{ .data = A_storage_grad[0].data };
		for (Float& s : A_storage_grad[0].data)
		{
			s = distribution(engine);
		}

		Float A_dir_sum = dm::abssum(dm::constant(A_dir));
		for (Float& s : A_storage_grad[0].data)
		{
			s /= A_dir_sum;
		}

		// ... and the right hand size b
		Float b_dir[N] = { 0 };
		for (Float& s : b_dir)
			s = distribution(engine);
		dm::mul<N>(b_dir, Float(1) / dm::abssum<N>(b_dir), b_dir);

		// Finite differences, a bit cumbersome (this is not a tensor framework)
		Float eps(0.000001);
		dm::SymmetricMatrix<Float, N> A_adv{ .data = A_storage_grad[1].data};
		dm::add(dm::constant(A_dir), dm::constant(A_adv), A_adv); // A_adv = A + eps*A_dir
		dm::mul(dm::constant(A_adv), eps, A_adv);                 // 
		dm::add(dm::constant(A), dm::constant(A_adv), A_adv);     //

		dm::SymmetricMatrix<Float, N> A_ret{ .data = A_storage_grad[2].data };
		dm::add(dm::constant(A_dir), dm::constant(A_ret), A_ret); // A_ret = A - eps*A_dir
		dm::mul(dm::constant(A_ret), -eps, A_ret);                // 
		dm::add(dm::constant(A), dm::constant(A_ret), A_ret);     //

		dm::LowerTriangularMatrix<Float, N> L_adv{ .data = L_storage_grad[0].data };
		bool cholesky_success = dm::cholesky<Float, N>(dm::constant(A_adv), L_adv);
		if (!cholesky_success)
		{
			if (verbose)
				printf("test_cholesky_solve_backward<%s,%d>(): matrix %d fd forward step matrix is not positive definite\n", typeid(Float).name(), N, idx);
			++num_no_decomposition;
			continue;
		}
		Float x_adv[N] = { 0 };
		dm::cholesky_solve(dm::constant(L_adv), b, x_adv);

		dm::LowerTriangularMatrix<Float, N> L_ret{ .data = L_storage_grad[1].data };
		cholesky_success = dm::cholesky<Float, N>(dm::constant(A_ret), L_ret);
		if (!cholesky_success)
		{
			if (verbose)
				printf("test_cholesky_solve_backward<%s,%d>(): matrix %d fd backward step matrix is not positive definite\n", typeid(Float).name(), N, idx);
			++num_no_decomposition;
			continue;
		}
		Float x_ret[N] = { 0 };
		dm::cholesky_solve(dm::constant(L_ret), b, x_ret);

		Float dx_A_fd[N] = { 0 };
		dm::sub<N>(x_adv, x_ret, dx_A_fd);
		dm::mul<N>(dx_A_fd, Float(1. / (2. * eps)), dx_A_fd);

		Float b_adv[N] = { 0 };
		Float b_ret[N] = { 0 };
		dm::fma<N>(+eps, b_dir, b, b_adv);
		dm::fma<N>(-eps, b_dir, b, b_ret);

		dm::cholesky_solve(dm::constant(L), b_adv, x_adv);
		dm::cholesky_solve(dm::constant(L), b_ret, x_ret);

		Float dx_b_fd[N] = { 0 };
		dm::sub<N>(x_adv, x_ret, dx_b_fd);
		dm::mul<N>(dx_b_fd, Float(1. / (2. * eps)), dx_b_fd);

		// Differentiate *one* output variable and compare to finite differences
		std::optional<Float> error_max;
		Float dx_A_bwd[N] = { 0 };
		Float dx_b_bwd[N] = {0};
		for (unsigned int k = 0; k < N; ++k)
		{
			Float dx[N] = { 0 };
			dx[k] = Float(1);

			// dA represents dxi/dA (gradient of xi w.r.t. A)
			dm::SymmetricMatrix<Float, N> dA{ .data = A_storage_grad[3].data };
			Float db[N] = { 0 };
			dm::cholesky_solve_backward(L, b, x, dA, db, dx);

			// Compute the directional derivative using the gradient: <dA/dxi, A_dir>
			dm::cmul(dm::constant(dA), dm::constant(A_dir), dA);
			dx_A_bwd[k] = dm::sum(dm::constant(dA));

			Float error = std::abs(dx_A_bwd[k] - dx_A_fd[k]) / std::abs(dx_A_fd[k]); // relative error
			if (!error_max || *error_max < error)
				error_max = error;

			// Compute the directional derivative using the gradient: <db/dxi, b_dir>
			dx_b_bwd[k] = dm::dot<N>(db, b_dir);
			
			error = relative_error_sym(dx_b_bwd[k], dx_b_fd[k]);
			if (!error_max || *error_max < error)
				error_max = error;
		}
		
		// Record the *global* maximum error across all polynomials
		if (!error_max_all || *error_max > *error_max_all)
			error_max_all = *error_max;

		if (*error_max > tolerance)
		{
			if (verbose)
			{
				printf("test_cholesky_solve_backward<%s,%d>(): error for matrix %d exceeds tolerance (%0.12f)\n", typeid(Float).name(), N, idx, *error_max);
				//printf("-- Matrix (A, det(A)=%f):\n", dm::det(L) * dm::det(L));
				//print_matrix(A);
				//printf("-- Cholesky Decomp (L, det(L)=%f):\n", dm::det(L));
				//print_matrix(L);
				//printf("-- Target (b):\n");
				//print_vector<N>(b);
				//printf("-- Direction (A_dir):\n");
				//print_matrix(A_dir);
				//printf("-- Directional Derivative for A (ref):\n");
				//print_vector<N>(dx_A_fd);
				//printf("-- Directional Derivative for A (adjoint):\n");
				//print_vector<N>(dx_A_bwd);
				//printf("-- Directional Derivative for b (ref):\n");
				//print_vector<N>(dx_b_fd);
				//printf("-- Directional Derivative for b (adjoint):\n");
				//print_vector<N>(dx_b_bwd);
			}
			num_failed++;
		}
	}

	unsigned int num_matrices_valid = num_matrices - num_no_decomposition;

	char const* status = num_failed > 0 ? "FAILED" : "passed";
	printf("test_cholesky_solve_backward<%s,%d>(): %s (%06d/%06d matrices passed, max. error=%e, tolerance=%e)\n", typeid(Float).name(), N, status, num_matrices_valid -num_failed, num_matrices_valid, *error_max_all, tolerance);
}

template<typename Float, unsigned int N, bool TestSpecialCases>
void test_bjoerck_pereyra_solve(bool verbose)
{
	constexpr unsigned int num_matrices = 1000000;

	Float tolerance = get_solver_tolerance<Float>();

	std::default_random_engine engine(N);
	std::normal_distribution<Float> high_normal_distribution(0, 10);
	std::normal_distribution<Float> unit_normal_distribution(0, 1);
	std::uniform_real<Float> uniform_distribution;

	std::optional<Float> error_max_all;
	unsigned int num_failed = 0;
	for (unsigned int idx = 0; idx < num_matrices; ++idx)
	{
		// Generate a random vector of support points that define the Vandermonde matrix V
		Float a[N] = { 0 };
		for (Float& v : a)
		{
			v = high_normal_distribution(engine);
		}

		if constexpr (TestSpecialCases)
		{
			// Set one of the support points to inf
			Float p           = uniform_distribution(engine) - Float(0.5);
			unsigned int random_idx = std::min(static_cast<unsigned int>(N*uniform_distribution(engine)), static_cast<unsigned int>(N-1));
			a[random_idx] = std::copysign(std::numeric_limits<Float>::infinity(), p);
		}

		// Generate a the reference solution vector (standard normal distribution, closer to one)
		Float x_ref[N] = { 0 };
		for (Float& v : x_ref)
		{
			v = unit_normal_distribution(engine);
		}

		// Generate the target vector b_ref = V*x_ref
		Float b_ref[N] = { 0 };
		for (unsigned int i = 0; i < N; ++i)
		{
			for (unsigned int k = 0; k < N; ++k)
			{
				b_ref[i] += std::pow(a[k], i) * x_ref[k];
			}
		}

		if constexpr (TestSpecialCases)
		{
			// Choose finite target vector
			for (unsigned int i = 0; i < N; ++i)
				b_ref[i] = x_ref[i];
			// FIXME: Adapt tests
		}

		//if constexpr (N == 3)
		//{
		//  // This is a prime example that fails without pivoting
		//	if (idx > 0)
		//		break;
		//	a[0] = Float(-4.403628e-01);
		//	a[1] = Float(-5.306334e+04);
		//	a[2] = Float(1.164453e+00);
		//	b_ref[0] = 4.482106;
		//	b_ref[1] = 1.785982;
		//	b_ref[2] = 3.591555;
		//}

		// Solve Vandermonde system Vx = b using the Bjoerck Pereyra algorithm
		Float x[N] = { 0 };
		dm::bjoerck_pereyra_solve<Float, N>(a, b_ref, x);

		if constexpr (TestSpecialCases)
		{
			Float da[N] = { 0 };
			Float db[N] = { 0 };
			Float dx[N] = { 0 };
			for (unsigned int i = 0; i < N; i++)
				dx[i] = Float(1);
			dm::bjoerck_pereyra_solve_backward<Float, N>(a, b_ref, x, da, db, dx);
			int i = 123.f;
		}

		// Generate the target vector obtained our solution b = Vx
		Float b[N] = { 0 };
		for (unsigned int i = 0; i < N; ++i)
		{
			for (unsigned int k = 0; k < N; ++k)
			{
				b[i] += std::pow(a[k], i) * x[k];
			}
		}

		// Compute the maximum (relative) error for each component
		std::optional<Float> error_max;
		for (unsigned int i = 0; i < N; ++i)
		{
			Float error = std::abs(b[i] - b_ref[i]) / std::abs(b_ref[i]);
			if (!error_max || error > *error_max)
				error_max = error;
		}

		// Record the *global* maximum error across all polynomials
		if (!error_max_all || *error_max > *error_max_all)
			error_max_all = *error_max;

		if (*error_max > tolerance)
		{
			if (verbose)
			{
				printf("test_bjoerck_pereyra_solve<%s,%d>(): error for matrix %d exceeds tolerance (%0.12f)\n", typeid(Float).name(), N, idx, *error_max);
				//printf("b (ref):");
				//print_vector<N>(b_ref);
				//printf("b (out):");
				//print_vector<N>(b);
				//printf("x (ref):");
				//print_vector<N>(x_ref);
				//printf("x (out):");
				//print_vector<N>(x);
			}
			num_failed++;
		}
	}

	char const* status = num_failed > 0 ? "FAILED" : "passed";
	printf("test_bjoerck_pereyra_solve<%s,%d>(): %s (%06d/%06d matrices passed, max. error=%e, tolerance=%e)\n", typeid(Float).name(), N, status, num_matrices - num_failed, num_matrices, *error_max_all, tolerance);
}

template<typename Float, unsigned int N, bool TestSpecialCases>
void test_bjoerck_pereyra_dual_solve(Float tolerance_factor, bool verbose)
{
	constexpr unsigned int num_matrices = 1000000;

	Float tolerance = tolerance_factor*get_solver_tolerance<Float>();

	std::default_random_engine engine(N);
	std::normal_distribution<Float> high_normal_distribution(0, 1000);
	std::normal_distribution<Float> unit_normal_distribution(0, 1);
	std::uniform_real<Float> uniform_distribution;

	std::optional<Float> error_max_all;
	std::chrono::nanoseconds time(0);
	unsigned int num_failed = 0;
	for (unsigned int idx = 0; idx < num_matrices; ++idx)
	{
		// Generate a random vector of support points that define the Vandermonde matrix V
		Float a[N] = { 0 };
		for (Float& v : a)
		{
			v = high_normal_distribution(engine);
		}

		if constexpr (TestSpecialCases)
		{
			// Set one of the support points to inf
			Float p = uniform_distribution(engine) - Float(0.5);
			unsigned int random_idx = std::min(static_cast<unsigned int>(N * uniform_distribution(engine)), static_cast<unsigned int>(N - 1));
			a[random_idx] = std::copysign(std::numeric_limits<Float>::infinity(), p);
		}

		// Generate a the reference solution vector
		Float x_ref[N] = { 0 };
		for (Float& v : x_ref)
		{
			v = unit_normal_distribution(engine);
		}

		// Generate the target vector b_ref = V^T*x_ref
		Float b_ref[N] = { 0 };
		for (unsigned int i = 0; i < N; ++i)
		{
			b_ref[i] = dm::evaluate_polynomial<Float, N-1>(x_ref, a[i]);
		}

		if constexpr (TestSpecialCases)
		{
			// Choose finite target vector
			for (unsigned int i = 0; i < N; ++i)
				b_ref[i] = x_ref[i];
			// FIXME: Adapt tests
		}

		// Solve Vandermonde system Vx = b using the Bjoerck Pereyra algorithm
		Float x[N] = { 0 };
		auto time_start = std::chrono::high_resolution_clock::now();
		dm::bjoerck_pereyra_dual_solve<Float, N>(a, b_ref, x);
		time += std::chrono::high_resolution_clock::now() - time_start;

		// Generate the target vector obtained our solution b = Vx
		Float b[N] = { 0 };
		for (unsigned int i = 0; i < N; ++i)
		{
			b[i] = dm::evaluate_polynomial<Float, N-1>(x, a[i]);
		}

		// Compute the maximum (relative) error for each component
		std::optional<Float> error_max;
		for (unsigned int i = 0; i < N; ++i)
		{
			Float error = std::abs(b[i] - b_ref[i]) / std::abs(b_ref[i]);
			if (!error_max || error > *error_max)
				error_max = error;
		}

		// Record the *global* maximum error across all polynomials
		if (!error_max_all || *error_max > *error_max_all)
			error_max_all = *error_max;

		if (*error_max > tolerance)
		{
			if (verbose)
			{
				printf("test_bjoerck_pereyra_dual_solve<%s,%d>(): error for matrix %d exceeds tolerance (%0.12f)\n", typeid(Float).name(), N, idx, *error_max);
				//printf("b (ref):");
				//print_vector<N>(b_ref);
				//printf("b (out):");
				//print_vector<N>(b);
				//printf("x (ref):");
				//print_vector<N>(x_ref);
				//printf("x (out):");
				//print_vector<N>(x);
			}
			num_failed++;
		}
	}

	time /= num_matrices;

	char const* status = num_failed > 0 ? "FAILED" : "passed";
	printf("test_bjoerck_pereyra_dual_solve<%s,%d>(): %s (%06d/%06d matrices passed, avg. time=%lld ns, max. error=%e, tolerance=%e)\n", typeid(Float).name(), N, status, num_matrices - num_failed, num_matrices, time.count(), *error_max_all, tolerance);
}

template<typename Float, unsigned int N>
void test_bjoerck_pereyra_solve_backward(Float tolerance, bool verbose)
{
	constexpr unsigned int num_matrices = 100000;

	std::default_random_engine engine(N);
	std::normal_distribution<Float> high_normal_distribution(0, 1000);
	std::normal_distribution<Float> unit_normal_distribution(0, 1);
	std::default_random_engine engine2(N);
	std::uniform_real_distribution<Float> distribution(-5, 5);

	std::optional<Float> error_max_all;
	unsigned int num_failed = 0;
	for (unsigned int idx = 0; idx < num_matrices; ++idx)
	{
		// Generate a random vector of support points that define the Vandermonde matrix V
		Float a[N] = { 0 };
		for (Float& v : a)
		{
			v = high_normal_distribution(engine);
		}

		// Generate a the reference solution vector
		Float x_ref[N] = { 0 };
		for (Float& v : x_ref)
		{
			v = unit_normal_distribution(engine);
		}

		// Generate the target vector b_ref = V*x_ref
		Float b_ref[N] = { 0 };
		for (unsigned int i = 0; i < N; ++i)
		{
			for (unsigned int k = 0; k < N; ++k)
			{
				b_ref[i] += std::pow(a[k], i) * x_ref[k];
			}
		}

		// Solve Vandermonde system Vx = b using the Bjoerck Pereyra algorithm
		Float x[N] = { 0 };
		dm::bjoerck_pereyra_solve<Float, N>(a, b_ref, x);

		// Generate the target vector obtained our solution b = Vx
		Float b[N] = { 0 };
		for (unsigned int i = 0; i < N; ++i)
		{
			for (unsigned int k = 0; k < N; ++k)
			{
				b[i] += std::pow(a[k], i) * x[k];
			}
		}

		// Test sensitivity for random directional derivative 

		// ... for the input vector a
		Float a_dir[N] = { 0 };
		for (Float& s : a_dir)
			s = distribution(engine2);
		dm::mul<N>(a_dir, Float(1) / dm::abssum<N>(a_dir), a_dir);

		// ... and the right hand size b
		Float b_dir[N] = { 0 };
		for (Float& s : b_dir)
			s = distribution(engine2);
		dm::mul<N>(b_dir, Float(1) / dm::abssum<N>(b_dir), b_dir);

		// Compute finite differences...

		Float eps(0.0001);
		Float x_adv[N] = { 0 };
		Float x_ret[N] = { 0 };

		// ... for a
		Float a_adv[N] = { 0 };
		Float a_ret[N] = { 0 };
		dm::fma<N>(+eps, a_dir, a, a_adv);
		dm::fma<N>(-eps, a_dir, a, a_ret);

		dm::bjoerck_pereyra_solve<Float, N>(a_adv, b_ref, x_adv);
		dm::bjoerck_pereyra_solve<Float, N>(a_ret, b_ref, x_ret);

		Float dx_a_fd[N] = { 0 };
		dm::sub<N>(x_adv, x_ret, dx_a_fd);
		dm::mul<N>(dx_a_fd, Float(1. / (2. * eps)), dx_a_fd);

		// ... and for b
		Float b_adv[N] = { 0 };
		Float b_ret[N] = { 0 };
		dm::fma<N>(+eps, b_dir, b_ref, b_adv);
		dm::fma<N>(-eps, b_dir, b_ref, b_ret);

		dm::bjoerck_pereyra_solve<Float, N>(a, b_adv, x_adv);
		dm::bjoerck_pereyra_solve<Float, N>(a, b_ret, x_ret);

		Float dx_b_fd[N] = { 0 };
		dm::sub<N>(x_adv, x_ret, dx_b_fd);
		dm::mul<N>(dx_b_fd, Float(1. / (2. * eps)), dx_b_fd);

		// Differentiate *one* output variable and compare to finite differences
		std::optional<Float> error_max;
		Float dx_a_bwd[N] = { 0 };
		Float dx_b_bwd[N] = { 0 };
		for (unsigned int k = 0; k < N; ++k)
		{
			Float dx[N] = { 0 };
			dx[k] = Float(1);

			Float da[N] = { 0 };
			Float db[N] = { 0 };
			dm::bjoerck_pereyra_solve_backward<Float, N>(a, b, x, da, db, dx);

			// Compute the directional derivative using the gradient: <dxi/da, a_dir>
			dx_a_bwd[k] = dm::dot<N>(da, a_dir);

			Float error = relative_error_sym(dx_a_bwd[k], dx_a_fd[k]);
			if (!error_max || *error_max < error)
				error_max = error;

			// Compute the directional derivative using the gradient: <dxi/db, b_dir>
			dx_b_bwd[k] = dm::dot<N>(db, b_dir);

			error = relative_error_sym(dx_b_bwd[k], dx_b_fd[k]);
			if (!error_max || *error_max < error)
				error_max = error;
		}

		// Record the *global* maximum error across all polynomials
		if (!error_max_all || *error_max > *error_max_all)
			error_max_all = *error_max;

		if (*error_max > tolerance)
		{
			if (verbose)
			{
				printf("test_bjoerck_pereyra_solve_backward<%s,%d>(): error for matrix %d exceeds tolerance (%0.12f)\n", typeid(Float).name(), N, idx, *error_max);
				//printf("-- Vector a:\n");
				//print_vector<N>(a);
				//printf("-- Target (b_ref):\n");
				//print_vector<N>(b_ref);
				//printf("-- Target (b):\n");
				//print_vector<N>(b);
				//printf("-- Direction (a_dir):\n");
				//print_vector<N>(a_dir);
				//printf("-- Direction (b_dir):\n");
				//print_vector<N>(b_dir);
				//printf("-- Directional Derivative for a (ref):\n");
				//print_vector<N>(dx_a_fd);
				//printf("-- Directional Derivative for a (adjoint):\n");
				//print_vector<N>(dx_a_bwd);
				//printf("-- Directional Derivative for b (ref):\n");
				//print_vector<N>(dx_b_fd);
				//printf("-- Directional Derivative for b (adjoint):\n");
				//print_vector<N>(dx_b_bwd);
			}
			num_failed++;
		}
	}

	char const* status = num_failed > 0 ? "FAILED" : "passed";
	printf("test_bjoerck_pereyra_solve_backward<%s,%d>(): %s (%06d/%06d matrices passed, max. error=%e, tolerance=%e)\n", typeid(Float).name(), N, status, num_matrices - num_failed, num_matrices, *error_max_all, tolerance);
}

template<typename Float, unsigned int Degree>
void test_moment_curve_backward(Float tolerance, bool verbose)
{
	constexpr unsigned int num_values = 100;

	std::default_random_engine engine(Degree);
	std::uniform_real_distribution<Float> distribution(-5, 5);

	std::optional<Float> error_max_all;
	unsigned int num_failed = 0;
	for (unsigned int idx = 0; idx < num_values; ++idx)
	{
		// Generate a value and evaluate the moment curve
		Float t = distribution(engine);

		Float u[Degree + 1] = { 0 };
		dm::moment_curve<Degree, Float>(t, u);

		// Test sensitivity for random directional derivative
		// FIXME: Dependency on the random number distribution
		Float t_dir = distribution(engine) / Float(5);

		// Compute directional derivative using finite differences (reference)
		Float eps(0.000001);
		Float t_adv = t + eps * t_dir;
		Float t_ret = t - eps * t_dir;

		Float u_adv[Degree + 1] = { 0 };
		dm::moment_curve<Degree, Float>(t_adv, u_adv);

		Float u_ret[Degree + 1] = { 0 };
		dm::moment_curve<Degree, Float>(t_ret, u_ret);

		Float du_fd[Degree + 1] = { 0 };
		dm::mul<Degree + 1>(u_ret, -1 / (2 * eps), du_fd);
		dm::fma<Degree + 1>(1 / (2 * eps), u_adv, du_fd, du_fd);

		// Differentiate w.r.t. *one* output and compare to finite differences
		std::optional<Float> error_max;
		Float du_bwd[Degree + 1] = { 0 };
		for (unsigned int k = 0; k < Degree + 1; ++k)
		{
			Float du[Degree + 1] = { 0 };
			du[k] = Float(1);

			Float dt(0);
			dm::moment_curve_backward<Degree, Float>(t, nullptr, dt, du);

			// Compute directional derivative
			du_bwd[k] = t_dir * dt;

			Float error = std::abs(du_bwd[k] - du_fd[k]);
			if (du_fd[k] != Float(0))
				error /= std::abs(du_fd[k]); // relative error

			if (!error_max || *error_max < error)
				error_max = error;
		}

		// Record the *global* maximum error across all polynomials
		if (!error_max_all || *error_max > *error_max_all)
			error_max_all = *error_max;

		if (*error_max > tolerance)
		{
			printf("test_moment_curve_backward<%s,%d>(): error for value %d exceeds tolerance (%0.12f)\n", typeid(Float).name(), Degree, idx, *error_max);
			if (verbose)
			{
				printf("value: %f\n", t);
			}
			num_failed++;
		}
	}

	char const* status = num_failed > 0 ? "FAILED" : "passed";
	printf("test_moment_curve_backward<%s,%d>(): %s (%06d/%06d values passed, max. error=%e, tolerance=%e)\n", typeid(Float).name(), Degree, status, num_values - num_failed, num_values, *error_max_all, tolerance);
}

template<typename Float, unsigned int N>
void test_compute_moment_bound(Float tolerance, bool verbose)
{
	constexpr unsigned int num_moments = 100000;

	constexpr Float root_finding_tolerance = std::numeric_limits<Float>::epsilon();
	constexpr unsigned int root_finding_max_iterations = 100u;

	std::default_random_engine engine(N);

	std::uniform_real_distribution<Float> uniform_distribution(0, 1);
	std::normal_distribution<Float> normal_distribution;

	std::optional<Float> error_max_all;
	unsigned int num_nobounds = 0;
	unsigned int num_failed   = 0;
	for (unsigned int idx = 0; idx < num_moments; ++idx)
	{
		// Generate N values (maybe more)
		Float xs_ref[N + 1];
		for (Float& x : xs_ref)
		{
			x = normal_distribution(engine);
		}

		// Compute a positive moment sequence from the values
		Float moments_ref[2 * N + 1] = { 0 };
		for (unsigned int i = 0; i < std::size(xs_ref); ++i)
		{
			Float u[2 * N + 1] = { 0 };
			dm::moment_curve<2 * N, Float>(xs_ref[i], u);

			Float a = uniform_distribution(engine);
			dm::fma<2 * N + 1>(a, u, moments_ref, moments_ref);
		}

		Float eta = normal_distribution(engine); // If we set xs_ref[0];

		dm::MomentBoundParams<Float> params
		{
			.bias = Float(0), // TODO: Need to apply a tiny bit of bias because not all inputs are positive sequences
			.overestimation_weight = uniform_distribution(engine),
			.newton_tolerance = root_finding_tolerance,
			.newton_max_iterations = root_finding_max_iterations
		};
		Float L[dm::LowerTriangularMatrix<Float, N + 1>::data_size];
		Float coeffs[N + 1] = { 0 };
		Float roots[N + 1]   = { 0 };
		Float weights[N + 1] = { 0 };
		Float bound;
		dm::MomentBoundResult result = dm::compute_moment_bound<N, Float>(params, moments_ref, eta, &bound, L, coeffs, roots, weights);
		if (result != dm::MomentBoundResult::Success)
		{
			if (verbose)
				printf("test_compute_moment_bound<%s,%d>(): vector %d computing bounds failed, reason '%s'\n", typeid(Float).name(), N, idx, dm::get_result_string(result));
			++num_nobounds;
			continue;
		}

		// Compute the predicted moment vector (from roots and weights)
		Float moments[2 * N + 1] = { 0 };
		for (unsigned int i = 0; i < std::size(roots); ++i)
		{
			Float u[2 * N + 1] = { 0 };
			dm::moment_curve<2 * N, Float>(roots[i], u);
			dm::fma<2 * N + 1>(weights[i], u, moments, moments);
		}

		// Compute the maximum error over the recovered moments
		// If we generate N + 1 points xs_ref, we expect the perfect reconstruction
		std::optional<Float> error_max;
		for (unsigned int k = 0; k < N + 1/*std::size(moments_ref)*/; ++k)
		{
			Float error = std::abs(moments[k] - moments_ref[k]) / std::abs(moments_ref[k]);

			if (!error_max || *error_max < error)
				error_max = error;
		}

		// Record the *global* maximum error across all polynomials
		if (!error_max_all || *error_max > *error_max_all)
			error_max_all = *error_max;

		if (*error_max > tolerance || !std::isfinite(*error_max))
		{
			if (verbose)
			{
				printf("test_compute_moment_bound<%s,%d>(): vector %d error exceeds tolerance (%0.12f)\n", typeid(Float).name(), N, idx, *error_max);
				printf("moments (ref):\n");
				print_vector<2 * N + 1>(moments_ref);
				printf("moments (solution):\n");
				print_vector<2 * N + 1>(moments);
			}
			num_failed++;
		}
	}

	unsigned int num_moments_valid = num_moments - num_nobounds;

	char const* status = num_failed > 0 ? "FAILED" : "passed";
	printf("test_compute_moment_bound<%s,%d>(): %s (%06d/%06d values passed, max. error=%e, tolerance=%e)\n", typeid(Float).name(), N, status, num_moments_valid - num_failed, num_moments_valid, *error_max_all, tolerance);
}

template<typename Float, unsigned int N>
void test_compute_moment_bound_backward(Float tolerance, bool verbose)
{
	constexpr unsigned int num_moments = 100000;

	constexpr Float root_finding_tolerance = std::numeric_limits<Float>::epsilon();
	constexpr unsigned int root_finding_max_iterations = 100u;

	std::default_random_engine engine(N);
	std::uniform_real_distribution<Float> distribution(-5, 5);
	std::uniform_real_distribution<Float> unit_distribution(0, 1);

	std::optional<Float> error_max_all;
	unsigned int num_failed = 0;
	for (unsigned int idx = 0; idx < num_moments; ++idx)
	{
		// Generate N values (maybe more)
		Float xs_ref[N + 1];
		for (Float& x : xs_ref)
		{
			x = distribution(engine);
		}

		// Compute a positive moment sequence from the values
		Float moments_ref[2 * N + 1] = { 0 };
		for (unsigned int i = 0; i < std::size(xs_ref); ++i)
		{
			Float u[2 * N + 1] = { 0 };
			dm::moment_curve<2 * N, Float>(xs_ref[i], u);
			Float c = std::abs(distribution(engine));
			dm::fma<2 * N + 1>(c, u, moments_ref, moments_ref);
		}

		// Input parameters
		dm::MomentBoundParams<Float> params
		{
			.bias = Float(3e-5),
			.overestimation_weight = unit_distribution(engine),
			.newton_tolerance = root_finding_tolerance,
			.newton_max_iterations = root_finding_max_iterations
		};
		Float eta   = 2*unit_distribution(engine) - 1; // Sample eta in [-1, 1]
		Float bound;

		// Output parameters (saved for backward)
		Float L[dm::SymmetricMatrix<Float, N + 1>::data_size];
		Float coeffs[N + 1];
		Float roots[N + 1];
		Float weights[N + 1];
		dm::compute_moment_bound<N, Float>(params, moments_ref, eta, &bound,
										   L, coeffs, roots, weights);

		// Test sensitivity for random directional derivative 

		// ... for the input moments
		Float m_dir[2 * N + 1] = { 0 };
		for (Float& s : m_dir)
			s = distribution(engine);
		Float m_dir_length = std::sqrt(dm::dot<2 * N + 1>(m_dir, m_dir));
		dm::mul<2 * N + 1>(m_dir, Float(1) / m_dir_length, m_dir);

		// ... and eta
		Float eta_dir = unit_distribution(engine) - 0.5; // Positive and negative direction

		// Compute finite differences...

		Float eps(0.000001); // TODO: Tune this for float
		Float bound_adv = 0;
		Float bound_ret = 0;

		// ... for the moments
		Float m_adv[2 * N + 1] = { 0 };
		Float m_ret[2 * N + 1] = { 0 };
		dm::fma<2 * N + 1>(+eps, m_dir, moments_ref, m_adv);
		dm::fma<2 * N + 1>(-eps, m_dir, moments_ref, m_ret);

		dm::compute_moment_bound<N, Float>(params, m_adv, eta, &bound_adv);
		dm::compute_moment_bound<N, Float>(params, m_ret, eta, &bound_ret);
		Float dbound_m_fd = (bound_adv - bound_ret) / (2. * eps);

		// ... and for eta
		Float eta_adv = eta + eps * eta_dir;
		Float eta_ret = eta - eps * eta_dir;

		dm::compute_moment_bound<N, Float>(params, moments_ref, eta_adv, &bound_adv);
		dm::compute_moment_bound<N, Float>(params, moments_ref, eta_ret, &bound_ret);
		Float dbound_eta_fd = (bound_adv - bound_ret) / (2. * eps);

		// Differentiate *one* output variable and compare to finite differences
		std::optional<Float> error_max;

		Float dm[2*N + 1] = { 0 };
		Float deta(0);
		dm::compute_moment_bound_backward<N, Float>(params, moments_ref, eta, bound, 
													L, coeffs, roots, weights,
													dm, &deta, Float(1));

		Float dbound_m_bwd = dm::dot<2 * N + 1>(dm, m_dir);
		Float error = std::abs(dbound_m_bwd - dbound_m_fd) / (std::abs(dbound_m_fd) + 1e-8);
		if (!error_max || *error_max < error)
			error_max = error;

		Float dbound_eta_bwd = deta * eta_dir;
		error = std::abs(dbound_eta_bwd - dbound_eta_fd) / (std::abs(dbound_eta_fd) + 1e-8);
		if (!error_max || *error_max < error)
			error_max = error;

		// Record the *global* maximum error across all polynomials
		if (!error_max_all || *error_max > *error_max_all)
			error_max_all = *error_max;

		bool has_derivative_different_sign = (dbound_m_bwd * dbound_m_fd < 0) || (dbound_eta_bwd * dbound_eta_fd < 0);

		if (*error_max > tolerance || has_derivative_different_sign)
		{
			if (has_derivative_different_sign)
				printf("test_compute_moment_bound_backward<%s,%d>(): vector %d derivative has wrong sign!\n", typeid(Float).name(), N, idx);
			if (verbose)
			{
				if(*error_max > tolerance)
					printf("test_compute_moment_bound_backward<%s,%d>(): vector %d error exceeds tolerance (%0.12f)\n", typeid(Float).name(), N, idx, *error_max);

				printf("-- Moments (moments_ref):\n");
				print_vector<2*N + 1>(moments_ref);
				printf("-- eta: %f\n", eta);
				printf("-- Direction (m_dir):\n");
				print_vector<2 * N + 1>(m_dir);
				printf("-- Direction (eta_dir): %f\n", eta_dir);
				printf("-- Directional Derivative for m (ref): %f\n", dbound_m_fd);
				printf("-- Directional Derivative for m (adjoint): %f\n", dbound_m_bwd);
				printf("-- Directional Derivative for eta (ref): %f\n", dbound_eta_fd);
				printf("-- Directional Derivative for eta (adjoint): %f\n", dbound_eta_bwd);
			}
			num_failed++;
		}
	}

	char const* status = num_failed > 0 ? "FAILED" : "passed";
	printf("test_compute_moment_bound_backward<%s,%d>(): %s (%06d/%06d vectors passed, max. error=%e, tolerance=%e)\n", typeid(Float).name(), N, status, num_moments - num_failed, num_moments, *error_max_all, tolerance);
}