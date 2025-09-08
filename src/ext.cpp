#include <limits>
#include <functional>
#include <stdexcept>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "dispatch.hpp"
#include "cuda_driver.hpp"
#include "cuda_module.hpp"
#include "log.hpp"
#include "kernel_flags.hpp"
#include "moment_problem.hpp"

namespace nb = nanobind;

// Checks if a each nanobind array is allocated on the given device
template<typename ArrayType, typename... Arrays>
void check_on_device(int device_type, int device_id, ArrayType&& a, Arrays&&... arrays)
{
    if (a.device_type() != device_type || a.device_id() != device_id)
        throw std::invalid_argument("Input arrays must be on the same device");

    if constexpr (sizeof...(arrays) > 0)
        check_on_device(device_type, device_id, std::forward<Arrays>(arrays)...);
}

template<typename Float>
dm::MomentBoundParams<Float> get_params_with_defaults(Float bias, Float overestimation_weight)
{
    dm::MomentBoundParams<Float> params
    {
        .bias = bias,
        .overestimation_weight = overestimation_weight,
        .newton_tolerance = std::numeric_limits<Float>::epsilon() // FIXME: SET REASONABLE!
    };

    if constexpr (std::is_same_v<Float, double>)
        params.newton_max_iterations = 200; // Higher!

    return params;
}

template<unsigned int N, typename Float>
void compute_moment_bounds_anydevice(int device_type, int device_id, unsigned int size, uint32_t flags, dm::MomentBoundParams<Float> const& params, Float const* moments, Float const* etas, Float* bounds, Float* roots, Float* weights)
{
    // Dispatch the computation based on the device type

    if (device_type == nb::device::cpu::value)
        run_with_flags<+ComputeMomentBoundsFlags::None, +ComputeMomentBoundsFlags::All>([&]<uint32_t Flags>()
    {
        compute_moment_bounds_cpu<N, Float, Flags>(size, moments, etas, bounds, roots, weights, params);
    }, flags);
    else if (device_type == nb::device::cuda::value)
        run_with_flags<+ComputeMomentBoundsFlags::None, +ComputeMomentBoundsFlags::All>([&]<uint32_t Flags>()
    {
        compute_moment_bounds_gpu<N, Float, Flags>(device_id, size, moments, etas, bounds, roots, weights, params);
    }, flags);
    else
        throw std::invalid_argument(format_message("Inputs must be either CPU or GPU (CUDA) arrays."));
}

template<typename Float>
void compute_moment_bounds(nb::ndarray<Float, nb::c_contig> moments,
                           nb::ndarray<Float, nb::c_contig> etas,
                           nb::ndarray<Float, nb::c_contig> bounds,
                           nb::ndarray<Float, nb::c_contig> storage,
                           Float bias,
                           Float overestimation_weight,
                           uint32_t flags)
{
    check_on_device(moments.device_type(), moments.device_id(), etas, bounds, storage);

    if (moments.ndim() != 2)
        throw std::invalid_argument(format_message("Expected moments with shape (num_moments, n) but array has %d dimensions", moments.ndim()));

    if (etas.ndim() != 1)
        throw std::invalid_argument(format_message("Expected etas with shape (n,) but array has %d dimensions", etas.ndim()));

    if (bounds.ndim() != 1)
        throw std::invalid_argument(format_message("Expected bounds with shape (n,) but array has %d dimensions", bounds.ndim()));

    if (storage.ndim() != 1)
        throw std::invalid_argument(format_message("Expected storage with shape (n,) but array has %d dimensions", storage.ndim()));

    BackwardStorageDesc backward_desc;
    get_backward_storage_desc<Float>(moments.shape(1), moments.shape(0), flags, &backward_desc);
    if (storage.shape(0) != backward_desc.total_size)
        throw std::invalid_argument(format_message("Expected storage with size %d but array has size %d", backward_desc.total_size, storage.shape(0)));

    unsigned int size = static_cast<unsigned int>(moments.shape(1));
    if (etas.shape(0) != size)
        throw std::invalid_argument(format_message("Expected moments (n=%d) and etas (n=%d) to have the same size", size, etas.shape(0)));

    if (bounds.shape(0) != size)
        throw std::invalid_argument(format_message("Expected moments (n=%d) and bounds (n=%d) to have the same size", size, bounds.shape(0)));

    // Get the number of moments
    unsigned int num_moments = moments.shape(0);
    unsigned int n = (num_moments - 1) / 2;

    dm::MomentBoundParams<Float> params = get_params_with_defaults<Float>(bias, overestimation_weight);

    // Get the data pointers from the storage
    // NOTE: Adding an offset to nullptr is undefined behavior, so explicitly handle it
    Float* roots = storage.data() ? storage.data() + backward_desc.offset_roots : nullptr;

    if (n == 1)
        compute_moment_bounds_anydevice<1, Float>(moments.device_type(), moments.device_id(), size, flags, params, moments.data(), etas.data(), bounds.data(), roots, nullptr);
    else if (n == 2)
        compute_moment_bounds_anydevice<2, Float>(moments.device_type(), moments.device_id(), size, flags, params, moments.data(), etas.data(), bounds.data(), roots, nullptr);
    else if (n == 3)
        compute_moment_bounds_anydevice<3, Float>(moments.device_type(), moments.device_id(), size, flags, params, moments.data(), etas.data(), bounds.data(), roots, nullptr);
    else if (n == 4)
        compute_moment_bounds_anydevice<4, Float>(moments.device_type(), moments.device_id(), size, flags, params, moments.data(), etas.data(), bounds.data(), roots, nullptr);
    else if (n == 5)
        compute_moment_bounds_anydevice<5, Float>(moments.device_type(), moments.device_id(), size, flags, params, moments.data(), etas.data(), bounds.data(), roots, nullptr);
    else if (n == 6)
        compute_moment_bounds_anydevice<6, Float>(moments.device_type(), moments.device_id(), size, flags, params, moments.data(), etas.data(), bounds.data(), roots, nullptr);
    else
        throw std::invalid_argument(format_message("compute_moment_bounds only supports <= %d moments (n <= %d), but received %d moments (n = %d)", 6*2 + 1, 6, num_moments, n));
}

// Returns the required size for the backward storage, in number of floating point elements.
template<typename Float>
unsigned int get_required_backward_storage_size(nb::ndarray<Float, nb::c_contig> moments, uint32_t flags)
{
    if (moments.ndim() != 2)
        throw std::invalid_argument(format_message("Expected moments with shape (num_moments, n) but array has %d dimensions", moments.ndim()));

    BackwardStorageDesc desc;
    get_backward_storage_desc<Float>(moments.shape(1), moments.shape(0), flags, &desc);
    return desc.total_size;
}

template<unsigned int N, typename Float>
void compute_moment_bounds_backward_anydevice(int device_type, int device_id, unsigned int size, uint32_t flags, dm::MomentBoundParams<Float> const& params, Float const* moments, Float const* etas, Float const* bounds, Float const* roots, Float* dmoments, Float* detas, Float const* dbounds)
{
    if (device_type == nb::device::cpu::value)
        run_with_flags<+ComputeMomentBoundsFlags::None, +ComputeMomentBoundsFlags::All>([&]<uint32_t Flags>() {
        compute_moment_bounds_backward_cpu<N, Float, Flags>(size, moments, etas, bounds, roots, params, dmoments, detas, dbounds);
    }, flags);
    else if (device_type == nb::device::cuda::value)
        run_with_flags<+ComputeMomentBoundsFlags::None, +ComputeMomentBoundsFlags::All>([&]<uint32_t Flags>() {
        compute_moment_bounds_backward_gpu<N, Float, Flags>(device_id, size, moments, etas, bounds, roots, params, dmoments, detas, dbounds);
    }, flags);
    else
        throw std::invalid_argument(format_message("Inputs must be either CPU or GPU (CUDA) arrays."));
}

template<typename Float>
void compute_moment_bounds_backward(nb::ndarray<Float, nb::c_contig> moments,
                                    nb::ndarray<Float, nb::c_contig> etas,
                                    nb::ndarray<Float, nb::c_contig> bounds,
                                    nb::ndarray<Float, nb::c_contig> storage,
                                    nb::ndarray<Float, nb::c_contig> dmoments,
                                    nb::ndarray<Float, nb::c_contig> detas,
                                    nb::ndarray<Float, nb::c_contig> dbounds,
                                    Float bias,
                                    Float overestimation_weight,
                                    uint32_t flags)
{
    check_on_device(moments.device_type(), moments.device_id(), etas, bounds, storage, dmoments, detas, dbounds);

    if (moments.ndim() != 2)
        throw std::invalid_argument(format_message("Expected moments with shape (num_moments, n) but array has %d dimensions", moments.ndim()));

    if (etas.ndim() != 1)
        throw std::invalid_argument(format_message("Expected etas with shape (n,) but array has %d dimensions", etas.ndim()));

    if (bounds.ndim() != 1)
        throw std::invalid_argument(format_message("Expected bounds with shape (n,) but array has %d dimensions", bounds.ndim()));

    if (dmoments.ndim() != 2)
        throw std::invalid_argument(format_message("Expected dmoments with shape (num_moments, n) but array has %d dimensions", moments.ndim()));

    if (detas.ndim() != 1)
        throw std::invalid_argument(format_message("Expected detas with shape (n,) but array has %d dimensions", etas.ndim()));

    if (dbounds.ndim() != 1)
        throw std::invalid_argument(format_message("Expected dbounds with shape (n,) but array has %d dimensions", bounds.ndim()));

    if (storage.ndim() != 1)
        throw std::invalid_argument(format_message("Expected storage with shape (n,) but array has %d dimensions", storage.ndim()));

    BackwardStorageDesc backward_desc;
    get_backward_storage_desc<Float>(moments.shape(1), moments.shape(0), flags, &backward_desc);
    if (storage.shape(0) != backward_desc.total_size)
        throw std::invalid_argument(format_message("Expected storage with size %d but array has size %d", backward_desc.total_size, storage.shape(0)));

    unsigned int size = static_cast<unsigned int>(moments.shape(1));
    if (dmoments.shape(0) != moments.shape(0) || dmoments.shape(1) != size)
        throw std::invalid_argument(format_message("Expected moments (m=%d,n=%d) and dmoments (m=%d,n=%d) to have the same shape", moments.shape(0), moments.shape(1), dmoments.shape(0), dmoments.shape(1)));

    if (etas.shape(0) != size)
        throw std::invalid_argument(format_message("Expected moments (n=%d) and etas (n=%d) to have the same size", size, etas.shape(0)));

    if (bounds.shape(0) != size)
        throw std::invalid_argument(format_message("Expected moments (n=%d) and bounds (n=%d) to have the same size", size, bounds.shape(0)));

    if (detas.shape(0) != size)
        throw std::invalid_argument(format_message("Expected moments (n=%d) and etas (n=%d) to have the same size", size, etas.shape(0)));

    if (dbounds.shape(0) != size)
        throw std::invalid_argument(format_message("Expected moments (n=%d) and bounds (n=%d) to have the same size", size, bounds.shape(0)));

    // Get the number of moments
    unsigned int num_moments = moments.shape(0);
    unsigned int n = (num_moments - 1) / 2;

    dm::MomentBoundParams<Float> params = get_params_with_defaults<Float>(bias, overestimation_weight);

    // Get the data pointers from the storage
    // NOTE: Adding an offset to nullptr is undefined behavior, so explicitly handle it
    Float* roots = storage.data() ? storage.data() + backward_desc.offset_roots : nullptr;

    if (n == 1)
        compute_moment_bounds_backward_anydevice<1, Float>(moments.device_type(), moments.device_id(), size, flags, params, moments.data(), etas.data(), bounds.data(), roots, dmoments.data(), detas.data(), dbounds.data());
    else if (n == 2)
        compute_moment_bounds_backward_anydevice<2, Float>(moments.device_type(), moments.device_id(), size, flags, params, moments.data(), etas.data(), bounds.data(), roots, dmoments.data(), detas.data(), dbounds.data());
    else if (n == 3)
        compute_moment_bounds_backward_anydevice<3, Float>(moments.device_type(), moments.device_id(), size, flags, params, moments.data(), etas.data(), bounds.data(), roots, dmoments.data(), detas.data(), dbounds.data());
    else if (n == 4)
        compute_moment_bounds_backward_anydevice<4, Float>(moments.device_type(), moments.device_id(), size, flags, params, moments.data(), etas.data(), bounds.data(), roots, dmoments.data(), detas.data(), dbounds.data());
    else if (n == 5)
        compute_moment_bounds_backward_anydevice<5, Float>(moments.device_type(), moments.device_id(), size, flags, params, moments.data(), etas.data(), bounds.data(), roots, dmoments.data(), detas.data(), dbounds.data());
    else if (n == 6)
        compute_moment_bounds_backward_anydevice<6, Float>(moments.device_type(), moments.device_id(), size, flags, params, moments.data(), etas.data(), bounds.data(), roots, dmoments.data(), detas.data(), dbounds.data());
    else
        throw std::invalid_argument(format_message("compute_moment_bounds_backward only supports <= %d moments (n <= %d), but received %d moments (n = %d)", 6 * 2 + 1, 6, num_moments, n));
}

///////////////////////////////////////////
// detect_bound_errors
///////////////////////////////////////////

template<typename Float>
void detect_bound_errors_anydevice(int device_type, int device_id, unsigned int size, Float* bounds)
{
    if (device_type == nb::device::cpu::value)
        detect_bound_errors_cpu<Float>(size, bounds);
    else if (device_type == nb::device::cuda::value)
        detect_bound_errors_gpu<Float>(device_id, size, bounds);
    else
        throw std::invalid_argument(format_message("Inputs must be either CPU or GPU (CUDA) arrays."));
}

template<typename Float>
void detect_bound_errors(nb::ndarray<Float, nb::c_contig> bounds)
{
    if (bounds.ndim() != 1)
        throw std::invalid_argument(format_message("Expected bounds with shape (n,) but array has %d dimensions", bounds.ndim()));

    detect_bound_errors_anydevice<Float>(bounds.device_type(), bounds.device_id(), bounds.shape(0), bounds.data());
}

///////////////////////////////////////////
// compute_moment_bounds_and_roots
///////////////////////////////////////////

template<typename Float>
void compute_moment_bounds_and_roots(nb::ndarray<Float, nb::c_contig> moments,
                                     nb::ndarray<Float, nb::c_contig> etas,
                                     nb::ndarray<Float, nb::c_contig> bounds,
                                     nb::ndarray<Float, nb::c_contig> roots,
                                     nb::ndarray<Float, nb::c_contig> weights,
                                     Float bias,
                                     Float overestimation_weight)
{
	// TODO: Merge with compute_moment_bounds to avoid duplicate code

    if (moments.device_type() != etas.device_type() || moments.device_id() != etas.device_id())
        throw std::invalid_argument("Input arrays must be on the same device");

    if (moments.device_type() != bounds.device_type() || moments.device_id() != bounds.device_id())
        throw std::invalid_argument("Input arrays must be on the same device");

    if (moments.device_type() != roots.device_type() || moments.device_id() != roots.device_id())
        throw std::invalid_argument("Input arrays must be on the same device");

    if (moments.device_type() != weights.device_type() || moments.device_id() != weights.device_id())
        throw std::invalid_argument("Input arrays must be on the same device");

    if (moments.ndim() != 2)
        throw std::invalid_argument(format_message("Expected moments with shape (num_moments, size) but array has %d dimensions", moments.ndim()));

    if (etas.ndim() != 1)
        throw std::invalid_argument(format_message("Expected etas with shape (size,) but array has %d dimensions", etas.ndim()));

    if (bounds.ndim() != 1)
        throw std::invalid_argument(format_message("Expected bounds with shape (size,) but array has %d dimensions", bounds.ndim()));

    if (roots.ndim() != 2)
        throw std::invalid_argument(format_message("Expected roots with shape (n+1,size) but array has %d dimensions", roots.ndim()));

    if (weights.ndim() != 2)
        throw std::invalid_argument(format_message("Expected weights with shape (n+1,size,) but array has %d dimensions", weights.ndim()));

    unsigned int size = static_cast<unsigned int>(moments.shape(1));
    if (etas.shape(0) != size)
        throw std::invalid_argument(format_message("Expected moments (size=%d) and etas (size=%d) to have the same size", size, etas.shape(0)));

    if (bounds.shape(0) != size)
        throw std::invalid_argument(format_message("Expected moments (n=%d) and bounds (size=%d) to have the same size", size, bounds.shape(0)));

    if (roots.shape(1) != size)
        throw std::invalid_argument(format_message("Expected moments (size=%d) and roots (size=%d) to have the same size", size, roots.shape(1)));

    if (weights.shape(1) != size)
        throw std::invalid_argument(format_message("Expected moments (size=%d) and weights (size=%d) to have the same size", size, weights.shape(1)));

    // Get the number of moments
    unsigned int num_moments = moments.shape(0);
    unsigned int n = (num_moments - 1) / 2;

    // Verify number of roots and weights
    if (roots.shape(0) != n + 1)
        throw std::invalid_argument(format_message("Expected roots with shape (n+1,size) = (%d, %d) but array has shape (%d, %d)", n+1, size, roots.shape(0), roots.shape(1)));

    if (weights.shape(0) != n + 1)
        throw std::invalid_argument(format_message("Expected weights with shape (n+1,size) = (%d, %d) but array has shape (%d, %d)", n + 1, size, weights.shape(0), weights.shape(1)));

    dm::MomentBoundParams<Float> params = get_params_with_defaults<Float>(bias, overestimation_weight);

    constexpr uint32_t flags = +ComputeMomentBoundsFlags::RetainRoots | +ComputeMomentBoundsFlags::RetainWeights;

    if (n == 1)
        compute_moment_bounds_anydevice<1, Float>(moments.device_type(), moments.device_id(), size, flags, params, moments.data(), etas.data(), bounds.data(), roots.data(), weights.data());
    else if (n == 2)
        compute_moment_bounds_anydevice<2, Float>(moments.device_type(), moments.device_id(), size, flags, params, moments.data(), etas.data(), bounds.data(), roots.data(), weights.data());
    else if (n == 3)
        compute_moment_bounds_anydevice<3, Float>(moments.device_type(), moments.device_id(), size, flags, params, moments.data(), etas.data(), bounds.data(), roots.data(), weights.data());
    else if (n == 4)
        compute_moment_bounds_anydevice<4, Float>(moments.device_type(), moments.device_id(), size, flags, params, moments.data(), etas.data(), bounds.data(), roots.data(), weights.data());
    else if (n == 5)
        compute_moment_bounds_anydevice<5, Float>(moments.device_type(), moments.device_id(), size, flags, params, moments.data(), etas.data(), bounds.data(), roots.data(), weights.data());
    else if (n == 6)
        compute_moment_bounds_anydevice<6, Float>(moments.device_type(), moments.device_id(), size, flags, params, moments.data(), etas.data(), bounds.data(), roots.data(), weights.data());
    else
        throw std::invalid_argument(format_message("compute_moment_bounds_and_roots only supports <= %d moments (n <= %d), but received %d moments (n = %d)", 6 * 2 + 1, 6, num_moments, n));
}

template<unsigned int N, typename Float>
void compute_singularities_anydevice(int device_type, int device_id, unsigned int size, dm::MomentBoundParams<Float> const& params, Float const* moments, Float* singularities)
{
    if (device_type == nb::device::cpu::value)
        compute_singularities_cpu<N, Float>(size, params, moments, singularities);
    else if (device_type == nb::device::cuda::value)
        compute_singularities_gpu<N, Float>(device_id, size, params, moments, singularities);
    else
        throw std::invalid_argument(format_message("Inputs must be either CPU or GPU (CUDA) arrays."));
}

template<typename Float>
void compute_singularities(nb::ndarray<Float, nb::c_contig> moments,
                           nb::ndarray<Float, nb::c_contig> singularities,
                           Float bias)
{
    check_on_device(moments.device_type(), moments.device_id(), singularities);

    if (moments.ndim() != 2)
        throw std::invalid_argument(format_message("Expected moments with shape (num_moments, n) but array has %d dimensions", moments.ndim()));

    if (singularities.ndim() != 2)
        throw std::invalid_argument(format_message("Expected singularities with shape (num_singularities, n) but array has %d dimensions", singularities.ndim()));

    unsigned int size = static_cast<unsigned int>(moments.shape(1));
    if (singularities.shape(1) != size)
        throw std::invalid_argument(format_message("Expected moments (n=%d) and singularities (n=%d) to have the same size", size, singularities.shape(0)));

    // Get the number of moments
    unsigned int num_moments = moments.shape(0);
    unsigned int n = (num_moments - 1) / 2;

    if (singularities.shape(0) != n)
        throw std::invalid_argument(format_message("Expected singularities with shape (n=%d, ...), received (n=%d,...", n, singularities.shape(0)));

    dm::MomentBoundParams<Float> params = get_params_with_defaults<Float>(bias, 0.);

    if (n == 1)
        compute_singularities_anydevice<1, Float>(moments.device_type(), moments.device_id(), size, params, moments.data(), singularities.data());
    else if (n == 2)
        compute_singularities_anydevice<2, Float>(moments.device_type(), moments.device_id(), size, params, moments.data(), singularities.data());
    else if (n == 3)
        compute_singularities_anydevice<3, Float>(moments.device_type(), moments.device_id(), size, params, moments.data(), singularities.data());
    else if (n == 4)
        compute_singularities_anydevice<4, Float>(moments.device_type(), moments.device_id(), size, params, moments.data(), singularities.data());
    else if (n == 5)
        compute_singularities_anydevice<5, Float>(moments.device_type(), moments.device_id(), size, params, moments.data(), singularities.data());
    else if (n == 6)
        compute_singularities_anydevice<6, Float>(moments.device_type(), moments.device_id(), size, params, moments.data(), singularities.data());
    else
        throw std::invalid_argument(format_message("compute_singularities only supports <= %d moments (n <= %d), but received %d moments (n = %d)", 6 * 2 + 1, 6, num_moments, n));
}

template<typename Float>
std::vector<Float> get_moment_bias_vector(unsigned int n)
{
#define MAYBE_FILL_VECTOR(N) \
if (n == N) \
    for (Float const& v : dm::moment_bias_vector<N, Float>) \
        bias_vector.push_back(v);

    std::vector<Float> bias_vector;
    MAYBE_FILL_VECTOR(1);
    MAYBE_FILL_VECTOR(2);
    MAYBE_FILL_VECTOR(3);
    MAYBE_FILL_VECTOR(4);
    MAYBE_FILL_VECTOR(5);
    MAYBE_FILL_VECTOR(6);
	
    if (n > 6)
        throw std::invalid_argument("Bias vector only available for n in [1, 2, 3, 4, 5, 6].");

    return bias_vector;
}

NB_MODULE(diffmoments_ext, m)
{
#if NDEBUG
    std::string build_type = "Release";
#else
    std::string build_type = "Debug";
#endif

    m.def("build_type", [=]() { return build_type; });

    nb::enum_<ComputeMomentBoundsFlags>(m, "ComputeMomentBoundsFlags", nb::is_arithmetic())
        .value("RetainRoots", ComputeMomentBoundsFlags::RetainRoots);
        // RetainWeights is not really an option, but only used internally.

    m.def("compute_moment_bounds_f32", compute_moment_bounds<float>, nb::arg("moments"), nb::arg("etas"), nb::arg("bounds"), nb::arg("storage"), nb::arg("bias"), nb::arg("overestimation_weight"), nb::arg("flags"));
    m.def("compute_moment_bounds_f64", compute_moment_bounds<double>, nb::arg("moments"), nb::arg("etas"), nb::arg("bounds"), nb::arg("storage"), nb::arg("bias"), nb::arg("overestimation_weight"), nb::arg("flags"));

    m.def("get_required_backward_storage_size_f32", get_required_backward_storage_size<float>,  nb::arg("moments"), nb::arg("flags"));
    m.def("get_required_backward_storage_size_f64", get_required_backward_storage_size<double>, nb::arg("moments"), nb::arg("flags"));

    m.def("compute_moment_bounds_backward_f32", compute_moment_bounds_backward<float>, nb::arg("moments"), nb::arg("etas"), nb::arg("bounds"), nb::arg("storage"), nb::arg("dmoments"), nb::arg("detas"), nb::arg("dbounds"), nb::arg("bias"), nb::arg("overestimation_weight"), nb::arg("flags"));
    m.def("compute_moment_bounds_backward_f64", compute_moment_bounds_backward<double>, nb::arg("moments"), nb::arg("etas"), nb::arg("bounds"), nb::arg("storage"), nb::arg("dmoments"), nb::arg("detas"), nb::arg("dbounds"), nb::arg("bias"), nb::arg("overestimation_weight"), nb::arg("flags"));

    m.def("detect_bound_errors_f32", detect_bound_errors<float>, nb::arg("bounds"));
    m.def("detect_bound_errors_f64", detect_bound_errors<double>, nb::arg("bounds"));

    m.def("compute_singularities_f32", compute_singularities<float>, nb::arg("moments"), nb::arg("singularities"),nb::arg("bias"));
    m.def("compute_singularities_f64", compute_singularities<double>, nb::arg("moments"), nb::arg("singularities"), nb::arg("bias"));

    m.def("compute_moment_bounds_and_roots_f32", compute_moment_bounds_and_roots<float>, nb::arg("moments"), nb::arg("etas"), nb::arg("bounds"), nb::arg("roots"), nb::arg("weights"), nb::arg("bias"), nb::arg("overestimation_weight"));
    m.def("compute_moment_bounds_and_roots_f64", compute_moment_bounds_and_roots<double>, nb::arg("moments"), nb::arg("etas"), nb::arg("bounds"), nb::arg("roots"), nb::arg("weights"), nb::arg("bias"), nb::arg("overestimation_weight"));

    m.def("get_moment_bias_vector_f32", get_moment_bias_vector<float>, nb::arg("n"));
    m.def("get_moment_bias_vector_f64", get_moment_bias_vector<double>, nb::arg("n"));

    if (init_cuda())
        load_module();

    // TODO: Shutdown CUDA when the Python module is unloaded
}