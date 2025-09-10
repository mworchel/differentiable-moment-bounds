# Test the different interfaces (torch, numpy, ...)

from pathlib import Path
import pytest
import numpy as np

TEST_BIAS = 1e-3
TEST_OVERESTIMATION_WEIGHT = 0.5 # Set to a value (0, 1) so that the lower and upper bounds contribute
DATA_DIR = Path(__file__).parent / "data"

def get_test_data_file(precision: int, n: int, ):
    return DATA_DIR / f"f{precision}_n{n}.npz"

def maybe_skip_framework_test(framework: str, device: str):
    pytest.importorskip(framework)

    if framework == 'numpy' and device != 'cpu':
        pytest.skip("Numpy only supports CPU tests.")

    if framework == 'torch':
        import torch
        if device == 'cuda' and not torch.cuda.is_available():
            pytest.skip("PyTorch CUDA backend is unavailable.")

    if framework == 'drjit':
        import drjit as dr
        if device == 'cuda' and not dr.has_backend(dr.JitBackend.CUDA):
            pytest.skip("Dr.Jit CUDA backend is unavailable.")
        if device == 'cpu' and not dr.has_backend(dr.JitBackend.LLVM):
            pytest.skip("Dr.Jit LLVM backend is unavailable.")

def get_drjit_types(device: str, precision: str):
    import drjit as dr
    type_map = {
        ('cpu', '32'): (dr.llvm.ad.TensorXf, dr.llvm.ad.Float),
        ('cpu', '64'): (dr.llvm.ad.TensorXf64, dr.llvm.ad.Float64),
        ('cuda', '32'): (dr.cuda.ad.TensorXf, dr.cuda.ad.Float),
        ('cuda', '64'): (dr.cuda.ad.TensorXf64, dr.cuda.ad.Float64),
    }
    return type_map[(device, precision)]

def compute_numerical_derivative(func, x, h=0.000001):
    """ 
    Compute the derivative of a function with finite differences.
    """
    f0 = func(x - h)
    f1 = func(x + h)
    return (f1 - f0)/(2*h)

@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6])
@pytest.mark.parametrize("precision", ["32", "64"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("framework", ["drjit", "numpy", "torch"])
def test_compute_moment_bounds_primal(framework: str, device: str, precision: str, n: int):
    maybe_skip_framework_test(framework, device)

    data = np.load(get_test_data_file(precision, n))
    m = data['m']
    η = data['η']
    bounds_ref = data['bounds']
    
    if framework == 'torch':
        import torch
        import diffmoments.torch as dm
        m_torch = torch.from_numpy(m).to(device)
        η_torch = torch.from_numpy(η).to(device)
        bounds = dm.compute_moment_bounds(m_torch, η_torch, bias=TEST_BIAS, overestimation_weight=TEST_OVERESTIMATION_WEIGHT).cpu().numpy()
    elif framework == 'numpy':
        import diffmoments.np as dm
        bounds = dm.compute_moment_bounds(m, η, bias=TEST_BIAS, overestimation_weight=TEST_OVERESTIMATION_WEIGHT)
    elif framework == 'drjit':
        import drjit as dr
        import diffmoments.drjit as dm
        TensorXf, Float = get_drjit_types(device, precision)
        m_drjit = TensorXf(m)
        η_drjit = Float(η)
        bounds = dm.compute_moment_bounds(m_drjit, η_drjit, bias=TEST_BIAS, overestimation_weight=TEST_OVERESTIMATION_WEIGHT).numpy()

    # Increase the tolerance for higher n on the CPU since the reference was generated on the GPU.
    # Use the numpy default tolerances for all other tests.
    tolerance_map = {
        ('cpu', 3): (1e-4, 1e-7),
        ('cpu', 4): (1e-4, 1e-7),
        ('cpu', 5): (1e-3, 1e-3),
        ('cpu', 6): (1e-3, 1e-3),
    }
    rtol, atol = tolerance_map.get((device, n), (1e-05, 1e-08))

    assert np.allclose(bounds, bounds_ref, rtol=rtol, atol=atol)

@pytest.mark.parametrize("framework", ["drjit", "numpy", "torch"]) 
def test_compute_moment_bounds_input_validation(framework: str):
    # TODO: Split into multiple tests

    pytest.importorskip(framework)

    precision = "32"
    if framework == 'torch':
        import torch
        import diffmoments.torch as dm
        def framework_supports_mixing_devices():
            return torch.cuda.is_available()
        def zeros(shape, device=None):
            arr = torch.zeros(shape, dtype=torch.float32)
            if device is not None:
                arr = arr.to(device)
            return arr
        def call_and_maybe_reraise_cause(func, *args, **kwargs):
            func(*args, **kwargs)
    elif framework == 'numpy':
        import numpy as np
        import diffmoments.np as dm
        def framework_supports_mixing_devices():
            return False
        def zeros(shape, device=None):
            return np.zeros(shape, dtype=np.float32)
        def call_and_maybe_reraise_cause(func, *args, **kwargs):
            func(*args, **kwargs)
    elif framework == 'drjit':
        import drjit as dr
        import diffmoments.drjit as dm
        # Select an appropriate device
        if dr.has_backend(dr.JitBackend.CUDA):
            TensorXf, _ = get_drjit_types('cuda', precision)
        elif dr.has_backend(dr.JitBackend.LLVM):
            TensorXf, _ = get_drjit_types('cpu', precision)
        else:
            pytest.skip("Dr.Jit is not available.")
        def framework_supports_mixing_devices():
            return False
        def zeros(shape, device=None):
            return dr.zeros(TensorXf, shape)
        def call_and_maybe_reraise_cause(func, *args, **kwargs):
            # Some Dr.Jit interface functions wrap calls to the C++ function in a CustomOp, 
            # which itself raises a RuntimeError with the original error as the cause.
            # Re-raise the original error:
            try:
                func(*args, **kwargs)
            except RuntimeError as e:
                raise e.__cause__

    # 1. no moments provided
    expected_substr = "where 0 < num_moments"
    m = zeros((0, 10))
    η = zeros((10,))
    with pytest.raises(ValueError, match=expected_substr):
        call_and_maybe_reraise_cause(dm.compute_moment_bounds, m, η, bias=TEST_BIAS, overestimation_weight=TEST_OVERESTIMATION_WEIGHT)
    with pytest.raises(ValueError, match=expected_substr):
        dm.compute_singularities(m, bias=TEST_BIAS)

    # # 2. TODO: odd number of moments

    # 3. moments and η have different sizes
    expected_substr = "to have the same size"
    m = zeros((3, 5))
    η = zeros((10,))
    with pytest.raises(ValueError, match=expected_substr):
        call_and_maybe_reraise_cause(dm.compute_moment_bounds, m, η, bias=TEST_BIAS, overestimation_weight=TEST_OVERESTIMATION_WEIGHT)

    # 4. device mismatch of input arguments
    if framework_supports_mixing_devices():
        expected_substr = "must be on the same device"
        m = zeros((3, 10), device='cpu')
        η = zeros((10,), device='cuda')
        with pytest.raises(ValueError, match=expected_substr):
            call_and_maybe_reraise_cause(dm.compute_moment_bounds, m, η, bias=TEST_BIAS, overestimation_weight=TEST_OVERESTIMATION_WEIGHT)


@pytest.mark.parametrize("root_mode", ['recompute', 'retain'])
@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6])
@pytest.mark.parametrize("precision", ["32", "64"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("framework", ["drjit", "numpy", "torch"])
def test_compute_moment_bounds_backward(framework: str, device: str, precision: str, n: int, root_mode: str):
    maybe_skip_framework_test(framework, device)

    if n > 3 and precision != "64":
        pytest.skip(f"Numerical precision of the derivatives is insufficient for n > 3 ({precision}-bit).")

    data = np.load(get_test_data_file(precision, n))
    m = data['m']
    η = data['η']

    m_f64 = m.astype(np.float64)
    η_f64 = η.astype(np.float64)

    # Compute reference derivative, measuring the effect of the moments and η on the bound, using finite differences (in double precision).
    from diffmoments.np import compute_moment_bounds as compute_moment_bounds_np
    dbound_dm_ref = np.zeros_like(m_f64)
    dbound_dη_ref = np.zeros_like(η_f64)
    fd_eps = 1e-10
    for i in range(m.shape[0]): 
        def compute_bounds_mi(mi: np.ndarray):
            m_ = m_f64.copy()
            m_[i, :] = mi
            return compute_moment_bounds_np(m_, η_f64, bias=TEST_BIAS, overestimation_weight=TEST_OVERESTIMATION_WEIGHT)
        dbound_dm_ref[i, :] = compute_numerical_derivative(compute_bounds_mi, m_f64[i, :], fd_eps)
        dbound_dη_ref = compute_numerical_derivative(lambda η_: compute_moment_bounds_np(m_f64, η_, bias=TEST_BIAS, overestimation_weight=TEST_OVERESTIMATION_WEIGHT), η_f64, h=fd_eps)

    retain_roots = root_mode == 'retain'

    if framework == 'torch':
        import torch
        import diffmoments.torch as dm
        m_torch = torch.from_numpy(m).to(device)
        η_torch = torch.from_numpy(η).to(device)
        m_torch.requires_grad_(True)
        η_torch.requires_grad_(True)
        # Since the output is a single scalar, the derivatives can simply be computed using `backward`
        bounds = dm.compute_moment_bounds(m_torch, η_torch, bias=TEST_BIAS, overestimation_weight=TEST_OVERESTIMATION_WEIGHT, retain_roots=retain_roots)
        bounds.sum().backward()
        dbound_dm = m_torch.grad.cpu().numpy().astype(np.float64)
        dbound_dη = η_torch.grad.cpu().numpy().astype(np.float64)
    elif framework == 'numpy':
        import diffmoments.np as dm
        bounds, ctx = dm.compute_moment_bounds(m, η, bias=TEST_BIAS, overestimation_weight=TEST_OVERESTIMATION_WEIGHT, retain_roots=retain_roots, return_backward_context=True)
        δbounds = np.ones_like(bounds)
        dbound_dm, dbound_dη = dm.compute_moment_bounds_backward(ctx, δbounds)
        dbound_dm = dbound_dm.astype(np.float64)
        dbound_dη = dbound_dη.astype(np.float64)
    elif framework == 'drjit':
        import drjit as dr
        import diffmoments.drjit as dm
        TensorXf, Float = get_drjit_types(device, precision)
        m_drjit = TensorXf(m)
        η_drjit = Float(η)
        dr.enable_grad(m_drjit, η_drjit)
        bounds = dm.compute_moment_bounds(m_drjit, η_drjit, bias=TEST_BIAS, overestimation_weight=TEST_OVERESTIMATION_WEIGHT, retain_roots=retain_roots)
        dr.backward_from(dr.sum(bounds))
        dbound_dm = dr.grad(m_drjit).numpy().astype(np.float64)
        dbound_dη = dr.grad(η_drjit).numpy().astype(np.float64)

    # TODO: The tolerance is a little high, but necessary for `n` = 6
    assert np.allclose(dbound_dm_ref, dbound_dm, rtol=1e-3, atol=9e-2), f"Max abs. diff: {np.abs(dbound_dm_ref - dbound_dm).max()}"
    assert np.allclose(dbound_dη_ref, dbound_dη, rtol=1e-4, atol=1e-4)

@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6])
@pytest.mark.parametrize("precision", ["32", "64"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("framework", ["drjit", "numpy", "torch"])
def test_detect_bound_errors(framework: str, device: str, precision: str, n: int):
    maybe_skip_framework_test(framework, device)

    # Generate non-positive moment vectors

    # 1. Sample a random point on the moment curve (which by definition lies on the moment cone)
    rng = np.random.default_rng(0)
    x = rng.normal(0, 10, size=(100000))
    u_x = np.stack([x**i for i in range(2*n+1)], axis=0)

    # 2. Nudge the point slightly outside of the moment cone (using the bias)
    import diffmoments_ext as dmx
    m_bias = np.array(dmx.get_moment_bias_vector_f64(n))[:, None]
    t = -1
    m = (1-t)*u_x + t*m_bias
    
    # The evaluation point doesn't matter.
    η = np.zeros_like(m[0, :])

    m = m.astype(np.float32) if precision == "32" else m.astype(np.float64)
    η = η.astype(np.float32) if precision == "32" else η.astype(np.float64)

    call_check: callable = None
    if framework == 'torch':
        import torch
        import diffmoments.torch as dm
        m_torch = torch.from_numpy(m).to(device)
        η_torch = torch.from_numpy(η).to(device)
        bounds = dm.compute_moment_bounds(m_torch, η_torch, bias=0, overestimation_weight=0)
        bounds_torch = bounds.clone()
        call_check = lambda: dm.detect_bound_errors(bounds_torch)
        bounds = bounds.cpu().numpy()
    elif framework == 'numpy':
        import diffmoments.np as dm
        bounds = dm.compute_moment_bounds(m, η, bias=0, overestimation_weight=0)
        call_check = lambda: dm.detect_bound_errors(bounds)
    elif framework == 'drjit':
        import drjit as dr
        import diffmoments.drjit as dm
        TensorXf, Float = get_drjit_types(device, precision)
        m_drjit = TensorXf(m)
        η_drjit = Float(η)
        bounds = dm.compute_moment_bounds(m_drjit, η_drjit, bias=0, overestimation_weight=0)
        bounds_drjit = Float(bounds)
        call_check = lambda: dm.detect_bound_errors(bounds_drjit)
        bounds = bounds.numpy()

    assert np.all(np.isnan(bounds))

    # Check if the Cholesky decomposition fails
    with pytest.raises(RuntimeError, match="The Cholesky decomposition of the Hankel matrix failed"):
        call_check()

@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("precision", ["32", "64"])
@pytest.mark.parametrize("framework", ["drjit", "numpy", "torch"])
def test_compute_singularities(framework: str, precision: str, device: str, n: int):
    maybe_skip_framework_test(framework, device)

    data = np.load(get_test_data_file(precision, n))
    m = data['m']
    singularities_ref = data['singularities']

    if framework == 'torch':
        import torch
        import diffmoments.torch as dm
        m_torch = torch.from_numpy(m).to(device)
        singularities = dm.compute_singularities(m_torch, bias=TEST_BIAS).cpu().numpy()
    elif framework == 'numpy':
        import diffmoments.np as dm
        singularities = dm.compute_singularities(m, bias=TEST_BIAS)
    elif framework == 'drjit':
        import diffmoments.drjit as dm
        TensorXf, _ = get_drjit_types(device, precision)
        m_drjit = TensorXf(m)
        singularities = dm.compute_singularities(m_drjit, bias=TEST_BIAS).numpy()

    if device == 'cpu' and precision == "32":
        # TODO: The tolerances are quite high (tuned for high `n`)
        atol, rtol = 2e-3, 2e-3
    else:
        atol, rtol = 1e-8, 1e-5

    assert np.allclose(singularities, singularities_ref, atol=atol, rtol=rtol)

@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6])
@pytest.mark.parametrize("precision", ["64"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("framework", ["torch"])
def test_compute_moment_bounds_and_roots(framework: str, device: str, precision: str, n: int):
    maybe_skip_framework_test(framework, device)

    data = np.load(get_test_data_file(precision, n))
    m = data['m']
    η = data['η']
    
    if framework == 'torch':
        import torch
        import diffmoments.torch as dm
        m_torch = torch.from_numpy(m).to(device)
        η_torch = torch.from_numpy(η).to(device)
        _, roots, weights = dm.compute_moment_bounds_and_roots(m_torch, η_torch, bias=0, overestimation_weight=TEST_OVERESTIMATION_WEIGHT)
        roots = roots.cpu().numpy()
        weights = weights.cpu().numpy()

    # Evaluate the moment curve at the roots and reconstruct the moments
    u_x = np.stack([roots**i for i in range(2*n+1)], axis=0)
    m_reconstructed = (u_x * weights[None, :, :]).sum(axis=1)

    assert np.allclose(m[:n+1], m_reconstructed[:n+1])

if __name__ == "__main__":
    """
    Generate reference data for all tests
    """
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    import torch

    from diffmoments.torch import generate_moment_vectors, compute_singularities, compute_moment_bounds

    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--num_samples', default=10000, type=int,
                        help='Number of test samples')
    args = parser.parse_args()

    device = torch.device('cuda:0')

    DATA_DIR.mkdir(exist_ok=True, parents=True)

    for dtype in [torch.float32, torch.float64]:
        for n in [1, 2, 3, 4, 5, 6]:
            torch.manual_seed(n)
            m = generate_moment_vectors(n, args.num_samples, 5*n+5, dtype=dtype, device=device)
            η = torch.randn(args.num_samples, dtype=dtype, device=device)

            bounds = compute_moment_bounds(m, η, bias=TEST_BIAS, overestimation_weight=TEST_OVERESTIMATION_WEIGHT)

            singularities = compute_singularities(m, bias=TEST_BIAS).to(device)

            precision = 8*dtype.itemsize
            np.savez(get_test_data_file(precision, n), 
                     m=m.cpu().numpy(), 
                     η=η.cpu().numpy(), 
                     bounds=bounds.cpu().numpy(), 
                     singularities=singularities.cpu().numpy())

            if not bounds.isfinite().all():
                print("Generated non-finite bounds:", bounds.isfinite().sum())

            if not singularities.isfinite().all():
                print("Generated non-finite singularities:", singularities.isfinite().sum())