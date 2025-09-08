import copy
import torch
from typing import Optional
import warnings

__all__ = ["compute_moment_bounds"]

def compute_moment_bounds(moments: torch.Tensor, etas: torch.Tensor, bias: float, overestimation_weight: float, return_roots: bool = False, use_closed_form: bool = False):
    """
    Compute bounds for the given moments and etas.

    This function is entirely based on PyTorch, so the derivatives are computed using PyTorch's autograd engine.

    Parameters
    ----------
    moments : torch.Tensor
        Moments, interleaved in a tensor of shape (N, 2n+1).
    etas : torch.Tensor
        Evaluation points (designated point of the canonical representation), tensor of shape (N).
    bias : float
        Bias value in [0, 1].
    overestimation_weight : float
        The blend weight in [0, 1] between lower (=0) and upper (=1) bound.
    return_roots : bool, optional
        Whether to return the roots of the canonical representation and their respective weights (default is False).
    use_closed_form : bool, optional
        Whether to use a closed-form solution for N=1 (3 moments) (default is False).

    Returns
    -------
    torch.Tensor or tuple
        If `return_roots` is False, returns the bounds as a tensor.
        If `return_roots` is True, returns a tuple of (bounds, roots, weights).
    """

    # Avoid circular import
    from diffmoments.torch import compute_moment_bias, build_hankel_matrix, moment_curve, build_vandermonde_matrix, PolynomialRootFindingFunc

    N = (moments.shape[-1] - 1) // 2

    # Bias the moment vector
    if bias > 0:
        with torch.profiler.record_function("Moment Biasing"):
            if N == 1:
                moment_bias_vector = torch.tensor([1, 0, 1], dtype=moments.dtype, device=moments.device)
            # For N = 2,...,4 take bias vectors from the supplementary material of
            # Münstermann et al. (2018) Moment-Based Order-Independent Transparency
            elif N == 2: 
                moment_bias_vector = torch.tensor([1, 0, 0.375, 0, 0.375], dtype=moments.dtype, device=moments.device)
            elif N == 3:
                moment_bias_vector = torch.tensor([1, 0, 0.48, 0, 0.451, 0, 0.45], dtype=moments.dtype, device=moments.device)
            elif N == 4:
                moment_bias_vector = torch.tensor([1, 0, 0.75, 0, 0.676666667, 0, 0.63, 0, 0.60030303], dtype=moments.dtype, device=moments.device)
            else:
                # For N > 4, use a less principled method
                moment_bias_vector = compute_moment_bias(N)[0].to(dtype=moments.dtype, device=moments.device) 

            moments = (1-bias) * moments + bias * moment_bias_vector[None, :]

    # Closed-form solution, similar to VSM
    # FIXME: Currently untested. 
    if use_closed_form and N == 1:
        warnings.warn("compute_moment_bounds: the closed-form solution is untested for the PyTorch AD variant.")
        with torch.profiler.record_function("Moment Normalization"):
            m0 = moments[:, 0]
            m1 = moments[:, 1] / m0
            m2 = moments[:, 2] / m0

        # TODO: Can variance be 0 after biasing?

        # Visibility function of Variance Shadow Mapping based on the Chebyshev-Cantelli inequality
        with torch.profiler.record_function("Moment Chebyshev"):
            mean     = m1
            variance = m2 - m1*m1
            a = (etas - mean)**2
            bound = torch.where(etas <= mean, 0, a / (variance + a))

        with torch.profiler.record_function("Moment Denormalization"):
            bound = m0 * bound

        if return_roots:
            print("Warning: `return_roots` is not implemented for the VSM fast path.")
            return (bound, None, None)
        else:
            return bound

    # Build Hankel matrix
    with torch.profiler.record_function("Moment Solve Coefficients"):
        H = build_hankel_matrix(moments)
        
        # Solve the system using Cholesky, to get the polynomial coefficients
        L = torch.linalg.cholesky(H)
        #L, info = torch.linalg.cholesky_ex(H)
        u_eta = moment_curve(etas, N)
        coeffs = torch.cholesky_solve(u_eta[..., None], L)[..., 0]

    # TODO: Clamp the leading coefficient of c by magnitude
    # TODO: Remove
    # ε = max(min(ε, 1), 0)
    # cn = coeffs[..., -1:]
    # cn = torch.where(cn < 0, cn.clamp_max(-ε), cn.clamp_min(ε))
    # coeffs = torch.cat([coeffs[..., :-1], cn], dim=-1)

    with torch.profiler.record_function("Moment Find Roots"):
        roots = PolynomialRootFindingFunc.apply(coeffs)
        roots = torch.concatenate([etas[:, None], roots], dim=-1) # (B,N+1)

    with torch.profiler.record_function("Moment Solve Weights"):
        Vt  = build_vandermonde_matrix(roots).transpose(-1, -2)      # (B,N+1,N+1)
        w = torch.linalg.solve(Vt, moments[..., :N+1, None])[..., 0] # (B,N+1)

    with torch.profiler.record_function("Moment Sum Weights"):
        # TODO: Could directly integrate the mask into the `solve` above
        # w[..., 0] *= β
        mask = roots <= etas[:, None]

        Beta = torch.ones_like(w)
        Beta[..., 0] = overestimation_weight
        bound = (w * Beta * mask).sum(dim=-1, keepdim=False)

    if return_roots:
        return (bound, roots, w)
    else:
        return bound