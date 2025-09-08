import torch
from typing import Any

import diffmoments_ext as dmx
from diffmoments.extension import get_flags

__all__ = ["compute_moment_bounds", "compute_moment_bounds_and_roots", "detect_bound_errors", "compute_singularities"]

class ComputeMomentBoundsFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, moments: torch.Tensor, η: torch.Tensor, bias: float, overestimation_weight: float, retain_roots: bool):
        flags = get_flags(retain_roots)

        # Allocate the temporary storage for reverse-mode differentiation
        storage_size = 0
        if moments.dtype == torch.float32:
            storage_size = dmx.get_required_backward_storage_size_f32(moments, flags)
        elif moments.dtype == torch.float64:
            storage_size = dmx.get_required_backward_storage_size_f64(moments, flags)
        storage = torch.empty((storage_size,), dtype=moments.dtype, device=moments.device)

        bounds = torch.empty((moments.shape[1],), dtype=moments.dtype, device=moments.device)
        if moments.dtype == torch.float32:
            dmx.compute_moment_bounds_f32(moments, η, bounds, storage, bias, overestimation_weight, flags)
        elif moments.dtype == torch.float64:
            dmx.compute_moment_bounds_f64(moments, η, bounds, storage, bias, overestimation_weight, flags)
        else:
            raise RuntimeError(f"Invalid dtype '{moments.dtype}'. Must be torch.float32 or torch.float64.")

        ctx.save_for_backward(moments, η, bounds, storage)
        ctx.bias = bias
        ctx.overestimation_weight = overestimation_weight
        ctx.flags = flags

        return bounds
    
    @staticmethod
    def backward(ctx: Any, δbounds: torch.Tensor):
        moments, η, bounds, storage = ctx.saved_tensors
        δmoments = torch.empty(moments.shape, dtype=moments.dtype, device=moments.device)
        δη = torch.empty(η.shape, dtype=η.dtype, device=η.device)

        if moments.dtype == torch.float32:
            dmx.compute_moment_bounds_backward_f32(moments, η, bounds, storage, δmoments, δη, δbounds, ctx.bias, ctx.overestimation_weight, ctx.flags)
        elif moments.dtype == torch.float64:
            dmx.compute_moment_bounds_backward_f64(moments, η, bounds, storage, δmoments, δη, δbounds, ctx.bias, ctx.overestimation_weight, ctx.flags)
        else:
            raise RuntimeError(f"Invalid dtype '{moments.dtype}'. Must be torch.float32 or torch.float64.")

        return δmoments, δη, None, None, None

def compute_moment_bounds(moments: torch.Tensor, η: torch.Tensor, bias: float, overestimation_weight: float, retain_roots: bool = False):
    """
    Compute the bound for a designated point `η`, given the moments.

    Parameters
    ----------
    moments : torch.Tensor
        Moments, non-interleaved, as an array of shape (2n+1,N).
    η : torch.Tensor
        Designated point of the canonical representation, as an array of shape (N).
    bias : float
        Bias value in [0, 1].
    overestimation_weight : float
        Blend weight in [0, 1] between lower (=0) and upper (=1) bound.
    retain_roots : bool, optional
        Use `retained roots` mode, which stores the roots of the canonical representation for the backward pass (recommended for `n` >= 3).

    Returns
    -------
    torch.Tensor
        The bounds as an array of shape (N).
    """
    return ComputeMomentBoundsFunc.apply(moments, η, bias, overestimation_weight, retain_roots)

def compute_moment_bounds_and_roots(moments: torch.Tensor, η: torch.Tensor, bias: float, overestimation_weight: float):
    """
    Compute the bound and the canonical representation (roots and weights) for a designated point `η`, given the moments.

    This function is for debugging only. No backward pass implemented.

    Parameters
    ----------
    moments : torch.Tensor
        Moments, non-interleaved, as an array of shape (2n+1,N).
    η : torch.Tensor
        Designated point of the canonical representation, as an array of shape (N).
    bias : float
        Bias value in [0, 1].
    overestimation_weight : float
        Blend weight in [0, 1] between lower (=0) and upper (=1) bound.

    Returns
    -------
    bounds
        The bounds as an array of shape (N).
    roots
        The roots of the canonical representation as an array of shape (n+1,N).
    weights
        The weights of the canonical representation as an array of shape (n+1,N).
    """

    # Do not handle the empty case here but pass along to the extension
    n = (moments.shape[0] - 1) // 2 if moments.shape[0] > 0 else 0

    bounds = torch.empty_like(moments[0])
    roots = torch.empty_like(moments[:n+1])
    weights = torch.empty_like(roots)
    if moments.dtype == torch.float32:
        dmx.compute_moment_bounds_and_roots_f32(moments, η, bounds, roots, weights, bias, overestimation_weight)
    elif moments.dtype == torch.float64:
        dmx.compute_moment_bounds_and_roots_f64(moments, η, bounds, roots, weights, bias, overestimation_weight)

    return bounds, roots, weights

def detect_bound_errors(bounds: torch.Tensor):
    """
    Check if any of the given bounds is invalid and if so throw an exception.

    Use for debugging purposes only.

    Parameters
    ----------
    bounds : torch.Tensor
        The bounds as an array of shape (N).
    """

    if bounds.dtype == torch.float32:
        dmx.detect_bound_errors_f32(bounds)
    elif bounds.dtype == torch.float64:
        dmx.detect_bound_errors_f64(bounds)
    else:
        raise RuntimeError(f"Invalid dtype '{bounds.dtype}'. Must be torch.float32 or torch.float64.")        
    
def compute_singularities(moments: torch.Tensor, bias: float):
    """
    Compute the singularity points (roots of the polynomial `P_n`) from the given moments.

    Note that this function is not necessarily optimized.

    Parameters
    ----------
    moments : torch.Tensor
        Moments, non-interleaved, as an array of shape (2n+1,N).
    bias : float
        Bias value in [0, 1].

    Returns
    -------
    singularities : torch.Tensor
        Singularity points, as an array of shape (n,N).
    """
    # Do not handle the empty case here but pass along to the extension
    n = (moments.shape[0] - 1) // 2 if moments.shape[0] > 0 else 0

    singularities  = torch.empty((n, moments.shape[1]), device=moments.device, dtype=moments.dtype)
    if moments.dtype == torch.float32:
        dmx.compute_singularities_f32(moments, singularities, bias)
    elif moments.dtype == torch.float64:
        dmx.compute_singularities_f64(moments, singularities, bias)
    else:
        raise RuntimeError(f"Invalid dtype '{moments.dtype}'. Must be torch.float32 or torch.float64.")

    return singularities