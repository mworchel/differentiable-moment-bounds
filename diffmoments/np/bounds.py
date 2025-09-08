import numpy as np

import diffmoments_ext as dmx
from diffmoments.extension import get_flags

__all__ = ["compute_moment_bounds", "compute_moment_bounds_backward", "detect_bound_errors", "compute_singularities"]

def compute_moment_bounds(moments: np.ndarray, η: np.ndarray, bias: float, overestimation_weight: float, retain_roots: bool = False, return_backward_context: bool = False):
    """
    Compute the bound for a designated point `η`, given the moments.

    Parameters
    ----------
    moments : np.ndarray
        Moments, non-interleaved, as an array of shape (2n+1, N).
    η : np.ndarray
        Designated point of the canonical representation, as an array of shape (N).
    bias : float
        Bias value in [0, 1].
    overestimation_weight : float
        Blend weight in [0, 1] between lower (=0) and upper (=1) bound.
    retain_roots : bool, optional
        Use `retained roots` mode, which stores the roots of the canonical representation for the backward pass (recommended for `n` >= 3).
    return_backward_context : bool, optional
        If True, also returns a context required for the backward pass (useful for reverse-mode differentiation).

    Returns
    -------
    np.ndarray or Tuple[np.ndarray, dict]
        The bounds as an array of shape (N). If `return_backward_context` is True, returns a tuple (bounds, ctx), where `ctx` is the backward pass context.
    """

    flags = get_flags(retain_roots)

    # Allocate the temporary storage for reverse-mode differentiation
    storage_size = 0
    if moments.dtype == np.float32:
        storage_size = dmx.get_required_backward_storage_size_f32(moments, flags)
    elif moments.dtype == np.float64:
        storage_size = dmx.get_required_backward_storage_size_f64(moments, flags)
    storage = np.empty((storage_size,), dtype=moments.dtype)

    bounds = np.empty((moments.shape[1],), dtype=moments.dtype)
    if moments.dtype == np.float32:
        dmx.compute_moment_bounds_f32(moments, η, bounds, storage, bias, overestimation_weight, flags)
    elif moments.dtype == np.float64:
        dmx.compute_moment_bounds_f64(moments, η, bounds, storage, bias, overestimation_weight, flags)
    else:
        raise RuntimeError(f"Invalid dtype '{moments.dtype}'. Must be np.float32 or np.float64.")

    if return_backward_context:
        ctx = {}
        ctx['moments'] = moments
        ctx['η'] = η
        ctx['bounds'] = bounds
        ctx['storage'] = storage
        ctx['bias'] = bias
        ctx['overestimation_weight'] = overestimation_weight
        ctx['flags'] = flags
        return bounds, ctx
    else:
        return bounds

def compute_moment_bounds_backward(ctx: dict, δbounds: np.ndarray):
    moments = ctx['moments']
    η = ctx['η']
    bounds = ctx['bounds']
    storage = ctx['storage']
    
    δmoments = np.empty(moments.shape, dtype=moments.dtype)
    δη    = np.empty(η.shape, dtype=η.dtype)

    if moments.dtype == np.float32:
        dmx.compute_moment_bounds_backward_f32(moments, η, bounds, storage, δmoments, δη, δbounds, ctx['bias'], ctx['overestimation_weight'], ctx['flags'])
    elif moments.dtype == np.float64:
        dmx.compute_moment_bounds_backward_f64(moments, η, bounds, storage, δmoments, δη, δbounds, ctx['bias'], ctx['overestimation_weight'], ctx['flags'])
    else:
        raise RuntimeError(f"Invalid dtype '{moments.dtype}'. Must be np.float32 or np.float64.")

    return δmoments, δη

def detect_bound_errors(bounds: np.ndarray):
    """
    Check if any of the given bounds is invalid and if so throw an exception.

    Use for debugging purposes only.

    Parameters
    ----------
    bounds : np.ndarray
        The bounds as an array of shape (N).
    """

    if bounds.dtype == np.float32:
        dmx.detect_bound_errors_f32(bounds)
    elif bounds.dtype == np.float64:
        dmx.detect_bound_errors_f64(bounds)
    else:
        raise RuntimeError(f"Invalid dtype '{bounds.dtype}'. Must be np.float32 or np.float64.")

def compute_singularities(moments: np.ndarray, bias: float):
    """
    Compute the singularity points (roots of the polynomial `P_n`) from the given moments.

    Note that this function is not necessarily optimized.

    Parameters
    ----------
    moments : np.ndarray
        Moments, non-interleaved, as an array of shape (2n+1, N).
    bias : float
        Bias value in [0, 1].

    Returns
    -------
    singularities : np.ndarray
        Singularity points, as an array of shape (n, N).
    """

    # Do not handle the empty case here but pass along to the extension
    n = (moments.shape[0] - 1) // 2 if moments.shape[0] > 0 else 0

    singularities  = np.empty((n, moments.shape[1]), dtype=moments.dtype)
    if moments.dtype == np.float32:
        dmx.compute_singularities_f32(moments, singularities, bias)
    elif moments.dtype == np.float64:
        dmx.compute_singularities_f64(moments, singularities, bias)
    else:
        raise RuntimeError(f"Invalid dtype '{moments.dtype}'. Must be np.float32 or np.float64.")

    return singularities