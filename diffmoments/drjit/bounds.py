import drjit as dr

from diffmoments.extension import get_flags

import diffmoments_ext as dmx

__all__ = ["compute_moment_bounds", "detect_bound_errors", "compute_singularities"]

class ComputeMomentBoundsOp(dr.CustomOp):
    def eval(self, moments: dr.auto.ad.TensorXf, η: dr.auto.ad.Float, bias: float, overestimation_weight: float, retain_roots: bool):
        flags = get_flags(retain_roots)

        Float    = type(η)
        var_type = dr.type_v(moments)

        # Allocate the temporary storage for reverse-mode differentiation
        storage_size = 0
        if var_type == dr.VarType.Float32:
            storage_size = dmx.get_required_backward_storage_size_f32(moments, flags)
        elif var_type == dr.VarType.Float64:
            storage_size = dmx.get_required_backward_storage_size_f64(moments, flags)
        storage = dr.empty(Float, storage_size)

        bounds = dr.empty(Float, moments.shape[1])
        if var_type == dr.VarType.Float32:
            dmx.compute_moment_bounds_f32(moments, η, bounds, storage, bias, overestimation_weight, flags)
        elif var_type == dr.VarType.Float64:
            dmx.compute_moment_bounds_f64(moments, η, bounds, storage, bias, overestimation_weight, flags)
        else:
            raise RuntimeError(f"Invalid dtype '{var_type}'. Must be dr.VarType.Float32 or dr.VarType.Float64.")

        self.moments = moments
        self.η = η
        self.bias = bias
        self.overestimation_weight = overestimation_weight

        self.flags = flags
        self.storage = storage
        self.bounds = bounds

        return bounds

    def backward(self):
        TensorXf = type(self.moments)
        Float    = type(self.η)
        var_type = dr.type_v(self.moments)

        δbounds = self.grad_out()

        δmoments = dr.empty(TensorXf, self.moments.shape)
        δη       = dr.empty(Float, self.moments.shape[1])

        if var_type == dr.VarType.Float32:
            dmx.compute_moment_bounds_backward_f32(self.moments, self.η, self.bounds, self.storage, δmoments, δη, δbounds, self.bias, self.overestimation_weight, self.flags)
        elif var_type == dr.VarType.Float64:
            dmx.compute_moment_bounds_backward_f64(self.moments, self.η, self.bounds, self.storage, δmoments, δη, δbounds, self.bias, self.overestimation_weight, self.flags)
        else:
            raise RuntimeError(f"Invalid dtype '{var_type}'. Must be dr.VarType.Float32 or dr.VarType.Float64.")   

        self.set_grad_in('moments', δmoments)
        self.set_grad_in('η', δη)

def compute_moment_bounds(moments: dr.auto.ad.TensorXf, η: dr.auto.ad.Float, bias: float, overestimation_weight: float, retain_roots: bool = False):
    """
    Compute the bound for a designated point `η`, given the moments.

    Parameters
    ----------
    moments : dr.auto.ad.TensorXf
        Moments, non-interleaved, as an array of shape (2n+1, N).
    η : dr.auto.ad.Float
        Designated point of the canonical representation, as an array of shape (N).
    bias : float
        Bias value in [0, 1].
    overestimation_weight : float
        Blend weight in [0, 1] between lower (=0) and upper (=1) bound.
    retain_roots : bool, optional
        Use `retained roots` mode, which stores the roots of the canonical representation for the backward pass (recommended for `n` >= 3).

    Returns
    -------
    dr.auto.ad.Float
        The bounds as an array of shape (N).
    """

    return dr.custom(ComputeMomentBoundsOp, moments, η, bias, overestimation_weight, retain_roots)

def detect_bound_errors(bounds: dr.auto.ad.Float):
    """
    Check if any of the given bounds is invalid and if so throw an exception.

    Use for debugging purposes only.

    Parameters
    ----------
    bounds : dr.auto.ad.Float
        The bounds as an array of shape (N).
    """

    var_type = dr.type_v(bounds)
    if var_type == dr.VarType.Float32:
        dmx.detect_bound_errors_f32(bounds)
    elif var_type == dr.VarType.Float64:
        dmx.detect_bound_errors_f64(bounds)
    else:
        raise RuntimeError(f"Invalid dtype '{var_type}'. Must be dr.VarType.Float32 or dr.VarType.Float64.")
    
def compute_singularities(moments: dr.auto.ad.TensorXf, bias: float):
    """
    Compute the singularity points (roots of the polynomial `P_n`) from the given moments.

    Note that this function is not necessarily optimized.

    Parameters
    ----------
    moments : dr.auto.ad.TensorXf
        Moments, non-interleaved, as an array of shape (2n+1,N).
    bias : float
        Bias value in [0, 1].

    Returns
    -------
    singularities : dr.auto.ad.TensorXf
        Singularity points, as an array of shape (n,N).
    """
    # Do not handle the empty case here but pass along to the extension
    n = (moments.shape[0] - 1) // 2 if moments.shape[0] > 0 else 0

    TensorXf = type(moments)
    singularities = dr.empty(TensorXf, (n, moments.shape[1]))

    var_type = dr.type_v(moments)
    if var_type == dr.VarType.Float32:
        dmx.compute_singularities_f32(moments, singularities, bias)
    elif var_type == dr.VarType.Float64:
        dmx.compute_singularities_f64(moments, singularities, bias)
    else:
        raise RuntimeError(f"Invalid dtype '{var_type}'. Must be dr.VarType.Float32 or dr.VarType.Float64.")

    return singularities