import numpy as np
import torch
from typing import Any

__all__ = ["evaluate_polynomial", "derivative_polynomial", "antiderivative_polynomial", 
           "companion_matrix", "find_polynomial_roots_forward", "find_polynomial_roots_backward",
           "PolynomialRootFindingFunc"]

def evaluate_polynomial(coefficients: torch.Tensor, x: torch.Tensor):
    """
    Evaluate an univariate polynomial efficiently using Horner's method.

    Parameters
    ----------
    coefficients : torch.Tensor
        Degree D Polynomial coefficients in ascending order (c0 + c1*x + c2*x^2 + ...), as a tensor of shape (*,D+1).
    x : torch.Tensor
        Evaluation points (*,N).

    Returns
    -------
    torch.Tensor
        Evaluation result (*,N).
    """

    b = coefficients[..., -1:]
    for i in range(1, coefficients.shape[-1]):
        b = coefficients[..., -i-1:-i] + x * b

    return b

def derivative_polynomial(coefficients: torch.Tensor):
    """
    Find the derivative of an univariate polynomial w.r.t. x.

    Parameters
    ----------
    coefficients : torch.Tensor
        Degree D Polynomial coefficients in ascending order (c0 + c1*x + c2*x^2 + ...), as a tensor of shape (*,D+1).

    Returns
    -------
    torch.Tensor
        Coefficients of the degree (D-1) derivative polynomial with shape (*,D).
    """
    N = coefficients.shape[-1]
    return torch.arange(1, N, device=coefficients.device) * coefficients[..., 1:]

def antiderivative_polynomial(coefficients: torch.Tensor, c0: float = 0):
    """
    Find the antiderivative of an univariate polynomial w.r.t. x.

    Parameters
    ----------
    coefficients : torch.Tensor
        Degree D Polynomial coefficients in ascending order (c0 + c1*x + c2*x^2 + ...), as a tensor of shape (*,D+1).
    c0 : float, optional
        Constant of integration (default is 0).

    Returns
    -------
    torch.Tensor
        Coefficients of the degree (D+1) antiderivative polynomial with shape (*,D+2).
    """
    N = coefficients.shape[-1]
    return torch.concatenate([torch.full_like(coefficients[..., :1], c0), coefficients / torch.arange(1, N+1, device=coefficients.device)], dim=-1)

def companion_matrix(coefficients: torch.Tensor):
    """
    Build the companion matrix of an univariate polynomial.

    Parameters
    ----------
    coefficients : torch.Tensor
        Degree D Polynomial coefficients in ascending order (c0 + c1*x + c2*x^2 + ...), as a tensor of shape (*,D+1).

    Returns
    -------
    torch.Tensor
        Companion matrix of shape (*,D,D).
    """

    d = coefficients.shape[-1] - 1

    C = torch.zeros((*coefficients.shape[:-1], d, d), dtype=coefficients.dtype, device=coefficients.device)
    C[..., 1:, :-1] = torch.eye(d-1, d-1, device=coefficients.device)
    C[...,  :, -1]  = -coefficients[..., :-1] / coefficients[..., -1:]

    return C

@torch.no_grad()
def find_polynomial_roots_forward(c: torch.Tensor, mode='companion'):
    """
    Find roots of a polynomial.

    Parameters
    ----------
    c : torch.Tensor
        Polynomial coefficients (*,D+1).
    mode : str, optional
        Method to find roots (default is 'companion').

    Returns
    -------
    torch.Tensor
        Roots of the polynomial (*,D).
    """
    degree = c.shape[-1] - 1

    # Fast paths
    if degree == 1:
        z = -c[..., 0:1] / c[..., 1:2]
    elif degree == 2:
        # c0, c1, c2 = c.chunk(3, dim=-1)
        # d = c1 * c1 - 4 * c2 * c0
        # r = -0.5 * (c1 + torch.sign(c1) * torch.sqrt(d))
        # z1 = c0 / r
        # z2 = r / c2
        # z = torch.concatenate([z1, z2], dim=-1)
        # z = torch.sort(z, dim=-1)[0]

        c1, c2, c3 = c.chunk(3, dim=-1)
        c3 = c3
        p  = c2/c3
        q  = c1/c3
        D = ((p*p) / 4)-q
        r = torch.sqrt(D)
        z2 = -(p / 2) - r
        z3 = -(p / 2) + r
        z = torch.concatenate([z2, z3], dim=-1)
    else: 
        if mode == 'companion':
            with torch.no_grad():
                # This follows the numpy implementation 
                # (including the flipping, which apparently improves numerical stability)
                C = torch.flip(companion_matrix(c), dims=(-1, -2))
                z = torch.linalg.eigvals(C).real
        else:
            # This should not be used.
            c_flat = c.reshape(-1, degree + 1)
            z_flat = torch.empty_like(c_flat[:, :-1])
            for i in range(len(c_flat)):
                z_flat[i] = torch.from_numpy(np.roots(c_flat[i].cpu().numpy()[::-1])).to(z_flat.device)
            z = z_flat.reshape_as(c[..., :-1])

    return z

@torch.no_grad()
def find_polynomial_roots_backward(c: torch.Tensor, z: torch.Tensor, δz: torch.Tensor):
    # Implicit differentiation
    dc   = derivative_polynomial(c)
    dfdz = evaluate_polynomial(dc, z) # Evaluate differential polynomial (...,D)

    k    = torch.arange(c.shape[-1], device=c.device)
    dfdc = torch.pow(z[..., None, :], k[..., :, None]) # z^0, z^1, z^2, ... Evaluate coefficient derivative polynomial (...,D+1,D)

    # Sum over the roots
    δc = (-dfdc * δz[..., None, :] / dfdz[..., None, :]).sum(-1)

    return δc

class PolynomialRootFindingFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, c: torch.Tensor):
        z = find_polynomial_roots_forward(c, mode='companion')
        ctx.save_for_backward(c, z)
        return z
    
    @staticmethod
    def backward(ctx: Any, δz: torch.Tensor):
        c, z = ctx.saved_tensors
        δc = find_polynomial_roots_backward(c, z, δz) 
        return δc