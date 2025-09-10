import gc
import math
import torch
from typing import Optional

__all__ = ["build_hankel_matrix", "get_moments_from_hankel_matrix", "build_vandermonde_matrix", 
           "moment_curve", "compute_moment_bias", "generate_moment_vectors"]

def build_hankel_matrix(moments: torch.Tensor, n: Optional[int] = None):
    """
    Build the (partial) Hankel matrix from an array of moments.

    Parameters
    ----------
    moments : torch.Tensor
        Array of moments in ascending order m0, m1, ... (*,m+1).
    n : int, optional
        If specified, build a partial Hankel matrix with 2n being the maximum moment index.
        The following condition must hold: 2n <= m.

    Returns
    -------
    torch.Tensor
        The Hankel matrix.
    """
    m = moments.shape[-1] - 1

    if n is not None:
        # Construct the reduced Hankel matrix if requested
        assert 2*n <= m, f"2n (2*{n} = {2*n}) must be smaller than m ({m})"
        m = 2*n

    # Generate indices into the moment array for each row of the Hankel matrix
    # [0       1 ... m/2  ]                   [0 1 2]
    # [1       2 ... m/2+1], for example m=4: [1 2 3]  
    # [|               |  ]                   [2 3 4]
    # [m/2 m/2+1 ...   m  ]
    i = torch.arange(0, m//2 + 1, device=moments.device)
    hankel_indices = i[None,:] + i[:,None]

    # The hankel indices to be used for gather must match the moment dimensions
    hankel_indices = hankel_indices.expand((*moments.shape[:-1], -1, -1))

    hankel_matrix = torch.gather(moments, -1, hankel_indices.flatten(start_dim=-2)).reshape_as(hankel_indices)

    return hankel_matrix

def get_moments_from_hankel_matrix(H: torch.Tensor):
    """
    Extract the array of moments from a Hankel matrix.

    Parameters
    ----------
    H : torch.Tensor
        Hankel matrix, as a tensor of shape (*,n,n).

    Returns
    -------
    torch.Tensor
        The moment vector (*,2n+1).
    """

    n = H.shape[-1]

    # Construct a bi-diagonal matrix to index into the hankel matrix
    # [1 1 0 0 0]               [m0 m1 m2 m3 m4]
    # [0 1 1 0 0]               [m1 m2 m3 m4 m5]
    # [0 0 1 1 0] -- index -->  [m2 m3 m4 m5 m6] -- output --> [m0, m1, m2, m3, m4, m5, m6, m7, m8]
    # [0 0 0 1 1]               [m3 m4 m5 m6 m7]
    # [0 0 0 0 1]               [m4 m5 m6 m7 m8]
    A = torch.eye((n))
    A = A + A.roll(1, 1)
    A[-1, 0] = 0
    
    return H.reshape(-1, n*n)[..., A.reshape(-1) > 0]

def build_vandermonde_matrix(x: torch.Tensor, n_offset: int = 0, n_extra: int = 0):
    """
    Build the Vandermonde matrix of order n from an array of support points.

    Parameters
    ----------
    x : torch.Tensor
        Support points, as a tensor of shape (*,n).
    n_offset : int, optional
        Offset for the exponents (default is 0).
    n_extra : int, optional
        Additional exponents beyond the default range (default is 0).

    Returns
    -------
    torch.Tensor
        Vandermonde matrix (*,n,n) of the form:
        [1 x1 x1^2 ... x1^(n-1)]
        [          ...         ]
        [1 xn xn^2 ... xn^(n-1)].
    """

    n = x.shape[-1]

    # Generate sequence of exponents (0, 1, ..., n-1)
    e = torch.arange(n_offset, n+n_offset+n_extra, device=x.device)

    # Expand the exponents shape to match that of x
    e = e.expand((*x.shape[:-1], -1))

    return torch.pow(x[..., :, None], e[..., None, :])

def moment_curve(x: torch.Tensor, order: int, stack_dim=-1):
    """
    Evaluate the moment curve u(x) = (x^0, x^1, ..., x^order).

    Parameters
    ----------
    x : torch.Tensor
        The evaluation point, as tensor of shape (*,N).
    order : int
        Order of the curve.
    stack_dim : int, optional
        Dimension along which to stack the vector (default is -1).
    """
    return torch.stack([x**i for i in range(order+1)], dim=stack_dim)

def compute_moment_bias(n: int, lower: float = -1, upper: float = 1):
    yi = torch.linspace(lower, upper, n+1)

    Y = build_vandermonde_matrix(yi)
    W = torch.diag(torch.full(yi.shape, 1. / (yi.shape[0])))
    B = Y.T @ W @ Y

    return get_moments_from_hankel_matrix(B)

def chunk(count, chunk_size):
    num_chunks  = math.ceil(count / chunk_size)
    chunk_start = 0
    for i in range(num_chunks):
        chunk_size_current = chunk_size if (i != num_chunks - 1) else count - chunk_start
        assert chunk_size_current > 0
        yield (chunk_start, chunk_size_current)
        chunk_start += chunk_size_current

def generate_moment_vectors(n: int, num_vectors: int, num_points: int, dtype: torch.dtype, num_min_points: Optional[int] = None, device: torch.device = None, do_chunk: bool = False):
    """
    Generate a set of moment vectors by taking a weighted combination of points on the moment curve.

    For each vector, the number of points on the moment curve is chosen randomly between `num_min_points` and `num_points`,
    and their weight is also chosen randomly.

    Parameters
    ----------
    n : int
        Determines the number of moments as 2n+1.
    num_vectors : int
        Number of moment vectors to generate.
    num_points : int
        (Maximum) number of points on the moment curve.
    dtype : torch.dtype
        Data type used to generate the moments (it's best to use higher precision for higher order).
    num_min_points : int, optional
        Minimum number of points to use; if None, use all points (default is None).
    device : torch.device, optional
        Device to use for computation (default is None).
    do_chunk : bool, optional
        Whether to process in chunks to restrict the memory usage (default is False).

    Returns
    -------
    torch.Tensor
        Generated moment vectors.
    """
    ms = []
    for (chunk_start, chunk_size) in chunk(num_vectors, 1000000 if do_chunk else num_vectors):
        gc.collect()
        torch.cuda.empty_cache()

        a = torch.rand(num_points, chunk_size, device=device, dtype=dtype)
        x = torch.randn(num_points, chunk_size, device=device, dtype=dtype)

        # If the number of minimum points is *not* given
        # it's assumed to be `num_points` (always use all points).
        # If it is given, use the number of minimum points and above that
        # a *random* number of points, i.e., each moment vector is
        # construct from point count between `num_min_points` and `num_points`.
        if num_min_points:
            # The `mask` determines which points are "nulled"
            # so do not contribute to the moment vector
            mask = torch.rand_like(a) > 0.5
            mask[:num_min_points] = True
            a *= mask.to(dtype)

        m = (a[:, None, :] * moment_curve(x, order=2*n, stack_dim=1)).sum(dim=0)
        ms.append(m)

    return torch.cat(ms, dim=-1)