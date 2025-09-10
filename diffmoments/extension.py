import diffmoments_ext as dmx

__all__ = ["extension_build_type", "get_flags", "default_bias"]

def extension_build_type():
    return dmx.build_type()

def get_flags(retain_roots: bool):
    """ Helper function to determine flags from arguments """
    flags = 0
    if retain_roots:
        flags = flags | dmx.ComputeMomentBoundsFlags.RetainRoots
    return flags

def default_bias(n: int, scale: float = 2):
    # The bias values for n = 2, 3, 4 are the 32-bit biases recommended in
    # Münstermann et al. (2018) Moment-Based Order-IndependentTransparency (Supplementary Material, Table 1)
    return {
        1: scale*5e-8,
        2: scale*5e-7,
        3: scale*5e-6,
        4: scale*5e-5,
        5: scale*5e-4,
        6: scale*5e-3
    }[n]
