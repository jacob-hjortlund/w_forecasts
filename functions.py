import numpy as np
import scipy as sp
from scipy.integrate import cumtrapz

# ----------------------------------------------------------------------------------

# ---- Constants ----

# speed of light in km / s
c = 299792.458

# Sound horizon used in fiducial cosmology (BOSS)
r_fid = 147.78

# ---- Functions ----


def H_LambdaCDM(z, H_0, O_m, O_k):
    """Calculates Hubble parameter for Lamba CDM cosmology at given redshifts
     and chosen cosmological parameters.

    Arguments:
        z : (N,) ndarray
            Redshifts
        H_0 : float
            Hubble constant
        O_m : float
            Matter density parameter
        O_k : float
            Curvature density parameter

    Returns:
        H : (N,) ndarray
            Hubble parameter at given redshifts
    """

    return H_0 * (O_m * (1+z) ** 3 + (1-O_m) + O_k * (1+z) ** 2) ** 0.5


def ComovingDistance(z, H_inv, H_0, O_k):
    """Calculates comoving distance for given redshifts and corresponding 
    Hubble parameter values for given a given cosmology.

    Arguments:
        z : (N,) ndarray
            Redshifts
        H_inv : (N,) ndarray
            Inverted Hubble parameter
            values at redshifts z
        H_0 : float
            Hubble constant
        O_k : float
            Curvature density parameter


    Returns:
        Dm : (N,) ndarray
            Comoving distances
    """

    DH = c / H_0
    Dm = cumtrapz(H_inv, z, initial=0) * c

    if O_k > 0.0:

        Dm = DH / O_k ** 0.5 * np.sinh(O_k ** 0.5 * Dm / DH)

    elif O_k < 0.0:

        Dm = DH / np.abs(O_k) ** 0.5 * np.sin(np.abs(O_k) ** 0.5 * Dm / DH)

    return Dm


def TimeDelayDistance(z_l, z_s, z, Dm_array, H_0, O_k):
    # TO DO: update docstring
    """Calculates the time delay distance from the lens and source redshifts 
    and comoving distances.

    Arguments:
        z_l : float
            Lens redshifts
        z_s : float
            Source redshifts
        z : (K,) ndarray
            Total array of redshifst
        Dm_array (K,) ndarray
            Comoving distances
        H_0 : float
            Hubble constant
        O_k : float
            Curvature parameter


    Returns:
        Ddt: float
            Time delay distance
    """

    Da_array = Dm_array / (1 + z_l)
    Da_l = Da_array[z == z_l]
    Da_s = Da_array[z == z_s]
    Dm_l = Dm_array[z == z_l]
    Dm_s = Dm_array[z == z_s]

    # TO DO: Finish this lol
    Da_ls = 1 / (1 + z_l) * (Dm_s * np.sqrt(1 + O_k *) )

    Ddt = (1+z_l) * Da_l * Da_s / Da_ls

    return Ddt


def Compute_LLH(r, sigma):
    """Compute loglikelihood given difference and covariance matrix. In case
    sigma is a 1D array, it is assumed to be an array of uncertainties.

    Arguments:
        r : (N,) ndarray
            Difference between measured and expected values
        sigma : (N,) or (N,N) ndarray
            Uncertainty array / inverse covariance matrix

    Returns:
        LLH : float
            Log likelihood for chosen parameters
    """

    if sigma.ndim == 1:
        LLH = -0.5 * np.sum(r ** 2 / sigma ** 2)

    else:
        LLH = -0.5 * r.T @ sigma @ r

    return LLH
