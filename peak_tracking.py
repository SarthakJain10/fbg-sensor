"""
peak_tracking.py
================
Conventional FBG peak-tracking interrogation.

Background
----------
The standard interrogation method for FBG sensors is to illuminate the grating
with a broadband source, acquire the reflection spectrum R_fbg(λ), and locate
the wavelength of maximum reflectivity.  As strain or temperature shifts λ_B,
the peak moves proportionally.

Resolution limit
----------------
In practice, the spectrum is sampled on a discrete wavelength grid and
corrupted by detector noise.  The minimum resolvable wavelength shift Δλ_min
is set by how precisely the peak position can be estimated in the presence of
noise.  For a smooth Lorentzian peak, sub-pixel estimation via centroid or
parabolic interpolation can push the resolution below one grid step, but it
remains ultimately noise-limited.

Metric
------
We estimate Δλ_min by Monte-Carlo: repeat the noisy peak-finding N times at
a fixed λ_B and compute the standard deviation of the estimated peak position.
This σ_λ is the single-shot resolution.  A 3σ or SNR-based criterion can be
applied for a 99 % confidence interval.
"""

import numpy as np
from scipy.interpolate import UnivariateSpline

from fbg_model import fbg_reflection
from interrogation import add_noise


# ---------------------------------------------------------------------------
# Peak-finding estimators
# ---------------------------------------------------------------------------

def estimate_lambda_B_peak(
    lambda_vals: np.ndarray,
    R_fbg: np.ndarray,
    method: str = "centroid",
) -> float:
    """
    Estimate the Bragg wavelength from a (possibly noisy) reflection spectrum.

    Parameters
    ----------
    lambda_vals : np.ndarray
        Wavelength axis [nm], shape (N,).
    R_fbg : np.ndarray
        Reflection spectrum (possibly noisy), shape (N,).  Values may go
        slightly negative due to noise — this is handled gracefully.
    method : str
        Peak-finding method:
        - 'argmax'   : coarse grid maximum (grid-limited resolution).
        - 'centroid' : intensity-weighted centroid within the half-max window
                       (sub-pixel, robust to noise).
        - 'parabola' : fit a parabola to the three points around the maximum
                       (sub-pixel, best for smooth spectra).

    Returns
    -------
    float
        Estimated Bragg wavelength [nm].
    """
    R_clipped = np.clip(R_fbg, 0.0, None)  # remove unphysical negatives

    if method == "argmax":
        return float(lambda_vals[np.argmax(R_clipped)])

    # Find coarse peak index first (used by both remaining methods)
    i_peak = int(np.argmax(R_clipped))

    if method == "parabola":
        # Need at least 3 points around the peak
        if i_peak == 0 or i_peak == len(lambda_vals) - 1:
            return float(lambda_vals[i_peak])

        # Parabolic interpolation: fit y = a(x-x0)² + b(x-x0) + c around peak
        # Exact formula for three points at indices i-1, i, i+1:
        y0, y1, y2 = R_clipped[i_peak - 1], R_clipped[i_peak], R_clipped[i_peak + 1]
        denom = 2.0 * (2.0 * y1 - y0 - y2)
        if abs(denom) < 1e-15:
            return float(lambda_vals[i_peak])
        delta_idx = (y0 - y2) / denom  # sub-pixel offset in index units
        dlam = lambda_vals[1] - lambda_vals[0]  # uniform grid step
        return float(lambda_vals[i_peak] + delta_idx * dlam)

    if method == "centroid":
        # Centroid within the top-50% window to suppress noise from far wings
        half_max = 0.5 * R_clipped[i_peak]
        mask = R_clipped >= half_max
        if mask.sum() < 2:
            return float(lambda_vals[i_peak])
        weights = R_clipped[mask]
        lam_masked = lambda_vals[mask]
        return float(np.sum(weights * lam_masked) / np.sum(weights))

    raise ValueError(f"Unknown method: {method!r}.  "
                     "Choose 'argmax', 'centroid', or 'parabola'.")


# ---------------------------------------------------------------------------
# Monte-Carlo resolution estimation
# ---------------------------------------------------------------------------

def peak_tracking_resolution(
    lambda_vals: np.ndarray,
    lambda_B: float,
    Gamma_fbg: float,
    sigma_I: float,
    N_trials: int = 2000,
    R0: float = 1.0,
    method: str = "centroid",
    rng_seed: int = 42,
) -> dict:
    """
    Estimate the peak-tracking resolution by repeated noisy measurements.

    For each trial:
    1. Generate R_fbg(λ) at the given λ_B.
    2. Add Gaussian noise with standard deviation σ_I.
    3. Estimate λ_B from the noisy spectrum.

    The standard deviation of the resulting λ̂_B distribution is the
    single-shot resolution Δλ_min.

    Parameters
    ----------
    lambda_vals : np.ndarray
        Wavelength grid [nm], shape (N,).
    lambda_B : float
        True Bragg wavelength [nm].
    Gamma_fbg : float
        FBG linewidth [nm].
    sigma_I : float
        Intensity noise standard deviation.
    N_trials : int
        Number of Monte-Carlo realisations.
    R0 : float
        Peak reflectivity.
    method : str
        Peak-finding method passed to estimate_lambda_B_peak().
    rng_seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict with keys:
        'lambda_B_estimates' : np.ndarray  – shape (N_trials,), estimated λ_B
        'mean_estimate'      : float       – mean of estimates [nm]
        'bias'               : float       – mean − true λ_B [nm]
        'std_estimate'       : float       – standard deviation [nm] = Δλ_min
        'delta_lambda_min'   : float       – = std_estimate (single-shot resolution)
    """
    rng = np.random.default_rng(rng_seed)

    # Clean reference spectrum (same for every trial, only noise changes)
    R_clean = fbg_reflection(lambda_vals, lambda_B, Gamma_fbg, R0)

    estimates = np.empty(N_trials)
    for i in range(N_trials):
        R_noisy = add_noise(R_clean, sigma_I, rng=rng)
        estimates[i] = estimate_lambda_B_peak(lambda_vals, R_noisy, method=method)

    mean_est = float(np.mean(estimates))
    std_est  = float(np.std(estimates))
    bias     = mean_est - lambda_B

    return {
        "lambda_B_estimates": estimates,
        "mean_estimate":      mean_est,
        "bias":               bias,
        "std_estimate":       std_est,
        "delta_lambda_min":   std_est,
    }


# ---------------------------------------------------------------------------
# Sensitivity (dλ / dI): used in comparisons
# ---------------------------------------------------------------------------

def peak_tracking_snr_estimate(
    lambda_vals: np.ndarray,
    lambda_B: float,
    Gamma_fbg: float,
    delta_lambda: float = 1e-4,
    R0: float = 1.0,
) -> float:
    """
    Estimate the slope of the FBG peak position vs. a small wavelength shift.

    This is always 1.0 nm/nm by definition (peak tracks λ_B exactly), but the
    function computes the numerical derivative of the *peak intensity* at the
    centroid, which would be zero.  Instead we return the peak-position slope
    = 1.0 as a reminder that peak tracking recovers λ_B changes 1:1.

    The relevant figure of merit for comparison is the *noise-limited*
    resolution from peak_tracking_resolution(), not this slope.

    Parameters
    ----------
    lambda_vals : np.ndarray
        Wavelength grid [nm].
    lambda_B : float
        Bragg wavelength [nm].
    Gamma_fbg : float
        FBG linewidth [nm].
    delta_lambda : float
        Small perturbation for numerical derivative [nm].
    R0 : float
        Peak reflectivity.

    Returns
    -------
    float
        d(λ̂_B) / d(λ_B) = 1.0 (ideal noiseless case).
    """
    # For an ideal estimator, every λ_B shift is recovered 1:1.
    return 1.0


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------

def _run_checks():
    from config import PARAMS

    p = PARAMS
    lam = p["lambda_grid"]

    # Noiseless peak should be at λ_B
    R_clean = fbg_reflection(lam, p["lambda_B0"], p["Gamma_fbg"], p["R0"])
    lam_hat = estimate_lambda_B_peak(lam, R_clean, method="centroid")
    error = abs(lam_hat - p["lambda_B0"])
    print(f"[peak_tracking] Noiseless centroid estimate: {lam_hat:.5f} nm  "
          f"(error {error:.5f} nm)")
    assert error < 1e-3, f"Large noiseless error: {error:.5f} nm"
    print("[peak_tracking] Noiseless centroid estimate correct  ✓")

    # Parabolic interpolation
    lam_hat_para = estimate_lambda_B_peak(lam, R_clean, method="parabola")
    err_para = abs(lam_hat_para - p["lambda_B0"])
    print(f"[peak_tracking] Noiseless parabola estimate: {lam_hat_para:.5f} nm  "
          f"(error {err_para:.5f} nm)")
    assert err_para < 1e-3
    print("[peak_tracking] Noiseless parabola estimate correct  ✓")

    # Monte-Carlo resolution
    res = peak_tracking_resolution(
        lam, p["lambda_B0"], p["Gamma_fbg"], p["sigma_I"],
        N_trials=p["N_noise_trials"], R0=p["R0"],
    )
    print(f"[peak_tracking] MC resolution (σ_λ): {res['delta_lambda_min'] * 1e3:.4f} pm")
    print(f"[peak_tracking] Bias: {res['bias'] * 1e3:.4f} pm")
    assert res["delta_lambda_min"] > 0, "Zero resolution estimate!"
    print("[peak_tracking] Monte-Carlo resolution check  ✓")


if __name__ == "__main__":
    _run_checks()
