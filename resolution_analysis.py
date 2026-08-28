"""
resolution_analysis.py
======================
Resolution metrics for both interrogation methods.

Definitions
-----------
For a measurement system that maps λ_B → I_out, the minimum resolvable
wavelength shift is:

    Δλ_min = σ_I / |dI_out/dλ_B|_max

where σ_I is the r.m.s. intensity noise and |dI_out/dλ_B|_max is the maximum
slope of the transfer function.  This is the optical sensitivity equivalent of
the classic "noise-equivalent input" metric.

Bistable scheme
~~~~~~~~~~~~~~~
The bistable interrogator exploits the steep edge of the hysteresis loop.
Near the switching point, |dI_out/dλ_B| can be orders of magnitude larger
than for a linear scheme, dramatically reducing Δλ_min.

Peak-tracking scheme
~~~~~~~~~~~~~~~~~~~~
The resolution is estimated from Monte-Carlo noise realisations:
Δλ_min = std(λ̂_B) over many trials.  This accounts for the actual noise
level, grid resolution, and peak-finding algorithm.

Improvement factor
------------------
    F = Δλ_min(peak tracking) / Δλ_min(bistable)

F > 1 means the bistable scheme resolves smaller shifts.
"""

import numpy as np
from scipy.signal import savgol_filter

from interrogation import wavelength_sweep_bistable
from peak_tracking import peak_tracking_resolution


# ---------------------------------------------------------------------------
# Bistable resolution
# ---------------------------------------------------------------------------

def bistable_slope(
    lambda_B_array: np.ndarray,
    I_out_array: np.ndarray,
    smooth: bool = True,
    window_frac: float = 0.05,
) -> tuple:
    """
    Compute dI_out/dλ_B numerically from sweep data.

    Uses a central-difference scheme on the (optionally Savitzky–Golay
    smoothed) I_out array to reduce numerical noise before differentiation.

    Parameters
    ----------
    lambda_B_array : np.ndarray
        Bragg wavelength values [nm], shape (N,).  Must be monotonically
        increasing (use the up-sweep for maximum slope near switching).
    I_out_array : np.ndarray
        Corresponding output intensities [a.u.], shape (N,).
    smooth : bool
        If True, apply Savitzky–Golay smoothing before differentiation.
    window_frac : float
        Fraction of the array length to use as the SG window.  Larger →
        smoother derivative but may under-estimate the peak slope.

    Returns
    -------
    dI_dlam : np.ndarray
        Numerical derivative dI_out/dλ_B [a.u./nm], shape (N,).
    max_slope : float
        Maximum absolute slope |dI_out/dλ_B| [a.u./nm].
    lambda_switch : float
        λ_B value at which |dI_out/dλ_B| is maximum [nm].
    """
    N = len(lambda_B_array)

    if smooth and N > 10:
        # Window length must be odd and ≥ poly_order + 1
        wl = max(5, int(window_frac * N))
        wl = wl if wl % 2 == 1 else wl + 1
        I_smooth = savgol_filter(I_out_array, window_length=wl, polyorder=3)
    else:
        I_smooth = I_out_array.copy()

    # Central differences (numpy gradient handles edges with 1st-order)
    dlam = lambda_B_array[1] - lambda_B_array[0]  # uniform grid assumed
    dI_dlam = np.gradient(I_smooth, dlam)

    idx_max = int(np.argmax(np.abs(dI_dlam)))
    max_slope     = float(np.abs(dI_dlam[idx_max]))
    lambda_switch = float(lambda_B_array[idx_max])

    return dI_dlam, max_slope, lambda_switch


def bistable_resolution(
    lambda_B_array: np.ndarray,
    I_out_array: np.ndarray,
    sigma_I: float,
    smooth: bool = True,
) -> dict:
    """
    Estimate the minimum resolvable wavelength shift for the bistable scheme.

    Parameters
    ----------
    lambda_B_array : np.ndarray
        Bragg wavelength sweep array [nm], shape (N,).
    I_out_array : np.ndarray
        Bistable output intensities [a.u.], shape (N,).
    sigma_I : float
        Intensity noise standard deviation [a.u.].
    smooth : bool
        Smooth I_out before differentiation.

    Returns
    -------
    dict with keys:
        'dI_dlam'          : np.ndarray  – derivative array [a.u./nm]
        'max_slope'        : float       – max |dI_out/dλ_B| [a.u./nm]
        'lambda_switch'    : float       – switching λ_B [nm]
        'delta_lambda_min' : float       – resolution [nm] = σ_I / max_slope
    """
    dI_dlam, max_slope, lam_sw = bistable_slope(
        lambda_B_array, I_out_array, smooth=smooth
    )

    if max_slope < 1e-15:
        delta_lambda_min = np.inf
    else:
        delta_lambda_min = sigma_I / max_slope

    return {
        "dI_dlam":          dI_dlam,
        "max_slope":        max_slope,
        "lambda_switch":    lam_sw,
        "delta_lambda_min": delta_lambda_min,
    }


# ---------------------------------------------------------------------------
# Hysteresis characterisation
# ---------------------------------------------------------------------------

def hysteresis_metrics(
    lambda_B_up: np.ndarray,
    I_out_up: np.ndarray,
    lambda_B_down: np.ndarray,
    I_out_down: np.ndarray,
) -> dict:
    """
    Characterise the bistable hysteresis loop.

    Parameters
    ----------
    lambda_B_up : np.ndarray
        Bragg wavelength array for the up-sweep [nm], ascending.
    I_out_up : np.ndarray
        Output intensity for the up-sweep.
    lambda_B_down : np.ndarray
        Bragg wavelength array for the down-sweep [nm], descending.
    I_out_down : np.ndarray
        Output intensity for the down-sweep.

    Returns
    -------
    dict with keys:
        'lambda_switch_up'   : float – λ_B at which I_out jumps on up-sweep [nm]
        'lambda_switch_down' : float – λ_B at which I_out jumps on down-sweep [nm]
        'hysteresis_width'   : float – switching width |Δλ_sw| [nm]
        'contrast_up'        : float – ΔI at up-switching event [a.u.]
        'contrast_down'      : float – ΔI at down-switching event [a.u.]
        'switching_contrast' : float – max(contrast_up, contrast_down)
    """
    # Up-sweep: find the point of largest positive jump in I_out
    dI_up = np.diff(I_out_up)
    idx_up = int(np.argmax(dI_up))
    lam_sw_up = float(lambda_B_up[idx_up])
    contrast_up = float(dI_up[idx_up])

    # Down-sweep: find the point of largest negative jump
    dI_down = np.diff(I_out_down)
    idx_down = int(np.argmin(dI_down))
    lam_sw_down = float(lambda_B_down[idx_down])
    contrast_down = float(abs(dI_down[idx_down]))

    hysteresis_width = abs(lam_sw_up - lam_sw_down)

    return {
        "lambda_switch_up":   lam_sw_up,
        "lambda_switch_down": lam_sw_down,
        "hysteresis_width":   hysteresis_width,
        "contrast_up":        contrast_up,
        "contrast_down":      contrast_down,
        "switching_contrast": max(contrast_up, contrast_down),
    }


# ---------------------------------------------------------------------------
# Combined comparison
# ---------------------------------------------------------------------------

def compare_resolution(
    lambda_B_up: np.ndarray,
    I_out_up: np.ndarray,
    sigma_I: float,
    lambda_vals: np.ndarray,
    lambda_B0: float,
    Gamma_fbg: float,
    R0: float,
    N_trials: int = 2000,
    peak_method: str = "centroid",
    rng_seed: int = 42,
) -> dict:
    """
    Compute and compare the resolution of bistable vs. peak-tracking methods.

    Parameters
    ----------
    lambda_B_up : np.ndarray
        Up-sweep Bragg wavelength array [nm], ascending.
    I_out_up : np.ndarray
        Bistable output for the up-sweep.
    sigma_I : float
        Intensity noise standard deviation.
    lambda_vals : np.ndarray
        Wavelength grid for FBG spectrum [nm].
    lambda_B0 : float
        Nominal Bragg wavelength [nm].
    Gamma_fbg : float
        FBG linewidth [nm].
    R0 : float
        Peak reflectivity.
    N_trials : int
        Monte-Carlo trials for peak-tracking.
    peak_method : str
        Peak-finding method for peak tracking.
    rng_seed : int
        RNG seed for reproducibility.

    Returns
    -------
    dict with keys:
        'bistable'        : dict from bistable_resolution()
        'peak_tracking'   : dict from peak_tracking_resolution()
        'delta_lam_bist'  : float – bistable Δλ_min [nm]
        'delta_lam_peak'  : float – peak-tracking Δλ_min [nm]
        'improvement'     : float – ratio Δλ_min(peak) / Δλ_min(bist)
    """
    bist_res = bistable_resolution(lambda_B_up, I_out_up, sigma_I)

    pt_res = peak_tracking_resolution(
        lambda_vals, lambda_B0, Gamma_fbg, sigma_I,
        N_trials=N_trials, R0=R0,
        method=peak_method, rng_seed=rng_seed,
    )

    dlam_bist = bist_res["delta_lambda_min"]
    dlam_peak = pt_res["delta_lambda_min"]

    if dlam_bist > 0 and dlam_bist < np.inf:
        improvement = dlam_peak / dlam_bist
    else:
        improvement = np.nan

    return {
        "bistable":       bist_res,
        "peak_tracking":  pt_res,
        "delta_lam_bist": dlam_bist,
        "delta_lam_peak": dlam_peak,
        "improvement":    improvement,
    }


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------

def _run_checks():
    from config import PARAMS

    p = PARAMS

    lam_up = p["lambda_B_sweep_up"]
    lam_down = p["lambda_B_sweep_down"]

    _, I_up, _ = wavelength_sweep_bistable(
        lam_up, p["lambda_r"],
        p["lambda_r"], p["Gamma_fbg"], p["Gamma_r"],
        p["I0"], p["delta"], p["alpha"], direction="up",
    )
    _, I_down, _ = wavelength_sweep_bistable(
        lam_down, p["lambda_r"],
        p["lambda_r"], p["Gamma_fbg"], p["Gamma_r"],
        p["I0"], p["delta"], p["alpha"], direction="down",
    )

    # Hysteresis metrics
    hm = hysteresis_metrics(lam_up, I_up, lam_down, I_down)
    print(f"[resolution_analysis] Hysteresis width: "
          f"{hm['hysteresis_width'] * 1e3:.2f} pm")
    print(f"[resolution_analysis] Switching contrast: "
          f"{hm['switching_contrast']:.4f} a.u.")

    # Resolution comparison
    cmp = compare_resolution(
        lam_up, I_up,
        p["sigma_I"],
        p["lambda_grid"], p["lambda_B0"],
        p["Gamma_fbg"], p["R0"],
        N_trials=500,  # quick check
    )
    print(f"[resolution_analysis] Δλ_min (bistable):     "
          f"{cmp['delta_lam_bist'] * 1e3:.4f} pm")
    print(f"[resolution_analysis] Δλ_min (peak tracking):"
          f" {cmp['delta_lam_peak'] * 1e3:.4f} pm")
    print(f"[resolution_analysis] Improvement factor:     "
          f"{cmp['improvement']:.1f}×")
    assert cmp["improvement"] > 1.0, (
        "Bistable method shows no improvement — check parameters."
    )
    print("[resolution_analysis] Bistable method outperforms peak tracking  ✓")


if __name__ == "__main__":
    _run_checks()
