"""
interrogation.py
================
Full bistable interrogation simulation.

This module ties together the FBG spectrum, resonator response, and bistable
solver to produce the quantities of interest for the interrogation study:

1. compute_combined_response()
   Given a fixed λ_B (Bragg wavelength), compute the spectral response
   S(λ) = R_fbg(λ) · H_r(λ) and the resulting bistable I_out for every
   wavelength in the grid.

2. wavelength_sweep_bistable()
   The core interrogation experiment: sweep λ_B over a small range and record
   I_out at a fixed probe wavelength λ_probe (= λ_r by default, the resonance
   centre).  Perform both up- and down-sweeps to reveal hysteresis.

3. add_noise()
   Add Gaussian intensity noise to simulate detector shot/thermal noise.

Probe-wavelength scheme
-----------------------
We evaluate the bistable output at the fixed wavelength λ_probe = λ_r.
As λ_B is swept:
    S(λ_B) = R_fbg(λ_probe | λ_B) · H_r(λ_probe)

This is a *single-wavelength* detection scheme — analogous to locking a
photodetector to a single CW probe laser at λ_probe and reading the
transmitted/reflected power.  The bistable cavity converts the gradual change
in S(λ_B) into a sharp power switching event, enabling fine-resolution
discrimination of λ_B shifts.

Bistability model
-----------------
Uses the dispersive (Kerr) model from bistable_solver.py:

    I_out · [1 + (Δ − α·I_out)²] = I0 · S(λ_B)
"""

import numpy as np

from fbg_model import fbg_reflection
from resonator_model import resonator_response, resonator_response_at
from bistable_solver import solve_bistable_sweep, solve_bistable


# ---------------------------------------------------------------------------
# Combined spectral response
# ---------------------------------------------------------------------------

def compute_combined_response(
    lambda_vals: np.ndarray,
    lambda_B: float,
    lambda_r: float,
    Gamma_fbg: float,
    Gamma_r: float,
    I0: float,
    delta: float,
    alpha: float,
) -> dict:
    """
    Compute all spectral quantities for a fixed Bragg wavelength.

    Parameters
    ----------
    lambda_vals : np.ndarray
        Wavelength grid [nm], shape (N,).
    lambda_B : float
        Bragg wavelength [nm].
    lambda_r : float
        Resonator centre wavelength [nm].
    Gamma_fbg : float
        FBG linewidth [nm].
    Gamma_r : float
        Resonator linewidth [nm].
    I0 : float
        Input power scale [a.u.].
    delta : float
        Normalised cavity detuning.
    alpha : float
        Kerr nonlinear coefficient.

    Returns
    -------
    dict with keys:
        'lambda_vals' : np.ndarray  – wavelength axis [nm]
        'R_fbg'       : np.ndarray  – FBG reflection spectrum
        'H_r'         : np.ndarray  – resonator transmission spectrum
        'S'           : np.ndarray  – combined response S = R_fbg · H_r
        'I_out'       : np.ndarray  – bistable output for each λ in grid
    """
    R_fbg = fbg_reflection(lambda_vals, lambda_B, Gamma_fbg)
    H_r   = resonator_response(lambda_vals, lambda_r, Gamma_r)
    S     = R_fbg * H_r  # combined spectral gate

    # Solve bistable equation for each (wavelength, S) pair.
    # No hysteresis tracking here — we plot the equilibrium lower-branch curve.
    I_out = solve_bistable_sweep(I0, S, delta, alpha, direction="up")

    return {
        "lambda_vals": lambda_vals,
        "R_fbg":       R_fbg,
        "H_r":         H_r,
        "S":           S,
        "I_out":       I_out,
    }


# ---------------------------------------------------------------------------
# Single-wavelength probe S(λ_B)
# ---------------------------------------------------------------------------

def probe_response(
    lambda_B_array: np.ndarray,
    lambda_probe: float,
    lambda_r: float,
    Gamma_fbg: float,
    Gamma_r: float,
) -> np.ndarray:
    """
    Compute the combined response S evaluated at the fixed probe wavelength
    λ_probe as a function of the Bragg wavelength λ_B.

    This is the scalar signal that drives the bistable cavity for each
    λ_B in the sweep.

    Parameters
    ----------
    lambda_B_array : np.ndarray
        Array of Bragg wavelengths [nm] to evaluate at.
    lambda_probe : float
        Fixed probe wavelength [nm] (typically = λ_r).
    lambda_r : float
        Resonator centre wavelength [nm].
    Gamma_fbg : float
        FBG linewidth [nm].
    Gamma_r : float
        Resonator linewidth [nm].

    Returns
    -------
    S_probe : np.ndarray
        Combined response values, shape equal to lambda_B_array.shape.
    """
    # FBG reflectivity at λ_probe as λ_B is swept
    # R_fbg(λ_probe | λ_B) = R0 / [1 + (2(λ_probe - λ_B)/Γ_fbg)²]
    R_fbg_probe = fbg_reflection(
        np.full_like(lambda_B_array, lambda_probe),
        lambda_B_array,
        Gamma_fbg,
    )
    # Resonator is static (fixed λ_r, Γ_r) — scalar
    H_r_probe = resonator_response_at(lambda_probe, lambda_r, Gamma_r)

    return R_fbg_probe * H_r_probe


# ---------------------------------------------------------------------------
# Hysteresis sweep
# ---------------------------------------------------------------------------

def wavelength_sweep_bistable(
    lambda_B_sweep: np.ndarray,
    lambda_probe: float,
    lambda_r: float,
    Gamma_fbg: float,
    Gamma_r: float,
    I0: float,
    delta: float,
    alpha: float,
    direction: str = "up",
) -> tuple:
    """
    Simulate the bistable interrogation output for a sweep of Bragg wavelengths.

    The probe wavelength λ_probe is fixed.  As λ_B moves, the FBG reflection
    at λ_probe changes, modulating S and hence I_out.

    Parameters
    ----------
    lambda_B_sweep : np.ndarray
        Ordered array of Bragg wavelengths [nm] to sweep through, shape (N,).
        For an up-sweep pass np.linspace(start, stop, N);
        for a down-sweep pass the reversed array.
    lambda_probe : float
        Fixed probe wavelength [nm].
    lambda_r : float
        Resonator centre wavelength [nm].
    Gamma_fbg : float
        FBG linewidth [nm].
    Gamma_r : float
        Resonator linewidth [nm].
    I0 : float
        Input power scale [a.u.].
    delta : float
        Normalised cavity detuning.
    alpha : float
        Kerr nonlinear coefficient.
    direction : str
        'up' or 'down'.  Must match the order of lambda_B_sweep.

    Returns
    -------
    lambda_B_sweep : np.ndarray
        The same Bragg-wavelength array (returned for convenience), shape (N,).
    I_out : np.ndarray
        Output intensity for each λ_B, shape (N,).
    S_probe : np.ndarray
        Combined spectral response at λ_probe, shape (N,).
    """
    S_probe = probe_response(
        lambda_B_sweep, lambda_probe, lambda_r, Gamma_fbg, Gamma_r
    )

    I_out = solve_bistable_sweep(I0, S_probe, delta, alpha, direction=direction)

    return lambda_B_sweep, I_out, S_probe


# ---------------------------------------------------------------------------
# Noise
# ---------------------------------------------------------------------------

def add_noise(
    I_out: np.ndarray,
    sigma_I: float,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """
    Add additive white Gaussian noise to the output intensity.

    Models detector shot noise / thermal noise as zero-mean Gaussian noise
    with standard deviation σ_I (normalised to the same units as I_out).

    Parameters
    ----------
    I_out : np.ndarray
        Clean output intensity array, shape (N,) or scalar.
    sigma_I : float
        Noise standard deviation [same units as I_out].
    rng : np.random.Generator or None
        NumPy random generator for reproducibility.  If None, creates a new
        default generator (non-deterministic).

    Returns
    -------
    I_noisy : np.ndarray
        Noisy output intensity, same shape as I_out.
    """
    if rng is None:
        rng = np.random.default_rng()

    noise = rng.normal(loc=0.0, scale=sigma_I, size=np.asarray(I_out).shape)
    return np.asarray(I_out) + noise


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------

def _run_checks():
    """
    Quick integration test:
    - Combined response should peak near min(λ_B0, λ_r).
    - Up and down sweeps should differ (hysteresis present).
    - Noisy signal should have the right standard deviation.
    """
    from config import PARAMS

    p = PARAMS
    lam = p["lambda_grid"]

    # Combined response spectrum
    result = compute_combined_response(
        lam,
        p["lambda_B0"],
        p["lambda_r"],
        p["Gamma_fbg"],
        p["Gamma_r"],
        p["I0"],
        p["delta"],
        p["alpha"],
    )
    S = result["S"]
    peak_lam = lam[np.argmax(S)]
    print(f"[interrogation] Combined response peak at: {peak_lam:.4f} nm")
    assert p["lambda_min"] < peak_lam < p["lambda_max"], "Peak outside grid!"
    print("[interrogation] Combined response peak within grid  ✓")

    # Hysteresis sweep
    lam_up   = p["lambda_B_sweep_up"]
    lam_down = p["lambda_B_sweep_down"]
    lambda_probe = p["lambda_r"]

    _, I_up, S_up = wavelength_sweep_bistable(
        lam_up, lambda_probe,
        p["lambda_r"], p["Gamma_fbg"], p["Gamma_r"],
        p["I0"], p["delta"], p["alpha"], direction="up",
    )
    _, I_down, S_down = wavelength_sweep_bistable(
        lam_down, lambda_probe,
        p["lambda_r"], p["Gamma_fbg"], p["Gamma_r"],
        p["I0"], p["delta"], p["alpha"], direction="down",
    )

    # Align down-sweep to ascending λ_B for comparison
    I_down_asc = I_down[::-1]
    gap = np.abs(I_up - I_down_asc).max()
    print(f"[interrogation] Max hysteresis gap in I_out: {gap:.4f} a.u.")
    assert gap > 0.1, "No hysteresis detected — check delta/alpha/I0!"
    print("[interrogation] Hysteresis detected  ✓")

    # Noise test
    I_test = np.ones(10_000) * 2.0
    sigma = 0.05
    I_noisy = add_noise(I_test, sigma, rng=np.random.default_rng(42))
    measured_sigma = float(np.std(I_noisy))
    assert abs(measured_sigma - sigma) < 0.005, (
        f"Noise σ measured {measured_sigma:.4f}, expected {sigma:.4f}"
    )
    print(f"[interrogation] Noise σ measured: {measured_sigma:.4f}  ✓")


if __name__ == "__main__":
    _run_checks()
