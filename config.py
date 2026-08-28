"""
config.py
=========
Central configuration for the FBG bistable interrogation simulation.

All physical and numerical parameters are defined here as a single dictionary
(PARAMS) so that every module imports from one place. Units are noted for each
parameter; wavelengths are in nanometres (nm) throughout for readability.

Research context
----------------
We simulate a resonant-bistable interrogation architecture for fiber Bragg
grating (FBG) sensors.  A narrowband FBG reflection spectrum is coupled to a
Lorentzian resonant filter with dispersive (Kerr) nonlinear feedback.
Near resonance, small shifts in the Bragg wavelength λ_B can trigger a
switching transition between two stable output-power states (optical
bistability), enabling discrimination of wavelength shifts well below the
FBG linewidth.

Bistability model
-----------------
The dispersive bistable model is:

    I_out · [1 + (Δ − α·I_out)²] = I_in

where:
    I_in  = I0 · S(λ_B)  — input drive (I0 = scale factor, S ∈ [0,1])
    Δ     — normalised cavity detuning (in units of half-linewidth)
    α     — Kerr nonlinear coefficient

Bistability requires |Δ| > √3 ≈ 1.732.  The fold bifurcation points (where
switching occurs) are computed analytically in bistable_solver.py.

I0 is chosen so that the maximum combined response S_max ≈ 1 maps into the
centre of the bistable window:
    I_in_max = I0 · S_max  falls between I_fold_low and I_fold_high.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Physical model parameters
# ---------------------------------------------------------------------------

PARAMS = {
    # --- FBG parameters ---------------------------------------------------
    "lambda_B0": 1550.0,   # [nm]  Nominal Bragg wavelength (centre of FBG peak)
    "Gamma_fbg": 0.2,      # [nm]  FBG reflection linewidth (FWHM).
                           #       Typical uniform FBG: 0.1–0.5 nm.
    "R0": 1.0,             # [–]   Peak reflectivity of the FBG (dimensionless, 0–1).
                           #       Set to 1.0 for a lossless, strongly coupled grating.

    # --- Resonant filter parameters ---------------------------------------
    "lambda_r": 1550.05,   # [nm]  Resonator centre wavelength.
                           #       Offset ~Γ_r/2 from λ_B0 to place the
                           #       operating point on the steep edge of S(λ_B),
                           #       maximising dI/dλ_B.
    "Gamma_r": 0.15,       # [nm]  Resonator linewidth (FWHM).
                           #       Narrower than Γ_fbg to sharpen the combined
                           #       spectral gate.

    # --- Kerr dispersive bistability parameters ---------------------------
    "delta": 3.0,          # [–]   Normalised cavity detuning Δ (in half-linewidth
                           #       units).  Must satisfy |Δ| > √3 ≈ 1.732 for
                           #       bistability.  Larger Δ → wider hysteresis loop.
    "alpha": 1.0,          # [a.u.⁻¹]  Kerr nonlinear coefficient.
                           #       Increasing α tightens the bistable window,
                           #       reducing the required I0.

    # --- Input power scale -----------------------------------------------
    # I0 is chosen so that the S probe range [S_min, S_max] as λ_B is swept
    # crosses BOTH fold bifurcation points.  For delta=3, alpha=1:
    #   I_in_fold_low  ≈ 2.91  (up-switch threshold:   I_in rises past this)
    #   I_in_fold_high ≈ 5.09  (down-switch threshold: I_in falls past this)
    # S_max ≈ 1.0 when λ_B ≈ λ_r, S_min ≈ 0.05 at the sweep edges.
    # Need I0·S_max > 5.09  →  I0 > 5.09.  Set I0=6.0 for margin.
    "I0": 6.0,             # [a.u.]  Input optical power scale.
                           #         Must satisfy I0·S_max > I_in_fold_high
                           #         AND I0·S_min < I_in_fold_low
                           #         so that both switching events occur in the sweep.

    # --- Wavelength grid --------------------------------------------------
    "lambda_min": 1549.0,  # [nm]  Lower bound of the simulation wavelength axis.
    "lambda_max": 1551.0,  # [nm]  Upper bound of the simulation wavelength axis.
    "N_lambda": 2000,      # [–]   Number of wavelength grid points.

    # --- Bragg-wavelength sweep (interrogation scan) ----------------------
    "sweep_half_width": 0.4,  # [nm]  Half-width of the λ_B sweep around λ_B0.
                               #       Sweep range: [λ_B0 − Δ, λ_B0 + Δ].
    "N_sweep": 500,            # [–]   Number of λ_B values in each sweep direction.

    # --- Noise ------------------------------------------------------------
    "sigma_I": 0.02,       # [a.u.] Standard deviation of additive Gaussian
                           #        intensity noise on I_out.
                           #        ~0.5% relative to a typical I_out ≈ 3–4 a.u.

    # --- Monte-Carlo noise analysis ---------------------------------------
    "N_noise_trials": 2000,  # [–]  Number of noise realisations used to
                              #      estimate peak-tracking resolution.

    # --- Output directory -------------------------------------------------
    "output_dir": "output",  # Directory where all figures are saved.
}

# ---------------------------------------------------------------------------
# Derived / convenience quantities (computed once at import time)
# ---------------------------------------------------------------------------

PARAMS["lambda_grid"] = np.linspace(
    PARAMS["lambda_min"], PARAMS["lambda_max"], PARAMS["N_lambda"]
)

PARAMS["lambda_B_sweep_up"] = np.linspace(
    PARAMS["lambda_B0"] - PARAMS["sweep_half_width"],
    PARAMS["lambda_B0"] + PARAMS["sweep_half_width"],
    PARAMS["N_sweep"],
)

PARAMS["lambda_B_sweep_down"] = PARAMS["lambda_B_sweep_up"][::-1].copy()


# ---------------------------------------------------------------------------
# Quick self-check
# ---------------------------------------------------------------------------

def _sanity_check():
    """Print a brief summary of the loaded configuration."""
    import numpy as np
    print("=== Configuration loaded ===")
    for key, val in PARAMS.items():
        if isinstance(val, np.ndarray):
            print(f"  {key}: array shape={val.shape}, "
                  f"range=[{val.min():.4f}, {val.max():.4f}]")
        else:
            print(f"  {key}: {val}")
    # Verify bistability condition
    delta = PARAMS["delta"]
    alpha = PARAMS["alpha"]
    I0    = PARAMS["I0"]
    cond  = abs(delta) > (3 ** 0.5)
    print(f"\n  Bistability condition |Δ|={abs(delta):.2f} > √3={3**0.5:.3f}: {cond}")
    # Fold points
    from bistable_solver import bistability_window
    w = bistability_window(delta, alpha, I0)
    if w["bistable"]:
        print(f"  Bistable window: S ∈ [{w['S_fold_low']:.3f}, {w['S_fold_high']:.3f}]")
        print(f"  I_in window:     [{w['I_in_fold_low']:.3f}, {w['I_in_fold_high']:.3f}]")
        print(f"  I0·S_max ≈ {I0 * 1.0:.2f}  (target: inside the window)")
    print("============================\n")


if __name__ == "__main__":
    _sanity_check()
