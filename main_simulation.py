"""
main_simulation.py
==================
Entry point for the FBG bistable interrogation numerical study.

Run with:
    python main_simulation.py

Outputs (saved to output/):
    fbg_spectrum.png
    resonator_response.png
    combined_response.png
    bistable_curve.png
    hysteresis_loop.png
    noisy_sweeps.png
    slope_profile.png
    peak_tracking_distribution.png
    resolution_comparison.png

Console summary includes:
    - Switching contrast
    - Hysteresis width (pm)
    - Maximum slope dI/dλ_B
    - Δλ_min for bistable and peak-tracking methods
    - Improvement factor
"""

import os
import sys
import time
import numpy as np

# ---------------------------------------------------------------------------
# Imports from project modules
# ---------------------------------------------------------------------------
from config import PARAMS
from fbg_model import fbg_reflection
from resonator_model import resonator_response
from interrogation import (
    compute_combined_response,
    wavelength_sweep_bistable,
    add_noise,
)
from peak_tracking import peak_tracking_resolution
from resolution_analysis import (
    bistable_resolution,
    hysteresis_metrics,
    compare_resolution,
)
from plotting import (
    plot_fbg_spectrum,
    plot_resonator_response,
    plot_combined_response,
    plot_bistable_curve,
    plot_hysteresis_loop,
    plot_noisy_sweeps,
    plot_slope,
    plot_peak_tracking_distribution,
    plot_resolution_comparison,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def section(title: str):
    """Print a clearly visible section header."""
    bar = "=" * 60
    print(f"\n{bar}")
    print(f"  {title}")
    print(bar)


def check_output_dir(path: str):
    os.makedirs(path, exist_ok=True)


# ---------------------------------------------------------------------------
# Main simulation
# ---------------------------------------------------------------------------

def run_simulation():
    t_start = time.time()

    p = PARAMS
    out = p["output_dir"]
    check_output_dir(out)

    rng = np.random.default_rng(seed=0)  # fixed seed for reproducibility

    # -----------------------------------------------------------------------
    # SECTION 1 – Spectral components
    # -----------------------------------------------------------------------
    section("1 / 6  Computing spectral components")

    lam = p["lambda_grid"]

    # FBG reflection spectrum at the nominal Bragg wavelength
    R_fbg = fbg_reflection(lam, p["lambda_B0"], p["Gamma_fbg"], p["R0"])

    # Resonator transmission spectrum
    H_r = resonator_response(lam, p["lambda_r"], p["Gamma_r"])

    # Combined spectral response
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
    S     = result["S"]
    I_out_spectrum = result["I_out"]

    # --- Sanity checks ---
    peak_fbg = lam[np.argmax(R_fbg)]
    peak_res = lam[np.argmax(H_r)]
    peak_S   = lam[np.argmax(S)]

    assert abs(peak_fbg - p["lambda_B0"]) < 0.005, \
        f"FBG peak mismatch: {peak_fbg:.4f} vs {p['lambda_B0']:.4f}"
    assert abs(peak_res - p["lambda_r"]) < 0.005, \
        f"Resonator peak mismatch: {peak_res:.4f} vs {p['lambda_r']:.4f}"

    print(f"  FBG peak:        {peak_fbg:.4f} nm  (target {p['lambda_B0']:.4f} nm)  ✓")
    print(f"  Resonator peak:  {peak_res:.4f} nm  (target {p['lambda_r']:.4f} nm)   ✓")
    print(f"  Combined S peak: {peak_S:.4f} nm")
    print(f"  R_fbg max:  {R_fbg.max():.4f}  (target {p['R0']:.4f})")
    print(f"  H_r max:    {H_r.max():.4f}  (target 1.0)")
    print(f"  S max:      {S.max():.4f}")

    # --- Save figures ---
    print("\n  Saving spectral figures...")
    plot_fbg_spectrum(lam, R_fbg, p["lambda_B0"], p["Gamma_fbg"], out)
    plot_resonator_response(lam, H_r, p["lambda_r"], p["Gamma_r"], out)
    plot_combined_response(lam, R_fbg, H_r, S, p["lambda_B0"], p["lambda_r"], out)
    plot_bistable_curve(lam, S, I_out_spectrum, out)

    # -----------------------------------------------------------------------
    # SECTION 2 – Bistable hysteresis sweeps
    # -----------------------------------------------------------------------
    section("2 / 6  Bistable hysteresis sweeps")

    lam_up   = p["lambda_B_sweep_up"]    # ascending
    lam_down = p["lambda_B_sweep_down"]  # descending
    lambda_probe = p["lambda_r"]         # probe at resonance centre

    lam_B_up, I_up, S_up = wavelength_sweep_bistable(
        lam_up, lambda_probe,
        p["lambda_r"], p["Gamma_fbg"], p["Gamma_r"],
        p["I0"], p["delta"], p["alpha"], direction="up",
    )
    lam_B_down, I_down, S_down = wavelength_sweep_bistable(
        lam_down, lambda_probe,
        p["lambda_r"], p["Gamma_fbg"], p["Gamma_r"],
        p["I0"], p["delta"], p["alpha"], direction="down",
    )

    # Hysteresis characterisation
    hm = hysteresis_metrics(lam_B_up, I_up, lam_B_down, I_down)

    print(f"  Up-switching point:   {hm['lambda_switch_up']:.5f} nm")
    print(f"  Down-switching point: {hm['lambda_switch_down']:.5f} nm")
    print(f"  Hysteresis width:     {hm['hysteresis_width'] * 1e3:.2f} pm")
    print(f"  Switching contrast:   {hm['switching_contrast']:.4f} a.u.")

    assert hm["hysteresis_width"] > 0, "No hysteresis detected — check β and λ_r offset!"
    assert hm["switching_contrast"] > 0.05, "Switching contrast too low!"
    print("  Hysteresis loop verified  ✓")

    plot_hysteresis_loop(
        lam_B_up, I_up, lam_B_down, I_down,
        lambda_switch_up=hm["lambda_switch_up"],
        lambda_switch_down=hm["lambda_switch_down"],
        hysteresis_width=hm["hysteresis_width"],
        output_dir=out,
    )

    # -----------------------------------------------------------------------
    # SECTION 3 – Noise realisation
    # -----------------------------------------------------------------------
    section("3 / 6  Adding detector noise")

    I_up_noisy   = add_noise(I_up,   p["sigma_I"], rng=rng)
    I_down_noisy = add_noise(I_down, p["sigma_I"], rng=rng)

    snr_up = float(np.mean(I_up) / p["sigma_I"]) if p["sigma_I"] > 0 else np.inf
    print(f"  Noise σ_I:  {p['sigma_I']:.4f} a.u.")
    print(f"  Mean I_out (up-sweep):  {np.mean(I_up):.4f} a.u.")
    print(f"  Estimated SNR:          {snr_up:.1f}")

    plot_noisy_sweeps(
        lam_B_up,   I_up,   I_up_noisy,
        lam_B_down, I_down, I_down_noisy,
        output_dir=out,
    )

    # -----------------------------------------------------------------------
    # SECTION 4 – Bistable resolution analysis
    # -----------------------------------------------------------------------
    section("4 / 6  Bistable resolution analysis")

    bist_res = bistable_resolution(lam_B_up, I_up, p["sigma_I"], smooth=True)

    print(f"  Max slope |dI/dλ_B|:  {bist_res['max_slope']:.4f} a.u./nm")
    print(f"  Switching point λ_sw: {bist_res['lambda_switch']:.5f} nm")
    print(f"  Δλ_min (bistable):    {bist_res['delta_lambda_min'] * 1e3:.4f} pm")

    assert bist_res["max_slope"] > 0, "Zero slope — bistable transition not detected!"
    print("  Bistable slope analysis  ✓")

    plot_slope(
        lam_B_up,
        bist_res["dI_dlam"],
        bist_res["lambda_switch"],
        bist_res["max_slope"],
        output_dir=out,
    )

    # -----------------------------------------------------------------------
    # SECTION 5 – Peak-tracking resolution (Monte-Carlo)
    # -----------------------------------------------------------------------
    section("5 / 6  Peak-tracking resolution (Monte-Carlo)")

    pt_res = peak_tracking_resolution(
        lam,
        p["lambda_B0"],
        p["Gamma_fbg"],
        p["sigma_I"],
        N_trials=p["N_noise_trials"],
        R0=p["R0"],
        method="centroid",
        rng_seed=42,
    )

    print(f"  N trials:              {p['N_noise_trials']}")
    print(f"  Mean estimate:         {pt_res['mean_estimate']:.6f} nm")
    print(f"  Bias:                  {pt_res['bias'] * 1e3:.4f} pm")
    print(f"  σ_λ (resolution):      {pt_res['std_estimate'] * 1e3:.4f} pm")
    print(f"  Δλ_min (peak tracking):{pt_res['delta_lambda_min'] * 1e3:.4f} pm")

    plot_peak_tracking_distribution(
        pt_res["lambda_B_estimates"],
        p["lambda_B0"],
        pt_res["std_estimate"],
        output_dir=out,
    )

    # -----------------------------------------------------------------------
    # SECTION 6 – Comparison and summary
    # -----------------------------------------------------------------------
    section("6 / 6  Comparison and final summary")

    dlam_bist = bist_res["delta_lambda_min"]
    dlam_peak = pt_res["delta_lambda_min"]

    if dlam_bist > 0 and not np.isinf(dlam_bist):
        improvement = dlam_peak / dlam_bist
    else:
        improvement = np.nan

    plot_resolution_comparison(dlam_bist, dlam_peak, improvement, output_dir=out)

    t_elapsed = time.time() - t_start

    # Pretty summary table
    SEP  = "─" * 52
    SEP2 = "═" * 52
    print(f"\n  {SEP2}")
    print(f"  {'SIMULATION SUMMARY':^50}")
    print(f"  {SEP2}")
    print(f"  {'Parameter':<38} {'Value':>10}")
    print(f"  {SEP}")
    print(f"  {'λ_B0 (Bragg wavelength)':<38} {p['lambda_B0']:>9.3f} nm")
    print(f"  {'Γ_FBG (FBG linewidth)':<38} {p['Gamma_fbg']*1e3:>9.1f} pm")
    print(f"  {'λ_r (resonator wavelength)':<38} {p['lambda_r']:>9.4f} nm")
    print(f"  {'Γ_r (resonator linewidth)':<38} {p['Gamma_r']*1e3:>9.1f} pm")
    print(f"  {'Δ (cavity detuning)':<38} {p['delta']:>10.2f}")
    print(f"  {'α (Kerr coefficient)':<38} {p['alpha']:>10.2f}")
    print(f"  {'I0 (input power scale)':<38} {p['I0']:>10.2f}")
    print(f"  {'σ_I (intensity noise)':<38} {p['sigma_I']:>10.4f}")
    print(f"  {SEP}")
    print(f"  {'Switching contrast':<38} {hm['switching_contrast']:>10.4f}")
    print(f"  {'Hysteresis width':<38} {hm['hysteresis_width']*1e3:>9.2f} pm")
    print(f"  {'Up-switch λ_B':<38} {hm['lambda_switch_up']:>9.5f} nm")
    print(f"  {'Down-switch λ_B':<38} {hm['lambda_switch_down']:>9.5f} nm")
    print(f"  {SEP}")
    print(f"  {'Max slope |dI/dλ_B|':<38} {bist_res['max_slope']:>7.4f} a.u./nm")
    print(f"  {'Δλ_min (bistable)':<38} {dlam_bist*1e3:>9.4f} pm")
    print(f"  {'Δλ_min (peak tracking)':<38} {dlam_peak*1e3:>9.4f} pm")
    print(f"  {'Improvement factor':<38} {improvement:>9.1f} ×")
    print(f"  {SEP}")
    print(f"  {'Elapsed time':<38} {t_elapsed:>9.2f} s")
    print(f"  {SEP2}")

    print(f"\n  All figures saved to: {os.path.abspath(out)}/")
    print("  Files generated:")
    for fname in sorted(os.listdir(out)):
        print(f"    {fname}")

    return {
        "switching_contrast":  hm["switching_contrast"],
        "hysteresis_width_pm": hm["hysteresis_width"] * 1e3,
        "max_slope":           bist_res["max_slope"],
        "delta_lam_bist_pm":   dlam_bist * 1e3,
        "delta_lam_peak_pm":   dlam_peak * 1e3,
        "improvement":         improvement,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\nFBG Bistable Optical Interrogation — Numerical Study")
    print("Author: numerical model, no external data required")
    print("Dependencies: numpy, scipy, matplotlib\n")

    results = run_simulation()

    # Exit with non-zero status if the bistable method shows no improvement
    if not np.isnan(results["improvement"]) and results["improvement"] < 1.0:
        print("\n[WARNING] Bistable method shows no improvement over peak tracking.")
        print("         Consider increasing β or adjusting λ_r offset.")
        sys.exit(1)

    print("\nSimulation completed successfully.")
