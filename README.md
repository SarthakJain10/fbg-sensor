This project numerically demonstrates a **resonant-bistable interrogation architecture** for fiber Bragg grating (FBG) sensors. A narrowband FBG reflection spectrum is coupled to a nonlinear Lorentzian resonant filter. Near the resonance, small shifts in the Bragg wavelength λ_B trigger a sharp switching transition between two stable output-power states (optical bistability), enabling discrimination of wavelength shifts well below the FBG linewidth — far surpassing the resolution of conventional peak-tracking interrogation.

## Model Equations

### FBG Reflection Spectrum (Lorentzian approximation)

```
R_fbg(λ) = R0 / [1 + (2(λ − λ_B) / Γ_fbg)²]
```

- `λ_B` — Bragg wavelength (shifts with strain/temperature)
- `Γ_fbg` — full-width at half-maximum (FWHM) linewidth

### Resonator Transmission

```
H_r(λ) = 1 / [1 + (2(λ − λ_r) / Γ_r)²]
```

- `λ_r` — resonance centre wavelength
- `Γ_r` — resonator linewidth (FWHM)

### Combined Spectral Response

```
S(λ_B) = R_fbg(λ_probe | λ_B) × H_r(λ_probe)
```

Evaluated at a fixed probe wavelength `λ_probe = λ_r`.

### Bistable Steady-State Equation

```
I_out = I0 · S(λ_B) / (1 + β · I_out)
```

Equivalently (quadratic form):

```
β · I_out² + I_out − I0 · S = 0
```

Two positive real roots exist in the bistable regime (`β · I0 · S` above a threshold), giving rise to hysteresis when λ_B is swept up and down.

### Resolution Metrics

**Bistable:**
```
Δλ_min = σ_I / max|dI_out/dλ_B|
```

**Peak tracking (Monte-Carlo):**
```
Δλ_min = std(λ̂_B)   over N_trials noise realisations
```

---

## Project Structure

```
hyster/
├── config.py               # All physical and numerical parameters
├── fbg_model.py            # FBG Lorentzian reflection spectrum
├── resonator_model.py      # Lorentzian resonator transmission
├── bistable_solver.py      # Quadratic bistable solver + hysteresis sweep
├── interrogation.py        # Full interrogation simulation (sweep + noise)
├── peak_tracking.py        # Conventional FBG peak-tracking method
├── resolution_analysis.py  # Resolution metrics and comparison
├── plotting.py             # All matplotlib figure generation
├── main_simulation.py      # Entry point — runs the full study
├── output/                 # Figures saved here (auto-created)
└── README.md
```

---

## Dependencies

Python 3.9+ with:

```bash
pip install numpy scipy matplotlib
```

`pandas` is optional (not required to run the simulation).

---

## How to Run

```bash
cd /path/to/hyster
python main_simulation.py
```

The simulation takes a few seconds and prints a summary table to the console.

---

## Outputs

All figures are saved to `output/`:

| File | Description |
|------|-------------|
| `fbg_spectrum.png` | FBG Lorentzian reflection spectrum |
| `resonator_response.png` | Resonator Lorentzian transmission |
| `combined_response.png` | Overlay of FBG, resonator, and S(λ) |
| `bistable_curve.png` | Bistable I_out vs wavelength and vs S |
| `hysteresis_loop.png` | Up/down λ_B sweeps showing hysteresis |
| `noisy_sweeps.png` | Clean vs noisy bistable sweeps |
| `slope_profile.png` | |dI_out/dλ_B| profile (sensitivity) |
| `peak_tracking_distribution.png` | Monte-Carlo λ̂_B noise histogram |
| `resolution_comparison.png` | Bar chart: Δλ_min comparison |

---

## Parameters to Tune

Edit `config.py` to change any of the following:

| Parameter | Key | Effect |
|-----------|-----|--------|
| Bragg wavelength | `lambda_B0` | Centre of FBG peak |
| FBG linewidth | `Gamma_fbg` | Width of reflection peak; wider → less sharp S |
| Resonator wavelength | `lambda_r` | Offset from λ_B0 sets operating point on S slope |
| Resonator linewidth | `Gamma_r` | Narrower → steeper combined edge, more sensitive |
| Nonlinear coefficient | `beta` | Higher → wider hysteresis loop; too high → no switching |
| Input power | `I0` | Scales with β; `β·I0` is the relevant product |
| Noise | `sigma_I` | ~0.5 % for shot-noise-limited detector |
| Sweep range | `sweep_half_width` | Must cover the full hysteresis window |

### Key sensitivity relationships

- **λ_r offset from λ_B0**: placing λ_r ~Γ_r/2 away from λ_B0 puts the operating point on the steepest slope of S(λ_B), maximising dI/dλ_B.
- **β**: must satisfy `β · I0 · S_max > ≈ 0.38` for bistability to occur. Increase to widen the hysteresis window.
- **Γ_r**: narrowing the resonator sharpens S and increases max|dI/dλ_B|, directly improving Δλ_min.

---

## Extending the Code

### Parameter sweeps

```python
from config import PARAMS
from resolution_analysis import compare_resolution
from interrogation import wavelength_sweep_bistable

betas = np.linspace(2, 20, 20)
results = []
for beta in betas:
    lam_up = PARAMS["lambda_B_sweep_up"]
    _, I_up, _ = wavelength_sweep_bistable(
        lam_up, PARAMS["lambda_r"],
        PARAMS["lambda_r"], PARAMS["Gamma_fbg"], PARAMS["Gamma_r"],
        PARAMS["I0"], beta, direction="up",
    )
    cmp = compare_resolution(
        lam_up, I_up, PARAMS["sigma_I"],
        PARAMS["lambda_grid"], PARAMS["lambda_B0"],
        PARAMS["Gamma_fbg"], PARAMS["R0"], N_trials=500,
    )
    results.append(cmp["improvement"])
```

### Optimization

Use `scipy.optimize.minimize` or a grid search over `(beta, lambda_r, Gamma_r)` with `improvement` as the objective.

### Coupled-mode FBG model

Replace `fbg_reflection()` in `fbg_model.py` with a transfer-matrix solver (e.g., Rouard's method) for full spectral accuracy including sidelobes.

---

## Physical Interpretation of Results

The key result is the **improvement factor**:

```
F = Δλ_min(peak tracking) / Δλ_min(bistable)
```

A value F >> 1 shows that the bistable interrogator can resolve wavelength shifts that are invisible to conventional peak tracking. This improvement grows as:

- The resonator linewidth Γ_r decreases (sharper spectral gate)
- The nonlinear coefficient β increases (steeper switching transition)
- The noise σ_I decreases (better detector)

The hysteresis width sets the **dynamic range** of unambiguous switching: shifts larger than the hysteresis window may not trigger a transition on the same branch.
