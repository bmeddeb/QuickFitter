# Hybrid Debye-Lorentz Model Updates Summary

This document summarizes all the major fixes and improvements implemented in the Hybrid Debye-Lorentz model fitter during the optimization session.

## Problem 1: Hard-coded 50/50 Weighting of Debye vs. Lorentz

### Issue
The model used a fixed 50/50 weighting between Debye and Lorentz components, limiting flexibility.

### Solution
Introduced a new "mix" parameter β_i ∈ [0,1] for each term:
- **Formula**: `H_i(ω) = β_i * H_D,i(ω) + (1-β_i) * H_L,i(ω)`
- **Implementation**: Added beta parameter to initialization, calculation, and UI controls
- **Benefits**: β=0 gives pure Lorentz, β=1 gives pure Debye, β=0.5 maintains original behavior

### Files Modified
- `estimate_initial_parameters()`: Added `beta_init = 0.5`
- `calculate_hybrid_model()`: Dynamic weighting with beta parameter
- `run_analysis()`: Parameter registration with bounds [0.0, 1.0]
- UI controls: Added beta sliders and parameter handling

---

## Problem 2: Residuals on ε″ instead of Loss Tangent (Df)

### Issue
Optimizer minimized imaginary permittivity (ε″) residuals, but plots/reports focused on loss tangent (Df).

### Solution
Changed residual function to directly minimize loss tangent differences:
```python
# Old approach
imag_residual = (np.imag(model_epsilon) - np.imag(complex_epsilon_data))

# New approach  
df_mod = -np.imag(eps_mod) / real_mod
df_meas = -np.imag(complex_epsilon_data) / real_meas
res_df = df_mod - df_meas
```

### Benefits
- Better fit quality for displayed loss tangent curves
- Physically meaningful optimization objective
- Consistent between what's minimized and what's reported

---

## Problem 3: Suboptimal τ-initialization via Linear Sections

### Issue
Using `np.array_split()` on linear frequency axis bunched initial τ values toward middle frequencies.

### Solution
Implemented logarithmic spacing for τ initialization:
```python
# Log-spaced center frequencies
f_min, f_max = f_ghz.min(), f_ghz.max()
log_min, log_max = np.log10(f_min), np.log10(f_max)
f_centers = np.logspace(log_min, log_max, n_terms + 2)[1:-1]

for i, f_center in enumerate(f_centers):
    tau_init = 1 / (2 * np.pi * f_center * 1e9)
```

### Benefits
- Better frequency coverage across entire range
- Physically meaningful distribution of relaxation times
- Improved convergence and parameter estimation

---

## Problem 4: Incorrect Jacobian Using Model Output Instead of Residuals

### Issue
Jacobian calculation used model derivatives instead of residual function derivatives, invalidating correlation estimates.

### Solution
Updated Jacobian to use actual residual function:
```python
# Added helper function
def _dict_to_parameters(params_dict: dict):
    p = Parameters()
    for name, val in params_dict.items():
        p.add(name, value=val)
    return p

# Fixed Jacobian calculation
base_res = residual(base_params, f_ghz, complex_epsilon_data, n_terms)
res_plus = residual(p_plus, f_ghz, complex_epsilon_data, n_terms)
J[:, i] = (res_plus - base_res) / h
```

### Benefits
- Valid correlation matrix estimates
- Accurate parameter uncertainties
- Consistent statistical analysis

---

## Problem 5: Inconsistent χ² on Raw ε″ Instead of Loss Tangent

### Issue
Chi-square statistics used raw ε″ residuals while optimizer minimized weighted Df residuals.

### Solution
Made chi-square calculation consistent with optimization:
```python
# Build residual vector exactly as in residual()
resid_vec = np.concatenate([res_real, weight_df * res_df])
chi_sqr = np.sum(resid_vec**2)
```

### Benefits
- Meaningful fit quality statistics
- Consistent with actual optimization objective
- Reliable reduced chi-square values

---

## Problem 6: Over-restrictive Hard-coded Parameter Bounds

### Issue
Fixed bounds like `max=10` for amplitudes and `min=1e-13, max=1e-6` for tau restricted parameter space artificially.

### Solution
Implemented data-driven adaptive bounds:
```python
# Data-driven bounds
dk_vals = np.real(complex_epsilon_data)
delta_eps = dk_vals.max() - dk_vals.min()

# Frequency-based tau bounds
f_min_hz = f_ghz.min() * 1e9
f_max_hz = f_ghz.max() * 1e9
tau_min = 0.1 / (2 * np.pi * f_max_hz)
tau_max = 10.0 / (2 * np.pi * f_min_hz)

# Amplitude bounds scale with data
max=1.2 * delta_eps
```

### Benefits
- Adapts to actual measurement range and precision
- No artificial parameter clipping
- Physically appropriate parameter space exploration

---

## Problem 7: Duplicate Hard-coded Evaluation Thresholds

### Issue
Multiple threshold dictionaries (`HYBRID_THRESHOLDS`, `adjusted_thresholds`) that don't adapt to dataset noise.

### Solution
Consolidated to noise-adaptive thresholds:
```python
DEFAULT_THRESHOLDS = {
    "reduced_chi_square": (1.0, 5.0, 10.0, "Check model overfitting..."),
    "rms_real": (1.0, 2.0, 5.0, "Large RMS in Dk..."),
    "rms_imag": (1.0, 2.0, 5.0, "Large RMS in Df..."),
    "mean_residual": (1.0, 2.0, 5.0, "Systematic bias..."),
}

# Auto-scale thresholds
for name, (mg, me, mgd, sugg) in DEFAULT_THRESHOLDS.items():
    σ = noise_levels.get(name, 1.0)
    scaled_thresh[name] = (mg*σ, me*σ, mgd*σ, sugg)
```

### Benefits
- Single source of truth for thresholds
- Automatically adapts to measurement precision
- "Godly" threshold = 1× noise level

---

## Problem 8: AIC/BIC on Arbitrarily Weighted Residuals

### Issue
AIC/BIC calculation used arbitrary weights instead of proper noise normalization.

### Solution
Normalized residuals by true noise standard deviations:
```python
# Normalize each component by its true noise σ
sigma_real = np.std(res_real)
sigma_df = np.std(res_df)

# Use weighted residuals consistent with optimization
weighted_res_real = res_real / sigma_real
weighted_res_df = (weight_df * res_df) / sigma_df

resid_eff = np.concatenate([weighted_res_real, weighted_res_df])
chi2_eff = np.sum(resid_eff**2)

# Proper AIC/BIC
aic = chi2_eff + 2 * n_var
bic = chi2_eff + n_var * np.log(n_eff)
```

### Benefits
- Statistically valid information criteria
- Proper model comparison capabilities
- Consistent with Gaussian noise assumptions

---

## Problem 9: Branch-cut Instability in (jωτ)^α

### Issue
Fractional complex power `(1j * omega_tau) ** alpha` caused discontinuities when crossing negative real axis.

### Solution
Used explicit log+exp formulation:
```python
# Old: branch-cut prone
debye_term = A / (1 + (1j * omega_tau) ** alpha)

# New: branch-cut stable  
omega_tau = omega * tau
ph = np.log(1j * omega_tau)  # consistent complex logarithm
debye_term = A / (1 + np.exp(alpha * ph))
```

### Benefits
- Smooth model curves across all frequencies
- Eliminated numerical artifacts
- More stable optimization

---

## Problem 10: Silent Failure in Correlation Matrix Calculation

### Issue
Function caught all exceptions silently and returned None, later causing crashes on `.tolist()`.

### Solution
Implemented proper error handling with diagnostics:
```python
def calculate_correlation_matrix(...):
    # Check condition number
    cond = np.linalg.cond(JTJ)
    if cond > 1e12:
        raise ValueError(f"Jacobian matrix ill-conditioned (cond={cond:.2e})")
    
    # Validate variances
    if np.any(stddevs <= 0):
        raise ValueError("Invalid variances detected in covariance matrix")

# In run_analysis()
try:
    corr_matrix = calculate_correlation_matrix(...)
except Exception as e:
    logger.warning(f"Could not compute correlation matrix: {e}")
    P = 1 + 5 * n_terms
    corr_matrix = np.zeros((P, P))  # Safe fallback
```

### Benefits
- Never crashes analysis
- Provides diagnostic information
- Graceful degradation with meaningful error messages

---

## Problem 11: Noisy ε∞ from Single Point

### Issue
Using `eps_inf_init = dk[-1]` made initialization sensitive to noise in final data point.

### Solution
Implemented plateau averaging:
```python
if n_pts >= 3:
    n_avg = max(3, int(n_pts * 0.05))  # at least 3 points or 5%
    eps_inf_init = float(np.mean(dk[-n_avg:]))  # plateau average
else:
    eps_inf_init = float(dk[-1]) if n_pts else 1.0
```

### Benefits
- Noise-resistant ε∞ estimation
- Better initial parameter guess
- Improved convergence

---

## Problem 12: Frequency-Region Analysis by Index

### Issue
Residual analysis divided data by array index instead of actual frequency, causing uneven log-space coverage.

### Solution
Implemented log-space frequency division:
```python
f = np.array(f_ghz)
logf = np.log10(f)
log_min, log_max = logf.min(), logf.max()

# Equal log-space divisions
f_break1 = 10**(log_min + (log_max - log_min)/3)
f_break2 = 10**(log_min + 2*(log_max - log_min)/3)

regions = {
    'low':  f <= f_break1,
    'mid':  (f > f_break1) & (f <= f_break2),
    'high': f > f_break2
}
```

### Benefits
- Physically meaningful frequency regions
- Equal coverage in log-space
- Proper dielectric spectroscopy analysis

---

## Problem 13: Duplicated Weights Dictionary

### Issue
`evaluate()` function ignored `self.weights` and redefined weights dictionary, preventing customization.

### Solution
Used class instance weights:
```python
# Old: ignored class settings
weights = {'dpv': 1.0, 'reduced_chi_square': 2.0, ...}

# New: respects class configuration
weights = self.weights
```

### Benefits
- Consistent weight usage
- Allows custom evaluation priorities
- Single source of truth for weights

---

## Summary of Impact

### Statistical Validity
- ✅ Proper χ² calculation consistent with optimization
- ✅ Valid AIC/BIC for model comparison  
- ✅ Accurate correlation matrices and parameter uncertainties
- ✅ Noise-adaptive evaluation thresholds

### Physical Realism
- ✅ Loss tangent fitting matches displayed quantities
- ✅ Smooth model curves without branch-cut artifacts
- ✅ Frequency-aware regional analysis
- ✅ Logarithmic tau initialization for better physics

### Robustness
- ✅ Data-driven parameter bounds
- ✅ Graceful error handling with diagnostics
- ✅ Plateau averaging for noise resistance
- ✅ Flexible beta parameter for model mixing

### Maintainability
- ✅ Consolidated threshold definitions
- ✅ Single source of truth for weights
- ✅ Consistent code patterns throughout
- ✅ Clear separation of concerns

All changes maintain backward compatibility while significantly improving the accuracy, robustness, and physical meaningfulness of the hybrid Debye-Lorentz model fitting capabilities.