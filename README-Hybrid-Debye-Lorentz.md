# Hybrid Debye-Lorentz Algorithm Implementation Details

## Table of Contents
1. [Model Definition](#1-model-definition)
2. [Initial Parameter Estimation](#2-initial-parameter-estimation)
3. [Data-Driven Parameter Bounds](#3-data-driven-parameter-bounds)
4. [Residual Function and Optimization](#4-residual-function-and-optimization)
5. [Statistical Analysis](#5-statistical-analysis)
6. [Fit Quality Evaluation](#6-fit-quality-evaluation)
7. [Numerical Stability](#7-numerical-stability)
8. [Reporting System](#8-reporting-system)

---

## 1. Model Definition

### 1.1 Complex Permittivity Model

The hybrid model represents complex permittivity as:

```
ε*(ω) = ε∞ + Σ[i=1 to N] Hi(ω)
```

Where each hybrid term Hi combines Debye and Lorentz characteristics:

```
Hi(ω) = βi · HD,i(ω) + (1-βi) · HL,i(ω)
```

### 1.2 Component Models

**Modified Debye (Cole-Cole) Term:**
```python
# Branch-cut stable implementation
omega_tau = omega * tau
ph = np.log(1j * omega_tau)  # consistent complex logarithm
HD,i(ω) = Ai / (1 + np.exp(alpha * ph))
```

**Damped Lorentz Oscillator Term:**
```python
ω0 = 1/τi  # Resonance frequency
HL,i(ω) = Ai · ω0² / (ω0² - ω² + j·γi·ω)
```

### 1.3 Model Parameters

For N terms, total parameters = 1 + 5N:
- `ε∞`: High-frequency permittivity limit
- For each term i:
  - `Ai`: Amplitude (strength of relaxation)
  - `τi`: Relaxation time (s)
  - `αi`: Cole-Cole exponent ∈ (0,1]
  - `γi`: Damping factor (dimensionless)
  - `βi`: Mixing parameter ∈ [0,1]

---

## 2. Initial Parameter Estimation

### 2.1 High-Frequency Permittivity (ε∞)

```python
def estimate_eps_inf(dk, n_pts):
    if n_pts >= 3:
        n_avg = max(3, int(n_pts * 0.05))  # at least 3 points or 5%
        eps_inf_init = float(np.mean(dk[-n_avg:]))  # plateau average
    else:
        eps_inf_init = float(dk[-1]) if n_pts else 1.0
```

**Rationale**: Averages highest frequency points to reduce noise sensitivity.

### 2.2 Relaxation Time Distribution

```python
# Logarithmic spacing for tau values
f_min, f_max = f_ghz.min(), f_ghz.max()
log_min, log_max = np.log10(f_min), np.log10(f_max)

# Generate n_terms center frequencies in log space
f_centers = np.logspace(log_min, log_max, n_terms + 2)[1:-1]

for i, f_center in enumerate(f_centers):
    tau_init = 1 / (2 * np.pi * f_center * 1e9)
```

**Rationale**: Logarithmic spacing ensures equal coverage per frequency decade, matching physical distribution of relaxation processes.

### 2.3 Amplitude Initialization

```python
delta_eps_total = dk[0] - dk[-1] if dk.size else 1.0
A_init = delta_eps_total / n_terms
```

**Rationale**: Distributes total permittivity change equally among terms.

### 2.4 Shape Parameters

```python
alpha_init = 0.8  # Moderate Cole-Cole behavior
gamma_init = 0.9  # Moderate damping
beta_init = 0.5   # Equal Debye/Lorentz mix
```

---

## 3. Data-Driven Parameter Bounds

### 3.1 Permittivity Bounds

```python
dk_vals = np.real(complex_epsilon_data)
dk_min, dk_max = dk_vals.min(), dk_vals.max()
delta_eps = dk_max - dk_min

# ε∞ bounds: within realistic range of high-freq plateau
eps_inf_min = max(0.5 * dk_max, 0.0)
eps_inf_max = 1.2 * dk_max
```

### 3.2 Relaxation Time Bounds

```python
# Convert frequency range to Hz
f_min_hz = f_ghz.min() * 1e9
f_max_hz = f_ghz.max() * 1e9

# τ bounds: 0.1× to 10× the measurement range
tau_min = 0.1 / (2 * np.pi * f_max_hz)
tau_max = 10.0 / (2 * np.pi * f_min_hz)
```

**Rationale**: Ensures relaxation times span physically relevant range for measured frequencies.

### 3.3 Amplitude Bounds

```python
A_min = 0.0
A_max = 1.2 * delta_eps  # 20% margin above total change
```

### 3.4 Shape Parameter Bounds

```python
alpha: [0.01, 1.0]  # Nearly pure Cole-Cole to pure Debye
gamma: [0.0, 10.0]  # Underdamped to heavily overdamped
beta:  [0.0, 1.0]   # Pure Lorentz to pure Debye
```

---

## 4. Residual Function and Optimization

### 4.1 Loss Tangent-Based Residuals

```python
def residual(params, f_ghz, complex_epsilon_data, n_terms, weight_df=2.0):
    # Calculate model
    eps_mod = calculate_hybrid_model(p_dict, f_ghz, n_terms)
    
    # Real-part residual (Dk)
    real_mod = np.real(eps_mod)
    real_meas = np.real(complex_epsilon_data)
    res_real = real_mod - real_meas
    
    # Loss-tangent residual (Df) - what we actually care about
    df_mod = -np.imag(eps_mod) / real_mod
    df_meas = -np.imag(complex_epsilon_data) / real_meas
    res_df = df_mod - df_meas
    
    # Stack with heavier weighting on Df
    return np.concatenate([res_real, weight_df * res_df])
```

**Rationale**: 
- Directly optimizes loss tangent (Df) which is displayed
- Weight factor of 2.0 emphasizes loss tangent fitting
- Consistent with physical measurements

### 4.2 Optimization Methods

Three methods available:
1. **least_squares**: Levenberg-Marquardt (default)
   - Best for well-conditioned problems
   - Uses gradient information efficiently
   
2. **nelder**: Nelder-Mead Simplex
   - Derivative-free, robust to noise
   - Slower but handles difficult surfaces
   
3. **lbfgsb**: Limited-memory BFGS
   - Memory efficient for many parameters
   - Good for smooth problems

---

## 5. Statistical Analysis

### 5.1 Chi-Square Calculation

```python
# Build residual vector exactly as optimized
resid_vec = np.concatenate([res_real, weight_df * res_df])
chi_sqr = np.sum(resid_vec**2)
n_dat = resid_vec.size
n_var = 1 + 5 * n_terms
dof = n_dat - n_var if n_dat > n_var else 1
red_chi_sqr = chi_sqr / dof
```

### 5.2 Information Criteria (AIC/BIC)

```python
# Normalize by noise estimates
sigma_real = np.std(res_real)
sigma_df = np.std(res_df)

# Weighted and normalized residuals
weighted_res_real = res_real / sigma_real
weighted_res_df = (weight_df * res_df) / sigma_df

resid_eff = np.concatenate([weighted_res_real, weighted_res_df])
chi2_eff = np.sum(resid_eff**2)

# Information criteria
aic = chi2_eff + 2 * n_var
bic = chi2_eff + n_var * np.log(n_eff)
aic_c = aic + (2*n_var*(n_var+1))/(n_eff - n_var - 1)  # small sample
```

**Rationale**: Proper likelihood-based model selection accounting for actual optimization objective.

### 5.3 Correlation Matrix

```python
def calculate_correlation_matrix(params_dict, f_ghz, complex_epsilon_data, n_terms):
    # Jacobian of residual function
    J = calculate_jacobian(params_dict, f_ghz, complex_epsilon_data, n_terms)
    
    # Check conditioning
    JTJ = J.T @ J
    cond = np.linalg.cond(JTJ)
    if cond > 1e12:
        raise ValueError(f"Ill-conditioned (cond={cond:.2e})")
    
    # Covariance and correlation
    cov = np.linalg.inv(JTJ + np.eye(JTJ.shape[0]) * 1e-12)
    stddevs = np.sqrt(np.diag(cov))
    corr = cov / np.outer(stddevs, stddevs)
```

---

## 6. Fit Quality Evaluation

### 6.1 Noise-Adaptive Thresholds

```python
DEFAULT_THRESHOLDS = {
    # metric: (mult_godly, mult_excellent, mult_good, suggestion)
    "reduced_chi_square": (1.0, 5.0, 10.0, "Check model/overfitting"),
    "rms_real": (1.0, 2.0, 5.0, "Large RMS in Dk"),
    "rms_imag": (1.0, 2.0, 5.0, "Large RMS in Df"),
    "mean_residual": (1.0, 2.0, 5.0, "Systematic bias"),
}

# Scale thresholds by noise level
for name, (mg, me, mgd, sugg) in DEFAULT_THRESHOLDS.items():
    σ = noise_levels.get(name, 1.0)
    scaled_thresh[name] = (mg*σ, me*σ, mgd*σ, sugg)
```

**Scoring**: 
- Godly (100): residual < 1σ
- Excellent (80): residual < 2-5σ
- Good (60): residual < 5-10σ
- Poor (30): residual > 10σ

### 6.2 Regional Frequency Analysis

```python
def evaluate_systematic_residuals(residuals_real, residuals_imag, f_ghz):
    # Log-space frequency division
    f = np.array(f_ghz)
    logf = np.log10(f)
    log_min, log_max = logf.min(), logf.max()
    
    # Equal thirds in log space
    f_break1 = 10**(log_min + (log_max - log_min)/3)
    f_break2 = 10**(log_min + 2*(log_max - log_min)/3)
    
    regions = {
        'low':  f <= f_break1,
        'mid':  (f > f_break1) & (f <= f_break2),
        'high': f > f_break2
    }
```

**Rationale**: Identifies frequency-dependent systematic errors.

### 6.3 Model-Specific Validation

```python
def validate_hybrid_model(params, n_terms):
    warnings = []
    
    # Check tau separation
    taus = sorted([params[f'tau{i+1}'] for i in range(n_terms)])
    for i in range(len(taus)-1):
        ratio = taus[i+1] / taus[i]
        if ratio < 2:
            warnings.append(f"Terms overlap at τ={taus[i]:.2e}")
    
    # Check contributions
    A_vals = [params[f'A{i+1}'] for i in range(n_terms)]
    total_A = sum(A_vals)
    for i, A in enumerate(A_vals):
        contrib = A/total_A * 100
        if contrib < 1:
            warnings.append(f"Term {i+1} contributes <1%")
        elif contrib > 95:
            warnings.append(f"Term {i+1} dominates >95%")
```

### 6.4 Overall Score Calculation

```python
# Weighted metrics
HYBRID_WEIGHTS = {
    'dpv': 1.0,                    # data points per variable
    'reduced_chi_square': 2.0,     # overall fit quality
    'rms_real': 3.0,              # Dk accuracy
    'rms_imag': 3.0,              # Df accuracy
    'mean_residual': 2.0,         # systematic bias
    'regional_fit': 4.0,          # frequency-dependent quality
    'tau_separation': 2.0,        # term independence
    'min_contribution': 3.0,      # term relevance
    'max_contribution': 2.0,      # term dominance
    'correlation': 4.0,           # parameter independence
    'complexity_penalty': 2.0     # model parsimony
}

overall_score = Σ(metric_score × weight) / Σ(weights)
```

---

## 7. Numerical Stability

### 7.1 Branch-Cut Stable Complex Power

```python
# Avoid: (1j * omega_tau) ** alpha
# Use: explicit log/exp formulation
ph = np.log(1j * omega_tau)
debye_term = A / (1 + np.exp(alpha * ph))
```

### 7.2 Condition Number Monitoring

```python
if np.linalg.cond(JTJ) > 1e12:
    raise ValueError("Matrix ill-conditioned")
```

### 7.3 Safe Division

```python
# Avoid division by zero in loss tangent
df = -np.imag(eps) / np.real(eps) if np.real(eps) != 0 else 0
```

---

## 8. Reporting System

### 8.1 Parameter Report

```
Fitted Parameters:
    eps_inf = 3.2451
    
    Term 1:
        A1     = 0.1234    # Amplitude
        tau1   = 1.234e-10 s (f0 = 1.29 GHz)
        alpha1 = 0.75      # Cole-Cole exponent
        gamma1 = 0.50      # Damping
        beta1  = 0.80      # 80% Debye, 20% Lorentz
```

### 8.2 Term Analysis

```python
for each term:
    f0 = 1/(2π·τ)                    # resonance frequency
    contribution = A/Σ(A) × 100       # relative importance
    character = based on α value      # Debye/Hybrid/Lorentz
    damping = based on γ value        # Under/Over-damped
    mixing = based on β value         # Debye/Lorentz percentage
```

### 8.3 Quality Metrics

1. **Statistical Measures**
   - Reduced χ²
   - RMS errors (Dk and Df)
   - Mean residuals
   - Regional fit quality

2. **Physical Validation**
   - Term separation (τ ratios)
   - Contribution analysis
   - Parameter correlations
   - Frequency range coverage

3. **Model Selection**
   - AIC/BIC/AICc values
   - Complexity penalty
   - Overfitting warnings

### 8.4 JSON Output Structure

```json
{
    "model": "Hybrid Debye-Lorentz",
    "n_terms": 2,
    "parameters": {...},
    "fit_statistics": {
        "chi_square": 0.0123,
        "reduced_chi_square": 0.0001
    },
    "residual_analysis": {
        "real_part": {"mean": 0.0001, "rms": 0.002},
        "loss_tangent": {"mean": 0.00001, "rms": 0.0003}
    },
    "information_criteria": {
        "AIC": 123.45,
        "BIC": 134.56,
        "AIC_c": 124.67
    },
    "evaluation": {
        "overall_score": 85.3,
        "overall_category": "Excellent",
        "metrics": {...}
    }
}
```

---

## Summary

This algorithm implements a physically meaningful approach to dielectric spectroscopy fitting:

1. **Data-driven initialization** ensures parameters start in reasonable ranges
2. **Adaptive bounds** prevent artificial constraints
3. **Loss tangent optimization** matches what users actually measure
4. **Noise-aware evaluation** provides meaningful quality assessment
5. **Comprehensive validation** catches common fitting problems
6. **Stable numerics** handle edge cases gracefully

The implementation balances physical accuracy, numerical stability, and practical usability for real-world dielectric measurements.