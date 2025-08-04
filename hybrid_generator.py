import numpy as np
import matplotlib.pyplot as plt

# --- Core Dielectric Models ---

def debye(omega, delta_eps, tau, alpha=1.0):
    """Calculates the complex permittivity for a Cole-Debye model."""
    return delta_eps / (1 + (1j * omega * tau)**alpha)

def lorentz(omega, delta_eps, omega0, gamma):
    """Calculates the complex permittivity for a Lorentz oscillator model."""
    return (delta_eps * omega0**2) / (omega0**2 - omega**2 - 1j * omega * gamma)

# --- Data Generation Function ---

def generate_synthetic_data(freq_ghz, eps_inf, debye_terms, lorentz_terms, noise_level=0.01):
    """
    Generates synthetic dielectric data for a given set of model parameters.

    Args:
        freq_ghz (np.array): Array of frequencies in GHz.
        eps_inf (float): Permittivity at infinite frequency.
        debye_terms (list of dicts): List of Debye term parameters.
            Each dict: {'delta_eps': float, 'tau': float, 'alpha': float}
        lorentz_terms (list of dicts): List of Lorentz term parameters.
            Each dict: {'delta_eps': float, 'omega0_ghz': float, 'gamma_ghz': float}
        noise_level (float): Standard deviation of the Gaussian noise to add.

    Returns:
        tuple: (frequency_hz, complex_permittivity)
    """
    omega = 2 * np.pi * freq_ghz * 1e9  # Convert frequency to rad/s
    epsilon = np.full_like(omega, eps_inf, dtype=np.complex128)

    # Add Debye terms
    for term in debye_terms:
        epsilon += debye(omega, term['delta_eps'], term['tau'], term.get('alpha', 1.0))

    # Add Lorentz terms
    for term in lorentz_terms:
        omega0_rad = term['omega0_ghz'] * 2 * np.pi * 1e9
        gamma_rad = term['gamma_ghz'] * 2 * np.pi * 1e9
        epsilon += lorentz(omega, term['delta_eps'], omega0_rad, gamma_rad)

    # Add realistic noise
    real_noise = np.random.normal(0, noise_level, epsilon.shape)
    imag_noise = np.random.normal(0, noise_level, epsilon.shape)
    epsilon += real_noise + 1j * imag_noise

    return freq_ghz * 1e9, epsilon

# --- Test Case Scenarios ---

# Define a standard frequency range for testing
freq_range_ghz = np.linspace(0.1, 1000, 2000) # 0.1 GHz to 1 THz

# Scenario 1: Simple Debye Peak (Well within range)
# 🧪 Tests if the script can find a simple, ideal peak.
test1_debye = [{'delta_eps': 4.0, 'tau': 3e-12, 'alpha': 0.9}] # Peak around 53 GHz
freq1, eps1 = generate_synthetic_data(freq_range_ghz, 2.5, test1_debye, [], noise_level=0.02)

# Scenario 2: Low-Frequency Debye (Tail in range)
# 🧪 The original challenging case you tested.
# Tests if the script correctly reports high parameter correlations.
test2_debye = [{'delta_eps': 5.0, 'tau': 1e-9}] # Peak around 159 MHz
freq2, eps2 = generate_synthetic_data(freq_range_ghz, 3.0, test2_debye, [], noise_level=0.01)

# Scenario 3: Sharp Lorentz Resonance (Well within range)
# 🧪 Tests the Lorentz part of the model.
test3_lorentz = [{'delta_eps': 0.5, 'omega0_ghz': 300, 'gamma_ghz': 15}]
freq3, eps3 = generate_synthetic_data(freq_range_ghz, 2.2, [], test3_lorentz, noise_level=0.01)

# Scenario 4: Multiple Peaks (Complex Scenario)
# 🧪 Tests the script's ability to deconvolve multiple overlapping features.
test4_debye = [{'delta_eps': 3.0, 'tau': 5e-12, 'alpha': 0.8}] # ~32 GHz peak
test4_lorentz = [{'delta_eps': 0.8, 'omega0_ghz': 500, 'gamma_ghz': 50}] # 500 GHz resonance
freq4, eps4 = generate_synthetic_data(freq_range_ghz, 2.0, test4_debye, test4_lorentz, noise_level=0.02)

# --- Visualization ---
# You can plot the generated data to see what it looks like.
fig, axes = plt.subplots(4, 1, figsize=(10, 16), sharex=True)
scenarios = {
    'Scenario 1: Simple Debye Peak': (freq1, eps1),
    'Scenario 2: Low-Frequency Tail': (freq2, eps2),
    'Scenario 3: Sharp Lorentz Peak': (freq3, eps3),
    'Scenario 4: Multiple Peaks': (freq4, eps4)
}

for ax, (title, (freq, eps)) in zip(axes, scenarios.items()):
    ax.plot(freq / 1e9, np.real(eps), 'b-', label="Real Part ($\epsilon'$)")
    ax.plot(freq / 1e9, np.imag(eps), 'r-', label="Imaginary Part ($\epsilon''$)")
    ax.set_title(title)
    ax.set_ylabel('Permittivity')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)

axes[-1].set_xlabel('Frequency (GHz)')
plt.tight_layout()
plt.show()


# Example for saving Scenario 1:
data_to_save = np.vstack((freq1, np.real(eps1), np.imag(eps1))).T
np.savetxt('scenario1_data.txt', data_to_save, header='Freq(Hz) Epsilon_Real Epsilon_Imag')