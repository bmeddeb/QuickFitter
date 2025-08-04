import numpy as np
import matplotlib.pyplot as plt
import csv
import re

# --- Core Dielectric Models ---

def debye(omega, delta_eps, tau, alpha=1.0):
    """Calculates the complex permittivity for a Cole-Debye model."""
    return delta_eps / (1 + (1j * omega * tau)**alpha)

def lorentz(omega, delta_eps, omega0, gamma):
    """Calculates the complex permittivity for a Lorentz oscillator model."""
    return (delta_eps * omega0**2) / (omega0**2 - omega**2 - 1j * omega * gamma)

# --- Data Generation and Saving ---

def generate_synthetic_data(freq_ghz, eps_inf, debye_terms, lorentz_terms, noise_level=0.01):
    """
    Generates synthetic dielectric data for a given set of model parameters.
    """
    omega = 2 * np.pi * freq_ghz * 1e9
    epsilon = np.full_like(omega, eps_inf, dtype=np.complex128)

    for term in debye_terms:
        epsilon += debye(omega, term['delta_eps'], term['tau'], term.get('alpha', 1.0))

    for term in lorentz_terms:
        omega0_rad = term['omega0_ghz'] * 2 * np.pi * 1e9
        gamma_rad = term['gamma_ghz'] * 2 * np.pi * 1e9
        epsilon += lorentz(omega, term['delta_eps'], omega0_rad, gamma_rad)

    real_noise = np.random.normal(0, noise_level, epsilon.shape)
    imag_noise = np.random.normal(0, noise_level, epsilon.shape)
    epsilon += real_noise + 1j * imag_noise

    return freq_ghz, epsilon

def save_to_csv(filename, freq_ghz, complex_epsilon):
    """
    Saves the generated dielectric data to a CSV file with Dk and Df.
    """
    dk = np.real(complex_epsilon)
    df = np.imag(complex_epsilon) / dk

    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Frequency_GHz', 'Dk', 'Df'])
        for i in range(len(freq_ghz)):
            writer.writerow([freq_ghz[i], dk[i], df[i]])
    print(f"✅ Data successfully saved to {filename}")

# --- Main Execution ---

# Define a standard frequency range for testing
freq_range_ghz = np.linspace(0.1, 1000, 2000)

# 🧪 Scenario 1: Simple Debye Peak
s1_params = {'debye': [{'delta_eps': 4.0, 'tau': 3e-12, 'alpha': 0.9}], 'lorentz': []}
freq1, eps1 = generate_synthetic_data(freq_range_ghz, 2.5, s1_params['debye'], s1_params['lorentz'], noise_level=0.02)

# 🧪 Scenario 2: Low-Frequency Debye Tail
s2_params = {'debye': [{'delta_eps': 5.0, 'tau': 1e-9}], 'lorentz': []}
freq2, eps2 = generate_synthetic_data(freq_range_ghz, 3.0, s2_params['debye'], s2_params['lorentz'], noise_level=0.01)

# 🧪 Scenario 3: Sharp Lorentz Resonance
s3_params = {'debye': [], 'lorentz': [{'delta_eps': 0.5, 'omega0_ghz': 300, 'gamma_ghz': 15}]}
freq3, eps3 = generate_synthetic_data(freq_range_ghz, 2.2, s3_params['debye'], s3_params['lorentz'], noise_level=0.01)

# 🧪 Scenario 4: Multiple Overlapping Peaks
s4_params = {
    'debye': [{'delta_eps': 3.0, 'tau': 5e-12, 'alpha': 0.8}],
    'lorentz': [{'delta_eps': 0.8, 'omega0_ghz': 500, 'gamma_ghz': 50}]
}
freq4, eps4 = generate_synthetic_data(freq_range_ghz, 2.0, s4_params['debye'], s4_params['lorentz'], noise_level=0.02)

# --- Process and Save All Scenarios ---
scenarios_to_process = {
    'Simple_Debye_Peak': (freq1, eps1),
    'Low_Frequency_Debye_Tail': (freq2, eps2),
    'Sharp_Lorentz_Resonance': (freq3, eps3),
    'Multiple_Overlapping_Peaks': (freq4, eps4)
}

for name, (freq, eps) in scenarios_to_process.items():
    filename = f"{name}.csv"
    save_to_csv(filename, freq, eps)

# --- Visualization ---
fig, axes = plt.subplots(4, 1, figsize=(10, 16), sharex=True)
plot_map = {
    'Scenario 1: Simple Debye Peak': (freq1, eps1),
    'Scenario 2: Low-Frequency Tail': (freq2, eps2),
    'Scenario 3: Sharp Lorentz Peak': (freq3, eps3),
    'Scenario 4: Multiple Peaks': (freq4, eps4)
}

for ax, (title, (freq, eps)) in zip(axes, plot_map.items()):
    ax.plot(freq, np.real(eps), 'b-', label="Dk ($\epsilon'$)")
    ax.plot(freq, np.imag(eps), 'r-', label="Loss ($\epsilon''$)")
    ax.set_title(title)
    ax.set_ylabel('Permittivity')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)

axes[-1].set_xlabel('Frequency (GHz)')
plt.tight_layout()
plt.show()