import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))
from quickfitter.sarkar import calculate_model, residual


def test_calculate_model_known_values():
    f_ghz = np.array([1.0, 2.0, 5.0])
    params = {
        "eps_inf": 2.0,
        "delta_eps": 1.0,
        "omega1": 2 * np.pi * 1e9,
        "omega2": 2 * np.pi * 10e9,
    }
    expected = np.array([
        2.85164569 - 0.29780854j,
        2.65903167 - 0.39510078j,
        2.34096833 - 0.39510078j,
    ])
    result = calculate_model(
        params["eps_inf"],
        params["delta_eps"],
        params["omega1"],
        params["omega2"],
        f_ghz,
    )
    np.testing.assert_allclose(result, expected, rtol=1e-8, atol=1e-10)


def test_residual_zero_when_data_matches():
    f_ghz = np.array([1.0, 2.0, 5.0])
    params = {
        "eps_inf": 2.0,
        "delta_eps": 1.0,
        "omega1": 2 * np.pi * 1e9,
        "omega2": 2 * np.pi * 10e9,
    }
    data = calculate_model(
        params["eps_inf"],
        params["delta_eps"],
        params["omega1"],
        params["omega2"],
        f_ghz,
    )
    res = residual(params, f_ghz, data)
    assert np.allclose(res, 0.0)
