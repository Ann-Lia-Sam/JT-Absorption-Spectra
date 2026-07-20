"""Regression guard: the absorption-spectrum pipeline is byte-for-byte unchanged.

The vibronic/heatmap work is entirely additive -- none of the spectrum-path
modules (basis, operators, hamiltonian, spectrum) were touched. This test pins
the raw physics of a small, fast case so that any accidental change to the
spectrum pipeline is caught immediately. Baseline captured on the
``vibronic-paper-reimplementation`` branch with NORMALIZATION='none'.
"""
import numpy as np

from spectrum.config import Config
from spectrum.spectrum import compute_spectrum_for_Nv

# Golden values for Nv=2, sigma=0.24, 2 realizations, NORMALIZATION='none'.
_BASELINE = {
    "dim": 30,
    "spectrum_sum": 982.8740107471,
    "spectrum_max": 3.3985529682,
    "E_at_max": 6.5682841421,
    "evals_sum": 425.2842566353,
    "intens_sum": 2.0000000000,
}


def test_spectrum_pipeline_unchanged():
    cfg = Config()
    cfg.NORMALIZATION = "none"
    cfg.n_realizations = 2
    cfg.sigma = 0.24
    res = compute_spectrum_for_Nv(2, cfg, show_progress=False)
    E, s = res["E"], res["spectrum"]

    assert res["dim"] == _BASELINE["dim"]
    assert np.isclose(s.sum(), _BASELINE["spectrum_sum"], rtol=0, atol=1e-6)
    assert np.isclose(s.max(), _BASELINE["spectrum_max"], rtol=0, atol=1e-6)
    assert np.isclose(E[np.argmax(s)], _BASELINE["E_at_max"], rtol=0, atol=1e-6)
    assert np.isclose(res["all_evals"].sum(), _BASELINE["evals_sum"], rtol=0, atol=1e-6)
    assert np.isclose(res["all_intensity"].sum(), _BASELINE["intens_sum"], rtol=0, atol=1e-9)


if __name__ == "__main__":
    test_spectrum_pipeline_unchanged()
    print("spectrum-unchanged test passed")
