import numpy as np
from scipy import constants, special

from conical_horn import CHI_11_PRIME, ETA_0, ConicalHorn, HornGeometry


def test_default_geometry_and_eigenvalue():
    horn = ConicalHorn(radial_order=120, power_order=60)
    estimate = CHI_11_PRIME / horn.geometry.flare_half_angle - 0.5
    assert abs(horn.nu / estimate - 1) < 0.03
    assert horn.geometry.cutoff_frequency < 8e9


def test_outgoing_impedance_and_matched_reflection():
    horn = ConicalHorn(radial_order=80, power_order=40)
    x = 1000.0
    h = horn.schelkunoff_hankel(horn.nu, x, 2)
    hp = horn.schelkunoff_hankel_derivative(horn.nu, x, 2)
    assert abs((-1j * ETA_0 * h / hp) / ETA_0 - 1) < 0.01


def test_zero_phase_integral_and_aperture_efficiency():
    horn = ConicalHorn(radial_order=300, power_order=40)
    f = 10e9
    kt = 0.63 * 2 * np.pi * f / constants.c
    numeric = horn.aperture_spectrum(f, kt, phase_error=False)
    a = horn.geometry.aperture_radius
    kc = CHI_11_PRIME / a
    closed = []
    for n in (0, 2):
        value = a * (kt * special.jv(n, CHI_11_PRIME)
                     * special.jvp(n, kt * a)
                     - kc * special.jvp(n, CHI_11_PRIME)
                     * special.jv(n, kt * a)) / (kc * kc - kt * kt)
        closed.append(value)
    assert np.allclose(numeric, closed, rtol=1e-10, atol=1e-14)
    efficiency = 2 / (CHI_11_PRIME ** 2 - 1)
    directivity = horn.boresight_directivity(f, phase_error=False)
    expected = (4 * np.pi / (constants.c / f) ** 2
                * np.pi * a * a * efficiency)
    assert np.isclose(efficiency, 0.8368, atol=5e-5)
    assert np.isclose(directivity, expected, rtol=1e-10)


def test_sweep_model_a_is_finite():
    horn = ConicalHorn(radial_order=80, power_order=40)
    result = horn.frequency_sweep(np.array([9e9, 10e9]), load_model="A")
    assert np.all(np.isfinite(result["Zin"]))
    assert np.all(np.isfinite(result["S11_dB"]))
