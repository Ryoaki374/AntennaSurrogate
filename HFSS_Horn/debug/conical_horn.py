"""Numerical analytical model of a circular-waveguide fed conical horn.

The implementation follows the equations in the accompanying derivation.  SI
units and the exp(+j omega t) convention are used throughout.  It deliberately
keeps the individual calculation steps public so that results can be inspected
against the analytical formulae.
"""

from dataclasses import dataclass

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy import constants, optimize, special


CHI_11_PRIME = 1.8411837813406593
ETA_0 = np.sqrt(constants.mu_0 / constants.epsilon_0)


@dataclass(frozen=True)
class HornGeometry:
    """Conical-horn dimensions in metres and radians."""

    waveguide_radius: float = 12e-3
    aperture_radius: float = 35e-3
    flare_half_angle: float = np.deg2rad(15.0)

    def __post_init__(self):
        if self.waveguide_radius <= 0 or self.aperture_radius <= 0:
            raise ValueError("radii must be positive")
        if self.aperture_radius <= self.waveguide_radius:
            raise ValueError("aperture_radius must exceed waveguide_radius")
        if not 0 < self.flare_half_angle < np.pi / 2:
            raise ValueError("flare_half_angle must lie between 0 and pi/2")

    @property
    def throat_radius(self):
        return self.waveguide_radius / np.sin(self.flare_half_angle)

    @property
    def slant_length(self):
        return self.aperture_radius / np.sin(self.flare_half_angle)

    @property
    def axial_length(self):
        return ((self.aperture_radius - self.waveguide_radius)
                / np.tan(self.flare_half_angle))

    @property
    def cutoff_frequency(self):
        return (CHI_11_PRIME * constants.c
                / (2 * np.pi * self.waveguide_radius))


class ConicalHorn:
    """Single spherical-mode analytical conical-horn calculator."""

    def __init__(self, geometry=HornGeometry(), radial_order=220,
                 power_order=180):
        self.geometry = geometry
        self.radial_order = int(radial_order)
        self.power_order = int(power_order)
        if self.radial_order < 16 or self.power_order < 16:
            raise ValueError("quadrature orders must be at least 16")
        self.nu = self.solve_eigenvalue()
        x, w = leggauss(self.radial_order)
        self._rho = geometry.aperture_radius * (x + 1) / 2
        self._rho_weight = geometry.aperture_radius * w / 2

    @staticmethod
    def _legendre_theta_derivative(nu, theta, m=1):
        """Stable numerical derivative of Ferrers P_nu^m(cos(theta))."""
        step = 2e-6
        return ((special.lpmv(m, nu, np.cos(theta + step))
                 - special.lpmv(m, nu, np.cos(theta - step))) / (2 * step))

    def solve_eigenvalue(self, m=1):
        """Return the lowest TE_r eigenvalue satisfying dP/dtheta=0."""
        alpha = self.geometry.flare_half_angle
        estimate = CHI_11_PRIME / alpha - 0.5
        # The first root is near the Mehler--Heine estimate.  A dense scan is
        # easier for a reviewer to audit than a nested heuristic bracket.
        upper = max(2 * estimate + 4, m + 10)
        grid = np.linspace(m + 1e-6, upper, 4000)
        values = np.array([self._legendre_theta_derivative(v, alpha, m)
                           for v in grid])
        for left, right, fl, fr in zip(grid[:-1], grid[1:],
                                       values[:-1], values[1:]):
            if np.isfinite(fl) and np.isfinite(fr) and fl * fr < 0:
                return optimize.brentq(
                    lambda v: self._legendre_theta_derivative(v, alpha, m),
                    left, right, xtol=1e-12)
        raise RuntimeError("could not bracket the conical TE eigenvalue")

    @staticmethod
    def schelkunoff_hankel(nu, x, kind=2):
        """Schelkunoff spherical Hankel function for non-integer order."""
        x = np.asarray(x, dtype=float)
        sign = -1j if kind == 2 else 1j
        return np.sqrt(np.pi * x / 2) * (
            special.jv(nu + 0.5, x) + sign * special.yv(nu + 0.5, x))

    @classmethod
    def schelkunoff_hankel_derivative(cls, nu, x, kind=2):
        return (cls.schelkunoff_hankel(nu - 1, x, kind)
                - nu / np.asarray(x) * cls.schelkunoff_hankel(nu, x, kind))

    def waveguide_impedance(self, frequency):
        """TE11 guide impedance; frequencies must be above cutoff."""
        frequency = np.asarray(frequency, dtype=float)
        if np.any(frequency <= self.geometry.cutoff_frequency):
            raise ValueError("all frequencies must be above TE11 cutoff")
        k = 2 * np.pi * frequency / constants.c
        beta = np.sqrt(k * k - (CHI_11_PRIME /
                                self.geometry.waveguide_radius) ** 2)
        return ETA_0 * k / beta

    def aperture_spectrum(self, frequency, transverse_wavenumber,
                          phase_error=True):
        """Return A_0 and A_2 from Eq. (18), using Gauss--Legendre quadrature."""
        k = 2 * np.pi * float(frequency) / constants.c
        kt = np.atleast_1d(np.asarray(transverse_wavenumber, dtype=float))
        kc = CHI_11_PRIME / self.geometry.aperture_radius
        delta = (np.sqrt(self.geometry.slant_length ** 2 + self._rho ** 2)
                 - self.geometry.slant_length) if phase_error else 0.0
        phase = np.exp(-1j * k * delta)
        results = []
        for n in (0, 2):
            base = (special.jv(n, kc * self._rho) * phase * self._rho
                    * self._rho_weight)
            results.append(special.jv(n, np.outer(kt, self._rho)) @ base)
        if np.ndim(transverse_wavenumber) == 0:
            return results[0][0], results[1][0]
        return tuple(results)

    def aperture_norm(self):
        a = self.geometry.aperture_radius
        return (np.pi * a * a / 2 * (1 - 1 / CHI_11_PRIME ** 2)
                * special.jv(1, CHI_11_PRIME) ** 2)

    def complex_power(self, frequency, evanescent_limit=40.0):
        """Complex half-space power for E0=1, evaluated from Eq. (23)."""
        k = 2 * np.pi * float(frequency) / constants.c
        omega = 2 * np.pi * float(frequency)
        x, w = leggauss(self.power_order)

        u = np.pi * (x + 1) / 4
        wu = np.pi * w / 4
        kt = k * np.sin(u)
        a0, a2 = self.aperture_spectrum(frequency, kt)
        numerator = (2 * (k * k - kt * kt) * (abs(a0) ** 2 + abs(a2) ** 2)
                     + kt * kt * abs(a0 - a2) ** 2)
        visible = np.sum(wu * numerator * k * np.sin(u))

        umax = np.arccosh(max(evanescent_limit /
                              (k * self.geometry.aperture_radius), 1.000001))
        ue = umax * (x + 1) / 2
        we = umax * w / 2
        kte = k * np.cosh(ue)
        a0, a2 = self.aperture_spectrum(frequency, kte)
        numerator = (2 * (k * k - kte * kte) *
                     (abs(a0) ** 2 + abs(a2) ** 2)
                     + kte * kte * abs(a0 - a2) ** 2)
        evanescent = np.sum(we * numerator * (-1j * k * np.cosh(ue)))
        return np.pi / (8 * omega * constants.mu_0) * (visible + evanescent)

    def aperture_load(self, frequency, model="B"):
        """A: free-space load; B: spectral radiation-admittance load."""
        if model.upper() == "A":
            return complex(ETA_0)
        if model.upper() != "B":
            raise ValueError("load model must be 'A' or 'B'")
        admittance = 2 * np.conj(self.complex_power(frequency)) / self.aperture_norm()
        return 1 / admittance

    def input_impedance(self, frequency, load_model="B"):
        k = 2 * np.pi * float(frequency) / constants.c
        load = self.aperture_load(frequency, load_model)
        x2 = k * self.geometry.slant_length
        h2 = self.schelkunoff_hankel(self.nu, x2, 2)
        h2p = self.schelkunoff_hankel_derivative(self.nu, x2, 2)
        h1 = self.schelkunoff_hankel(self.nu, x2, 1)
        h1p = self.schelkunoff_hankel_derivative(self.nu, x2, 1)
        gamma = -(load * h2p + 1j * ETA_0 * h2) / (
            load * h1p + 1j * ETA_0 * h1)
        x1 = k * self.geometry.throat_radius
        numerator = (self.schelkunoff_hankel(self.nu, x1, 2)
                     + gamma * self.schelkunoff_hankel(self.nu, x1, 1))
        denominator = (self.schelkunoff_hankel_derivative(self.nu, x1, 2)
                       + gamma * self.schelkunoff_hankel_derivative(
                           self.nu, x1, 1))
        return -1j * ETA_0 * numerator / denominator

    def frequency_sweep(self, frequencies, load_model="B"):
        frequencies = np.asarray(frequencies, dtype=float)
        zg = self.waveguide_impedance(frequencies)
        zin = np.array([self.input_impedance(f, load_model)
                        for f in frequencies])
        s11 = (zin - zg) / (zin + zg)
        d0 = np.array([self.boresight_directivity(f) for f in frequencies])
        return {"frequency": frequencies, "Zg": zg, "Zin": zin, "S11": s11,
                "S11_dB": 20 * np.log10(np.maximum(abs(s11), 1e-15)),
                "directivity": d0, "realized_gain": (1 - abs(s11) ** 2) * d0}

    def boresight_directivity(self, frequency, phase_error=True):
        wavelength = constants.c / float(frequency)
        a0, _ = self.aperture_spectrum(frequency, 0.0, phase_error)
        return 4 * np.pi / wavelength ** 2 * np.pi ** 2 * abs(a0) ** 2 / self.aperture_norm()

    def beam(self, frequency, theta=None, phi=np.pi / 4):
        """Normalized E/H principal cuts and Ludwig-3 co/cross cut."""
        if theta is None:
            theta = np.linspace(-np.pi / 2, np.pi / 2, 721)
        theta = np.asarray(theta, dtype=float)
        k = 2 * np.pi * float(frequency) / constants.c
        a0, a2 = self.aperture_spectrum(frequency, k * np.abs(np.sin(theta)))
        e_cut = a0 - a2
        h_cut = np.cos(theta) * (a0 + a2)
        ex = -a2 * np.sin(2 * phi)
        ey = a0 + a2 * np.cos(2 * phi)
        e_theta = ex * np.cos(phi) + ey * np.sin(phi)
        e_phi = np.cos(theta) * (ey * np.cos(phi) - ex * np.sin(phi))
        co = e_theta * np.sin(phi) + e_phi * np.cos(phi)
        cross = e_theta * np.cos(phi) - e_phi * np.sin(phi)

        def normalized_db(value):
            value = abs(value) / np.max(abs(value))
            return 20 * np.log10(np.maximum(value, 1e-6))

        return {"theta": theta, "E_dB": normalized_db(e_cut),
                "H_dB": normalized_db(h_cut), "co_dB": normalized_db(co),
                "cross_dB": normalized_db(cross)}

    @staticmethod
    def hpbw(theta, pattern_db):
        """Full -3 dB beamwidth in radians for a boresight-centred cut."""
        theta, pattern_db = np.asarray(theta), np.asarray(pattern_db)
        positive = theta >= 0
        xp, yp = theta[positive], pattern_db[positive]
        below = np.flatnonzero(yp <= -3)
        if not len(below) or below[0] == 0:
            return np.nan
        i = below[0]
        crossing = np.interp(-3, yp[i - 1:i + 1][::-1], xp[i - 1:i + 1][::-1])
        return 2 * crossing
