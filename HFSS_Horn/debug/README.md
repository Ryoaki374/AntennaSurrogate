# Conical horn analytical calculator

`conical_horn.py` implements the single spherical-mode model from the supplied
derivation without invoking HFSS or another full-wave simulator.  All lengths
are in metres, frequencies in hertz, angles in radians, and impedances in ohms.

The public methods expose the intermediate eigenvalue, aperture spectrum,
complex power/load, input impedance, S11, directivity, and beam cuts so each
stage can be checked independently.  `conical_horn_demo.ipynb` is the intended
front end and produces the requested impedance, S11, beam, and gain plots.

Model A uses a free-space aperture load and is fast.  Model B evaluates the
propagating and evanescent plane-wave spectrum and is therefore slower.  This
is an engineering single-mode approximation: it omits throat discontinuity
susceptance, higher spherical modes, conductor loss, and rim diffraction.
