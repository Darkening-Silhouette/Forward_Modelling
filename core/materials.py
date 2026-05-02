"""
Material properties and conversions.
"""
import numpy as np

# Physical constants
mu0 = 4 * np.pi * 1e-7
eps0 = 8.854187817e-12

def resistivity_to_conductivity(rho):
    """Convert resistivity (ohm·m) to conductivity (S/m)."""
    return 1.0 / rho

def permittivity_from_epsilon_r(epsilon_r):
    """Convert relative permittivity to absolute permittivity (F/m)."""
    from core.materials import eps0
    return epsilon_r * eps0

def velocity_to_permratio(vp, freq):
    """
    Compute relative permittivity from velocity (approximate).
    (Not physically accurate for GPR; placeholder.)
    """
    # This is a placeholder; actual relation is different.
    return (3e8 / vp)**2
