"""
Topography utilities: read CSV and interpolate topographic profile.
"""
import numpy as np

def read_topography(csv_file):
    """
    Read topography CSV with columns 'x', 'elevation'.
    Returns (x, z) numpy arrays sorted by x.
    """
    import pandas as pd
    df = pd.read_csv(csv_file)
    # Ensure sorted by x
    df = df.sort_values('x')
    x = df['x'].to_numpy()
    z = df['elevation'].to_numpy()
    return x, z

def interpolate_topo(x, z, xq):
    """
    Cubic-spline interpolate topography at query points xq.
    """
    from scipy.interpolate import interp1d
    f = interp1d(x, z, kind='cubic', fill_value="extrapolate")
    return f(xq)
