"""Real NASA PCoE Li-ion 18650 ageing data (cells B0005, B0006, B0007, B0018, 24 C, 2 Ah rated).
Each cell: ~130-170 charge/discharge cycles until capacity faded ~30%.
Charge: 1.5 A constant current to 4.2 V then constant voltage.  Discharge: 2 A to 2.7 V.
The discharge 'Capacity' (Ah) is the measured ground truth; it is NEVER a model input."""
import os, numpy as np
CELLS = ["B0005", "B0006", "B0007", "B0018"]
RATED_AH = 2.0
EOL_AH = 1.4          # NASA's usual end-of-life criterion: 30% capacity fade
_Z = None

def _z():
    global _z_cache
    if "_z_cache" not in globals():
        p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "nasa_cells.npz")
        globals()["_z_cache"] = np.load(p)
    return globals()["_z_cache"]

def n_cycles(cell): return int(_z()[cell + "_dis_n"])

def capacity(cell):
    z = _z(); return np.array([float(z[f"{cell}_dis_{k}_cap"]) for k in range(n_cycles(cell))])

def charge_session(cell, k):
    """Charge session that preceded discharge k -> dict(t [s], v [V], i [A], temp [C])."""
    z = _z(); j = int(z[cell + "_pair"][k])
    g = lambda f: z[f"{cell}_chg_{j}_{f}"].astype(float)
    o = np.argsort(g("t"))
    return dict(t=g("t")[o], v=g("v")[o], i=g("i")[o], temp=g("T")[o])

def discharge_session(cell, k):
    z = _z(); g = lambda f: z[f"{cell}_dis_{k}_{f}"].astype(float)
    return dict(t=g("t"), v=g("v"), i=g("i"), temp=g("T"))
