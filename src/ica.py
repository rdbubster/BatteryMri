"""Incremental Capacity Analysis (dQ/dV) on the constant-current part of a charge.
Works on ANY charge log with columns t[s], v[V], i[A], temp[C] -> same code for NASA data and ESP32 logs."""
import numpy as np
from scipy.ndimage import gaussian_filter1d

V_LO, V_HI = 3.92, 4.17            # voltage window used everywhere (covered by every healthy+aged cycle)
GRID = np.linspace(V_LO, V_HI, 100)
PEAK_LO, PEAK_HI = 3.94, 4.07

def cc_level(v, i):
    """Constant-current level of a charge log = median current while the cell is between 3.7 and 4.1 V."""
    m = (v > 3.7) & (v < 4.1) & (i > 0.05)
    return float(np.median(i[m])) if m.sum() > 10 else float("nan")

def cc_mask(v, i):
    c = cc_level(v, i)
    return (np.abs(i - c) < 0.12 * c) if c == c else np.zeros(len(i), bool)

def ica_curve(s, smooth=3.0, v_noise=0.0, quant=0.0, rng=None):
    """Return (Q_on_grid [Ah], dQ/dV_on_grid [Ah/V], mean_temp) or None if the session cannot be used.
    v_noise / quant let the robustness lab emulate a cheap voltage sensor (Gaussian noise, ADC step in V)."""
    t, v, i, T = s["t"], s["v"].copy(), s["i"], s["temp"]
    m = cc_mask(v, i)
    if m.sum() < 40: return None
    t, v, i, T = t[m], v[m], i[m], T[m]
    if v_noise > 0:
        v = v + (rng or np.random.default_rng(0)).normal(0, v_noise, len(v))
    if quant > 0:
        v = np.round(v / quant) * quant
    q = np.cumsum(i * np.gradient(t)) / 3600.0
    vm = np.maximum.accumulate(v)                   # make V monotone (removes tiny dips)
    u, ix = np.unique(vm, return_index=True)
    if len(u) < 20 or u.min() > V_LO + 0.01 or u.max() < V_HI: return None
    qg = gaussian_filter1d(np.interp(GRID, u, q[ix]), smooth)
    return qg, np.gradient(qg, GRID), float(np.mean(T))

FEATURES = ["peak_h", "peak_v", "q_window", "area_low", "area_high", "plateau", "ratio", "mean_temp"]
NICE = {"peak_h": "Main peak height", "peak_v": "Main peak position", "q_window": "Charge in window (Ah)",
        "area_low": "Area 3.92-4.02 V", "area_high": "Area 4.02-4.17 V",
        "plateau": "Plateau height 4.10-4.16 V", "ratio": "Peak / plateau ratio", "mean_temp": "Mean temperature"}

def features(qg, d, mean_temp):
    pk = (GRID >= PEAK_LO) & (GRID <= PEAK_HI)
    ip = np.argmax(np.where(pk, d, -np.inf))
    lo = (GRID < 4.02); hi = ~lo
    pl = (GRID >= 4.10) & (GRID <= 4.16)
    f = dict(peak_h=d[ip], peak_v=GRID[ip], q_window=qg[-1] - qg[0],
             area_low=qg[lo][-1] - qg[lo][0], area_high=qg[hi][-1] - qg[hi][0],
             plateau=float(d[pl].mean()), mean_temp=mean_temp)
    f["ratio"] = f["peak_h"] / f["plateau"]
    return f
