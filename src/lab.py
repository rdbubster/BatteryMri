"""Robustness lab: degrade the REAL charge logs with a cheap-sensor model and see how the SOH estimate copes."""
import numpy as np, pandas as pd
from . import nasa, ica

def sweep(feat_df, noises_mv, quant_mv=0.0, every=4, smooth=3.0, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for hold in nasa.CELLS:
        tr = feat_df[feat_df.cell != hold]
        slope, icpt = np.polyfit(tr.peak_h, tr.cap, 1)
        te = feat_df[(feat_df.cell == hold) & (feat_df.cycle % every == 0)]
        for nmv in noises_mv:
            errs, miss = [], 0
            for cy in te.cycle:
                r = ica.ica_curve(nasa.charge_session(hold, int(cy) - 1), smooth=smooth,
                                  v_noise=nmv / 1000.0, quant=quant_mv / 1000.0, rng=rng)
                if r is None: miss += 1; continue
                f = ica.features(*r)
                errs.append(abs(slope * f["peak_h"] + icpt - float(te[te.cycle == cy].cap.iloc[0])))
            rows.append(dict(cell=hold, noise_mv=nmv, mae=np.mean(errs) if errs else np.nan, unusable=miss, n=len(te)))
    return pd.DataFrame(rows)
