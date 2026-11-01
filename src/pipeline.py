"""Run:  python run.py     -> builds every file the dashboard reads (outputs/*.json|csv).
Everything is computed from the REAL NASA discharge/charge logs in data/nasa_cells.npz."""
import json, os, numpy as np, pandas as pd
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from . import nasa, ica

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs")
N_REF = 8                      # first 8 cycles of a cell = its own "healthy reference"
Z_FEATS = ["peak_h", "peak_v", "area_low", "plateau", "ratio"]
Z_THR = 3.0
PRIMARY = ["peak_h"]           # single, interpretable feature
MODELS = {
    "Cycle count only (no sensors)": ["cycle"],
    "Coulomb counting (charge in 3.92-4.17 V)": ["q_window"],
    "ICA main-peak height (PRIMARY)": ["peak_h"],
    "ICA, 7 features": ["peak_h", "peak_v", "q_window", "area_low", "area_high", "plateau", "ratio"],
}

def build_features():
    rows, store = [], {}
    for c in nasa.CELLS:
        cap = nasa.capacity(c)
        for k in range(nasa.n_cycles(c)):
            r = ica.ica_curve(nasa.charge_session(c, k))
            if r is None:
                continue
            f = ica.features(*r); f.update(cell=c, cycle=k + 1, cap=float(cap[k]))
            rows.append(f)
            store.setdefault(c, {})[str(k + 1)] = [round(float(x), 4) for x in r[1]]
    return pd.DataFrame(rows), store

def fit(cols, tr):
    return make_pipeline(StandardScaler(), Ridge(1.0)).fit(tr[cols], tr.cap)

def first_persistent(flags, run=2):
    for i in range(len(flags) - run + 1):
        if all(flags[i:i + run]): return i
    return None

def run_pipeline():
    os.makedirs(OUT, exist_ok=True)
    df, store = build_features()
    df.to_csv(os.path.join(OUT, "features.csv"), index=False)
    json.dump(store, open(os.path.join(OUT, "ica_store.json"), "w"))
    json.dump([round(float(x), 4) for x in ica.GRID], open(os.path.join(OUT, "grid.json"), "w"))

    # ---- leave-one-cell-out: train on 3 cells, predict the 4th (never seen) ----
    pred_rows, metrics = [], {}
    for name, cols in MODELS.items():
        maes = {}
        for hold in nasa.CELLS:
            tr, te = df[df.cell != hold], df[df.cell == hold]
            p = fit(cols, tr).predict(te[cols])
            maes[hold] = float(np.abs(p - te.cap).mean())
            if name.endswith("(PRIMARY)"):
                for cy, a, b in zip(te.cycle, te.cap, p):
                    pred_rows.append(dict(cell=hold, cycle=int(cy), actual=a, predicted=float(b)))
        metrics[name] = dict(per_cell=maes, mean_mae=float(np.mean(list(maes.values()))))
    pr = pd.DataFrame(pred_rows)
    pr["pred_smooth"] = pr.groupby("cell").predicted.transform(lambda s: s.rolling(5, min_periods=1).median())
    pr.to_csv(os.path.join(OUT, "loco_predictions.csv"), index=False)

    # ---- end-of-life call (1.4 Ah) on held-out cells ----
    eol = {}
    for c in nasa.CELLS:
        s = pr[pr.cell == c]
        a = s[s.actual < nasa.EOL_AH].cycle.min()
        p = s[s.pred_smooth < nasa.EOL_AH].cycle.min()
        eol[c] = dict(actual_cycle=None if np.isnan(a) else int(a), predicted_cycle=None if np.isnan(p) else int(p))
    metrics["_eol"] = eol

    # ---- deviation alert vs each cell's own first-8-cycle reference ----
    dev = []
    for c in nasa.CELLS:
        s = df[df.cell == c].reset_index(drop=True)
        X = s[Z_FEATS].to_numpy(float)
        mu, sg = X[:N_REF].mean(0), X[:N_REF].std(0, ddof=1) + 1e-9
        z = np.abs((X - mu) / sg).mean(1)
        for cy, zz, cp in zip(s.cycle, z, s.cap): dev.append(dict(cell=c, cycle=int(cy), z=float(zz), cap=float(cp)))
    dv = pd.DataFrame(dev); dv.to_csv(os.path.join(OUT, "deviation.csv"), index=False)
    first = {}
    for c in nasa.CELLS:
        s = dv[dv.cell == c].reset_index(drop=True)
        i = first_persistent((s.z > Z_THR).tolist())
        c0 = s.cap[:N_REF].mean()
        first[c] = None if i is None else dict(cycle=int(s.cycle[i]), capacity_loss_pct=float((1 - s.cap[i] / c0) * 100))
    metrics["_first_alert"] = first

    # ---- final deployable model: trained on all 4 cells (used by the Upload tab) ----
    m = fit(PRIMARY, df)
    sc, rg = m.named_steps["standardscaler"], m.named_steps["ridge"]
    # collapse scaler+ridge into plain slope/intercept on peak_h (so the dashboard needs no pickle)
    slope = float(rg.coef_[0] / sc.scale_[0]); icpt = float(rg.intercept_ - slope * sc.mean_[0])
    healthy = df[df.cycle <= N_REF]
    json.dump(dict(slope=slope, intercept=icpt, feature="peak_h", train_temp=24.0,
                   healthy_peak_h=float(healthy.peak_h.mean()), grid=[float(x) for x in ica.GRID],
                   z_mu=healthy[Z_FEATS].mean().tolist(), z_sigma=healthy[Z_FEATS].std().tolist(), z_feats=Z_FEATS),
              open(os.path.join(OUT, "model.json"), "w"), indent=1)
    json.dump(metrics, open(os.path.join(OUT, "metrics.json"), "w"), indent=1)
    print("Cycles used:", len(df), "of", sum(nasa.n_cycles(c) for c in nasa.CELLS))
    for k, v in metrics.items():
        if not k.startswith("_"): print(f"  {k:45s} mean MAE {v['mean_mae']:.3f} Ah")
    print("  EOL (actual vs predicted cycle):", eol)
    print("  first persistent deviation:", first)
    return df, metrics
