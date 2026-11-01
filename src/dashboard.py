import io, json, math, os, sys
import numpy as np, pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
P = lambda *a: os.path.join(ROOT, *a)
from src import nasa, ica, lab
from src.pipeline import Z_FEATS, Z_THR, N_REF

st.set_page_config(page_title="BatteryMRI", page_icon="🔋", layout="wide")

CY, GR, RD, AM, VI, SL = "#22D3EE", "#34D399", "#F87171", "#FBBF24", "#A78BFA", "#94A3B8"
LAY = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(17,26,46,0.6)", font=dict(color="#E6EDF7"),
           margin=dict(t=30, b=40, l=50, r=20), xaxis=dict(gridcolor="#1E2A44"), yaxis=dict(gridcolor="#1E2A44"))
st.markdown("""<style>
.block-container{padding-top:1.4rem;max-width:1250px}
.hero{background:linear-gradient(120deg,#0E2A47 0%,#14203A 55%,#1B1740 100%);border:1px solid #1E2A44;
 border-radius:16px;padding:22px 28px;margin-bottom:14px}
.hero h1{margin:0;font-size:2.1rem}.hero p{margin:.3rem 0 0;color:#9FB3D1;font-size:1.02rem}
.tile{background:#111A2E;border:1px solid #1E2A44;border-radius:12px;padding:12px 16px}
.tile b{font-size:1.6rem;color:#22D3EE}.tile span{display:block;color:#9FB3D1;font-size:.82rem}
.chip{display:inline-block;padding:6px 14px;border-radius:999px;font-weight:700;color:#0B1220}
.note{background:#111A2E;border-left:4px solid #FBBF24;padding:10px 14px;border-radius:6px;color:#D5DEEC}
</style>""", unsafe_allow_html=True)


def pf(fig, h=360, **kw):
    fig.update_layout(height=h, **{**LAY, **kw})
    st.plotly_chart(fig, width="stretch")


# ───────────────────────── data ─────────────────────────
@st.cache_data
def load():
    need = ["features.csv", "loco_predictions.csv", "deviation.csv", "metrics.json", "model.json", "ica_store.json", "grid.json"]
    if not all(os.path.exists(P("outputs", n)) for n in need):
        return None
    d = dict(F=pd.read_csv(P("outputs", "features.csv")), PR=pd.read_csv(P("outputs", "loco_predictions.csv")),
             DV=pd.read_csv(P("outputs", "deviation.csv")))
    for k, n in [("M", "metrics.json"), ("MOD", "model.json"), ("ICA", "ica_store.json"), ("GRID", "grid.json")]:
        d[k] = json.load(open(P("outputs", n)))
    d["GRID"] = np.array(d["GRID"])
    return d


D = load()
if D is None:
    st.error("No results found. Run `python run.py` first (takes ~30 s), then reload this page.")
    st.stop()
F, PR, DV, M, MOD, ICAS, GRID = D["F"], D["PR"], D["DV"], D["M"], D["MOD"], D["ICA"], D["GRID"]
PR["pred_smooth"] = PR.groupby("cell").predicted.transform(lambda s: s.rolling(5, min_periods=1).median())


def health(est):
    if est < nasa.EOL_AH: return "END OF LIFE", RD
    if est < 1.5: return "NEAR END OF LIFE", "#FB923C"
    if est < 1.7: return "AGEING", AM
    return "HEALTHY", GR


def chip(txt, col): return f"<span class='chip' style='background:{col}'>{txt}</span>"


def gauge(val, title):
    g = go.Figure(go.Indicator(mode="gauge+number", value=val, number=dict(suffix=" Ah", valueformat=".2f"),
        title=dict(text=title, font=dict(size=14)),
        gauge=dict(axis=dict(range=[1.0, 2.1]), bar=dict(color=CY),
                   steps=[dict(range=[1.0, 1.4], color="#5B2530"), dict(range=[1.4, 1.7], color="#5B4A1E"),
                          dict(range=[1.7, 2.1], color="#1D4B3A")],
                   threshold=dict(line=dict(color="white", width=3), value=nasa.EOL_AH))))
    g.update_layout(height=250, paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#E6EDF7"), margin=dict(t=50, b=10, l=25, r=25))
    return g


# ───────────────────────── header ─────────────────────────
prim = [k for k in M if k.endswith("(PRIMARY)")][0]
st.markdown(f"""<div class='hero'><h1>🔋 BatteryMRI</h1>
<p>Reads the electrochemical fingerprint of an EV-type Li-ion cell from an <b>ordinary charge</b> — no deep-discharge test —
and tells you how healthy it is. Built on <b>real</b> NASA ageing data; designed to run on real ESP32 logs.</p></div>""",
            unsafe_allow_html=True)
t1, t2, t3, t4 = st.columns(4)
for col, big, small in [(t1, f"{len(F)}", "real charge cycles analysed"), (t2, "4", "real 18650 cells, run to ~30% fade"),
                        (t3, f"{M[prim]['mean_mae']:.3f} Ah", "error on cells the model never saw"),
                        (t4, "0", "simulated numbers on this page")]:
    col.markdown(f"<div class='tile'><b>{big}</b><span>{small}</span></div>", unsafe_allow_html=True)
st.write("")

tabs = st.tabs(["⏱ Battery life time-machine", "🔌 Live charge session", "🧬 Cells compared", "✅ Does it work?",
                "🎛 Sensor-noise lab", "📤 Diagnose your data", "🛠 Build the hardware", "📖 How it works"])

# ═══════════════════ 1. TIME MACHINE ═══════════════════
with tabs[0]:
    st.markdown("Press **Play** and watch a real cell age. Every cycle shown is a real measured charge. The health estimate "
                "uses **only the charging curve** — the lab-measured capacity is plotted alongside purely for comparison.")
    c1, c2, c3 = st.columns([1.2, 1, 1])
    cell = c1.selectbox("Cell", nasa.CELLS, key="tm_cell", help="Estimates for this cell come from a model trained on the OTHER three cells.",
                        on_change=lambda: st.session_state.pop("tm_cycle", None))
    speed = c2.slider("Speed (cycles / second)", 1, 12, 4, key="tm_speed")
    play = c3.toggle("▶ Play", key="tm_play")
    S = F[F.cell == cell].reset_index(drop=True)
    pr = PR[PR.cell == cell].reset_index(drop=True)
    dv = DV[DV.cell == cell].reset_index(drop=True)
    nmax = len(S)
    opts = [int(x) for x in S.cycle]
    st.session_state.setdefault("tm_cycle", opts[-1])
    if st.session_state.tm_cycle not in opts: st.session_state.tm_cycle = opts[-1]

    @st.fragment(run_every=(1.0 / speed) if play else None)
    def time_machine():
        if play:
            j = opts.index(st.session_state.tm_cycle) + 1
            st.session_state.tm_cycle = opts[j] if j < len(opts) else opts[0]
        idx = opts.index(st.select_slider("Cycle (drag, or press Play)", options=opts, key="tm_cycle"))
        row, p = S.iloc[idx], pr.iloc[idx]
        est = float(p.pred_smooth)
        label, colr = health(est)
        z = float(dv.z.iloc[idx])
        left, right = st.columns([1.5, 1])
        with left:
            cyc = str(int(row.cycle))
            fig = go.Figure()
            for j in range(0, idx, max(1, nmax // 12)):
                fig.add_trace(go.Scatter(x=GRID, y=ICAS[cell][str(int(S.cycle[j]))], mode="lines",
                                         line=dict(color="rgba(148,163,184,.22)", width=1), showlegend=False, hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=GRID, y=ICAS[cell][str(int(S.cycle[0]))], name="New cell (first cycle)",
                                     line=dict(color=GR, width=2.5)))
            fig.add_trace(go.Scatter(x=GRID, y=ICAS[cell][cyc], name=f"Cycle {cyc} (now)", line=dict(color=RD, width=3.5)))
            fig.add_trace(go.Scatter(x=[row.peak_v], y=[row.peak_h], mode="markers", name="main peak", marker=dict(size=12, color="white", symbol="diamond")))
            fig.update_layout(xaxis_title="Charging voltage (V)", yaxis_title="dQ/dV  (Ah per V)", yaxis_range=[0, 6.2],
                              legend=dict(x=0.62, y=0.98, bgcolor="rgba(0,0,0,0)"), title=dict(text="The cell's fingerprint (real data)", font=dict(size=15)))
            pf(fig, 380)
        with right:
            st.plotly_chart(gauge(est, "Estimated capacity (from charge curve)"), width="stretch")
            st.markdown(chip(label, colr) + ("&nbsp;&nbsp;" + chip("FINGERPRINT DEVIATION", VI) if z > Z_THR else ""), unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Cycle", f"{int(row.cycle)} / {opts[-1]}")
        m2.metric("Estimated capacity", f"{est:.3f} Ah", f"{(est / nasa.RATED_AH) * 100:.0f}% of rated")
        m3.metric("Lab-measured (for comparison)", f"{row.cap:.3f} Ah", f"error {est - row.cap:+.3f} Ah", delta_color="off")
        m4.metric("Deviation score", f"{z:.1f}", f"threshold {Z_THR:.0f}", delta_color="off")
        tr = go.Figure()
        tr.add_trace(go.Scatter(x=S.cycle[:idx + 1], y=S.cap[:idx + 1], name="Measured capacity (lab)", mode="lines", line=dict(color=AM, width=2)))
        tr.add_trace(go.Scatter(x=pr.cycle[:idx + 1], y=pr.pred_smooth[:idx + 1], name="Estimated from charge curve", mode="lines", line=dict(color=CY, width=3)))
        tr.add_hline(y=nasa.EOL_AH, line_dash="dot", line_color=RD, annotation_text="end of life (1.4 Ah)")
        tr.update_layout(xaxis_title="Cycle", yaxis_title="Capacity (Ah)", xaxis_range=[1, opts[-1]], yaxis_range=[1.1, 2.1],
                         legend=dict(x=0.62, y=0.98, bgcolor="rgba(0,0,0,0)"))
        pf(tr, 300)

    time_machine()

# ═══════════════════ 2. LIVE CHARGE SESSION ═══════════════════
with tabs[1]:
    st.markdown("This is what the ESP32 logger would stream: **voltage, current, temperature** every few seconds. "
                "Replaying a real recorded charge — watch the fingerprint get built from the shaded voltage window.")
    a, b, c = st.columns([1, 1, 1])
    lc = a.selectbox("Cell", nasa.CELLS, key="ls_cell")
    lk = b.slider("Which charge (cycle)", 1, nasa.n_cycles(lc), 60, key="ls_k")
    lplay = c.toggle("▶ Stream", key="ls_play")
    sess = nasa.charge_session(lc, lk - 1)
    n = len(sess["t"]); step = max(1, n // 70)
    st.session_state.setdefault("ls_pos", 1)
    if st.session_state.get("ls_sig") != (lc, lk):
        st.session_state.ls_sig = (lc, lk); st.session_state.ls_pos = 1

    @st.fragment(run_every=0.25 if lplay else None)
    def stream():
        if lplay:
            st.session_state.ls_pos = min(n, st.session_state.ls_pos + step) if st.session_state.ls_pos < n else 1
        pos = st.slider("Samples received", 1, n, key="ls_pos")
        t, v, i, T = (sess[x][:pos] for x in ("t", "v", "i", "temp"))
        r1, r2, r3 = st.columns(3)
        r1.metric("Voltage", f"{v[-1]:.3f} V"); r2.metric("Current", f"{i[-1]:.2f} A"); r3.metric("Temperature", f"{T[-1]:.1f} °C")
        fg = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                           subplot_titles=("Voltage (V)", "Current (A)", "Temperature (°C)"))
        for r, (y, col) in enumerate([(v, CY), (i, AM), (T, RD)], 1):
            fg.add_trace(go.Scatter(x=t / 60, y=y, mode="lines", line=dict(color=col, width=2), showlegend=False), row=r, col=1)
        fg.add_hrect(y0=ica.V_LO, y1=ica.V_HI, fillcolor=GR, opacity=0.12, line_width=0, row=1, col=1)
        fg.update_xaxes(range=[0, sess["t"][-1] / 60], gridcolor="#1E2A44"); fg.update_yaxes(gridcolor="#1E2A44")
        fg.update_yaxes(range=[3.0, 4.35], row=1, col=1); fg.update_yaxes(range=[-0.2, 2.0], row=2, col=1)
        fg.update_xaxes(title_text="Minutes", row=3, col=1)
        pf(fg, 520, margin=dict(t=40, b=40, l=50, r=20))
        partial = dict(t=sess["t"][:pos], v=sess["v"][:pos], i=sess["i"][:pos], temp=sess["temp"][:pos])
        res = ica.ica_curve(partial)
        if res is None:
            st.info("Waiting for enough constant-current samples inside the 3.92 - 4.17 V window...")
        else:
            f = ica.features(*res)
            st.success(f"Fingerprint ready — main peak {f['peak_h']:.2f} Ah/V at {f['peak_v']:.3f} V  →  estimated capacity "
                       f"**{MOD['slope'] * f['peak_h'] + MOD['intercept']:.2f} Ah** (lab-measured for this cycle: {nasa.capacity(lc)[lk - 1]:.2f} Ah).")
    stream()

# ═══════════════════ 3. CELLS COMPARED ═══════════════════
with tabs[2]:
    fg = go.Figure()
    for c_, col in zip(nasa.CELLS, [CY, AM, GR, VI]):
        cp = nasa.capacity(c_); fg.add_trace(go.Scatter(x=np.arange(1, len(cp) + 1), y=cp, name=c_, mode="lines", line=dict(color=col, width=2.2)))
    fg.add_hline(y=nasa.EOL_AH, line_dash="dot", line_color=RD, annotation_text="end of life (1.4 Ah)")
    fg.update_layout(xaxis_title="Cycle", yaxis_title="Measured capacity (Ah)", title=dict(text="Four real cells, four different ageing paths", font=dict(size=15)))
    pf(fg, 340)
    st.caption("The sudden upward jumps are real: cells partially recover capacity after rest periods in the original experiment. "
               "They are genuine 'noise' that any health estimator has to live with.")
    cc1, cc2 = st.columns([1, 2])
    hc = cc1.selectbox("Fingerprint map for cell", nasa.CELLS, key="hm_cell")
    keys = [int(k) for k in sorted(ICAS[hc], key=int)]
    z = np.array([ICAS[hc][str(k)] for k in keys])
    hm = go.Figure(go.Heatmap(z=z, x=GRID, y=keys, colorscale="Inferno", colorbar=dict(title="dQ/dV")))
    hm.update_layout(xaxis_title="Charging voltage (V)", yaxis_title="Cycle", title=dict(text="Every cycle at once — bright band = main peak", font=dict(size=15)))
    pf(hm, 420)
    st.caption("Read it top to bottom: the bright band dims and slides as the cell ages. That dimming is the quantity the model uses.")
    summ = pd.DataFrame([dict(Cell=c_, Cycles=nasa.n_cycles(c_), **{"First capacity (Ah)": round(nasa.capacity(c_)[0], 3),
                                                                    "Last capacity (Ah)": round(nasa.capacity(c_)[-1], 3),
                                                                    "Fade %": round((1 - nasa.capacity(c_)[-1] / nasa.capacity(c_)[0]) * 100, 1)}) for c_ in nasa.CELLS])
    st.dataframe(summ, hide_index=True, width="stretch")

# ═══════════════════ 4. DOES IT WORK ═══════════════════
with tabs[3]:
    st.markdown("**Test method — leave-one-cell-out:** train on three cells, predict the fourth, which the model has never seen. Repeat for every cell.")
    names = [k for k in M if not k.startswith("_")]
    bar = go.Figure(go.Bar(x=[M[k]["mean_mae"] for k in names], y=names, orientation="h",
                           marker_color=[CY if k == prim else SL for k in names],
                           text=[f"{M[k]['mean_mae']:.3f} Ah" for k in names], textposition="outside"))
    bar.update_layout(xaxis_title="Average error on unseen cells (Ah) — lower is better", xaxis_range=[0, 0.1], yaxis=dict(autorange="reversed", gridcolor="#1E2A44"))
    pf(bar, 280)
    st.markdown(f"<div class='note'><b>Honest reading.</b> The ICA main-peak height is the best of the four, about "
                f"{(1 - M[prim]['mean_mae'] / M['Coulomb counting (charge in 3.92-4.17 V)']['mean_mae']) * 100:.0f}% better than plain charge counting and "
                f"{(1 - M[prim]['mean_mae'] / M['Cycle count only (no sensors)']['mean_mae']) * 100:.0f}% better than guessing from cycle count — a real but modest gain. "
                f"Using 7 features does <i>worse</i> (over-fitting: only 3 training cells). We chose the single-feature model after seeing these results, "
                f"so confirming it on more cells is the first thing a bigger study should do.</div>", unsafe_allow_html=True)
    st.write("")
    hv = st.radio("Held-out cell", nasa.CELLS, horizontal=True, key="pf_cell")
    s = PR[PR.cell == hv]
    pv = go.Figure()
    pv.add_trace(go.Scatter(x=s.cycle, y=s.actual, name="Measured (lab)", line=dict(color=AM, width=2)))
    pv.add_trace(go.Scatter(x=s.cycle, y=s.pred_smooth, name="Estimated from charge curve", line=dict(color=CY, width=3)))
    pv.add_hline(y=nasa.EOL_AH, line_dash="dot", line_color=RD)
    pv.update_layout(xaxis_title="Cycle", yaxis_title="Capacity (Ah)", legend=dict(x=0.6, y=0.98, bgcolor="rgba(0,0,0,0)"))
    pf(pv, 320)
    e1, e2 = st.columns(2)
    with e1:
        st.markdown("**End-of-life call (capacity < 1.4 Ah)**")
        st.dataframe(pd.DataFrame([dict(Cell=c_, **{"Actual cycle": str(v["actual_cycle"] or "never reached"), "Predicted cycle": str(v["predicted_cycle"] or "none")})
                                   for c_, v in M["_eol"].items()]), hide_index=True, width="stretch")
        st.caption("Off by roughly 17-21 cycles on three cells — useful as a trend, not as a precise countdown. B0007 never reached 1.4 Ah and the model correctly raised no alarm.")
    with e2:
        st.markdown("**First persistent fingerprint deviation** (vs the cell's own first 8 cycles)")
        st.dataframe(pd.DataFrame([dict(Cell=c_, Cycle=int(v["cycle"]), **{"Capacity lost then (%)": round(v["capacity_loss_pct"], 1)}) for c_, v in M["_first_alert"].items()]),
                     hide_index=True, width="stretch")
        st.caption("The alert fires after 0-8% capacity loss. B0007 fired at ~0% — that could be an early fingerprint change or a false alarm; 4 cells cannot tell us which.")

# ═══════════════════ 5. SENSOR-NOISE LAB ═══════════════════
with tabs[4]:
    st.markdown("**Why this matters for your hardware:** dQ/dV is a *derivative*, and derivatives amplify noise. "
                "Here we add cheap-sensor noise to the real voltage data and see what happens to the health estimate.")
    l1, l2, l3 = st.columns(3)
    nmv = l1.slider("Voltage noise (mV, 1 sigma)", 0.0, 20.0, 5.0, 0.5)
    qmv = l2.select_slider("ADC step", [0.0, 0.125, 1.0, 4.0], 4.0, format_func=lambda x: "ideal" if x == 0 else f"{x} mV",
                           help="INA219 bus voltage steps in 4 mV; ADS1115 in 0.125 mV.")
    smo = l3.slider("Smoothing strength", 1.0, 12.0, 3.0, 0.5, help="Wider smoothing = less noise but a blurrier peak.")
    lc2, lk2 = st.columns(2)
    lcell = lc2.selectbox("Cell", nasa.CELLS, key="lab_cell"); lcyc = lk2.slider("Cycle", 1, nasa.n_cycles(lcell), 80, key="lab_cyc")
    sess = nasa.charge_session(lcell, lcyc - 1)
    clean = ica.ica_curve(sess); noisy = ica.ica_curve(sess, smooth=smo, v_noise=nmv / 1000, quant=qmv / 1000, rng=np.random.default_rng(1))
    fg = go.Figure()
    if clean: fg.add_trace(go.Scatter(x=GRID, y=clean[1], name="Clean (real data)", line=dict(color=GR, width=3)))
    if noisy: fg.add_trace(go.Scatter(x=GRID, y=noisy[1], name="With your sensor settings", line=dict(color=RD, width=2.5)))
    fg.update_layout(xaxis_title="Charging voltage (V)", yaxis_title="dQ/dV", legend=dict(x=0.62, y=0.98, bgcolor="rgba(0,0,0,0)"))
    pf(fg, 320)
    if noisy:
        fe = ica.features(*noisy); est = MOD["slope"] * fe["peak_h"] + MOD["intercept"]
        st.metric("Estimated capacity with this sensor", f"{est:.3f} Ah", f"error {est - nasa.capacity(lcell)[lcyc - 1]:+.3f} Ah vs lab", delta_color="off")

    @st.cache_data(show_spinner="Re-analysing real cycles with degraded sensor...")
    def sweep_all(q):
        out = []
        for sm in (3.0, 6.0, 10.0):
            r = lab.sweep(F, [0, 2, 5, 10, 20], q, every=6, smooth=sm)
            g = r.groupby("noise_mv").mae.mean().reset_index(); g["smoothing"] = f"smoothing {sm:g}"; out.append(g)
        return pd.concat(out)
    sw = sweep_all(qmv)
    fg2 = go.Figure()
    for (nm, g), col in zip(sw.groupby("smoothing"), [RD, AM, GR]):
        fg2.add_trace(go.Scatter(x=g.noise_mv, y=g.mae, name=nm, mode="lines+markers", line=dict(color=col, width=2.5)))
    fg2.update_layout(xaxis_title="Voltage noise (mV)", yaxis_title="Average capacity error (Ah)", title=dict(text="Error vs sensor noise (all 4 cells, every 6th cycle)", font=dict(size=15)))
    pf(fg2, 320)
    st.markdown("<div class='note'><b>Take-away:</b> with light smoothing the method breaks above ~5 mV noise; stronger smoothing restores it. "
                "So the hardware plan is: a 16-bit ADC (ADS1115, 0.125 mV step) or an averaged INA219, plus the right smoothing. "
                "This is a genuine engineering finding that only exists because we use real data.</div>", unsafe_allow_html=True)

# ═══════════════════ 6. DIAGNOSE YOUR DATA ═══════════════════
with tabs[5]:
    st.markdown("Upload **one charge log** (CSV with columns `t, v, i, temp` — seconds, volts, amps, °C). "
                "This is exactly the file the ESP32 logger produces.")
    d1, d2, d3 = st.columns(3)
    sc_ = d1.selectbox("Need a sample? Cell", nasa.CELLS, key="up_sc"); sk = d2.slider("Cycle", 1, nasa.n_cycles(sc_), 120, key="up_sk")
    s_ = nasa.charge_session(sc_, sk - 1)
    d3.download_button("⬇ Download this real charge as CSV", pd.DataFrame(dict(t=s_["t"], v=s_["v"], i=s_["i"], temp=s_["temp"])).to_csv(index=False),
                       f"{sc_}_cycle{sk}_charge.csv", "text/csv")
    st.caption("Heads-up: the sample files come from the same four cells the final model was trained on, so estimates on them are optimistic. "
               "The honest accuracy numbers are on the 'Does it work?' tab (cells held out).")
    up = st.file_uploader("Charge log (.csv)", type="csv")
    if up is not None:
        try:
            df = pd.read_csv(up)
            df.columns = [c.strip().lower() for c in df.columns]
            df = df.rename(columns={"time": "t", "time_s": "t", "voltage": "v", "current": "i", "temperature": "temp"})
            miss = {"t", "v", "i", "temp"} - set(df.columns)
        except Exception as e:
            df, miss = None, {f"(could not read file: {e})"}
        if miss:
            st.error("Missing columns: " + ", ".join(sorted(miss)))
        else:
            sess = {k: df[k].to_numpy(float) for k in ("t", "v", "i", "temp")}
            cc = ica.cc_level(sess["v"], sess["i"]); res = ica.ica_curve(sess)
            if res is None:
                st.error("Could not build a fingerprint: need a constant-current charge crossing the full 3.92 - 4.17 V window "
                         "with at least 40 usable samples.")
            else:
                fe = ica.features(*res); est = MOD["slope"] * fe["peak_h"] + MOD["intercept"]
                dT = abs(fe["mean_temp"] - MOD["train_temp"])
                if abs(cc - 1.5) > 0.25:
                    st.markdown(chip("CALIBRATION NEEDED", VI), unsafe_allow_html=True)
                    st.warning(f"This charge ran at {cc:.2f} A; the model was trained at 1.5 A (0.75 C). The fingerprint is shown, but a capacity "
                               "estimate would not be trustworthy. Log a few charges of your own healthy cell and use them as its reference.")
                else:
                    lab_, col_ = health(est)
                    if dT > 8: lab_, col_ = "TEMPERATURE MISMATCH", AM
                    st.markdown(chip(lab_, col_), unsafe_allow_html=True)
                    u1, u2, u3 = st.columns(3)
                    u1.metric("Estimated capacity", f"{est:.2f} Ah", f"{est / nasa.RATED_AH * 100:.0f}% of 2 Ah rated")
                    u2.metric("Charge current", f"{cc:.2f} A"); u3.metric("Mean temperature", f"{fe['mean_temp']:.1f} °C", f"model trained at {MOD['train_temp']:.0f} °C", delta_color="off")
                    if dT > 8: st.warning("Temperature differs from the 24 °C training data by more than 8 °C — retest near room temperature.")
                fg = go.Figure()
                h0 = np.mean([ICAS[c_][str(k)] for c_ in nasa.CELLS for k in range(1, 6) if str(k) in ICAS[c_]], axis=0)
                fg.add_trace(go.Scatter(x=GRID, y=h0, name="Healthy reference (NASA, first cycles)", line=dict(color=GR, width=2.5)))
                fg.add_trace(go.Scatter(x=GRID, y=res[1], name="Your upload", line=dict(color=RD, width=3)))
                fg.update_layout(xaxis_title="Charging voltage (V)", yaxis_title="dQ/dV", legend=dict(x=0.62, y=0.98, bgcolor="rgba(0,0,0,0)"))
                pf(fg, 340)
                st.dataframe(pd.DataFrame([dict(Feature=ica.NICE[k], Value=round(float(fe[k]), 3)) for k in ica.FEATURES]), hide_index=True, width="stretch")

# ═══════════════════ 7. HARDWARE ═══════════════════
with tabs[6]:
    st.markdown("### From NASA's lab to your bench")
    st.markdown("The software above is already hardware-ready: it consumes the same four signals a ~₹1,000 logger can record. "
                "The real-hardware work is to **log your own cell** and check that it shows the same behaviour.")
    st.dataframe(pd.DataFrame([
        ("ESP32 DevKit", "Reads sensors, streams CSV over USB/WiFi", "₹450"),
        ("INA219 (or ADS1115 + shunt)", "Battery voltage + charge current (see noise lab: ADC quality matters)", "₹150-300"),
        ("DS18B20 temperature probe", "Cell temperature", "₹80"),
        ("Protected 18650 cell + holder", "The battery under test (use protected cells)", "₹250-400"),
        ("TP4056 charger module w/ protection", "Does the actual charging — the ESP32 only watches", "₹60"),
        ("Breadboard, wires, enclosure", "Make it look and feel like a product", "₹300"),
    ], columns=["Part", "Role", "Approx. cost"]), hide_index=True, width="stretch")
    st.markdown("**Pipeline on the bench:**  charger → *[ INA219 in series ]* → cell,  DS18B20 taped to the cell,  ESP32 → USB → laptop → this dashboard.")
    st.markdown("<div class='note'><b>Honest scope.</b> A TP4056 charges at ≤1 A, not NASA's 1.5 A, so the first bench task is to log a few charges "
                "of your own healthy cell and use them as its reference (the dashboard already flags 'calibration needed' for other currents). "
                "A cell ages over weeks, so the hardware demo is: <i>log real charges live, show the fingerprint building in real time, and compare "
                "a fresh cell against an old/used one</i>. Safety: charge only protected cells, supervised.</div>", unsafe_allow_html=True)
    st.markdown("#### ESP32 logger firmware (template — not yet tested on a real cell)")
    fw = P("firmware", "esp32_logger.ino")
    if os.path.exists(fw): st.code(open(fw).read(), language="cpp")

# ═══════════════════ 8. HOW IT WORKS ═══════════════════
with tabs[7]:
    st.markdown("""
#### In one sentence
A Li-ion cell's voltage curve during charging has bumps (peaks) that shrink and shift as the cell wears out. We measure those bumps and turn them into a health number.

#### The pipeline
1. **Measure** voltage, current and temperature during a normal constant-current charge.
2. **Convert** to *incremental capacity*: dQ/dV = charge added per volt. Plateaus in the voltage curve become peaks.
3. **Extract** a few numbers: peak height, position, areas.
4. **Estimate** capacity from the main-peak height with a small linear model trained on other cells.
5. **Report** health state, plus a deviation alert versus the cell's own first 8 cycles.

#### Glossary
- **Capacity (Ah)** – charge the cell can hold. New ≈ 2 Ah here; **end of life** at 1.4 Ah (30% fade).
- **ICA / dQ/dV** – the "fingerprint" curve. **SOH** – state of health = capacity relative to new.
- **Leave-one-cell-out** – train on 3 cells, test on the 4th: tests generalisation, not memorisation.

#### Limits — say these first, before anyone asks
- Only **4 cells**, one chemistry, one temperature (24 °C), one charge current (1.5 A).
- Error is **~0.05 Ah (~2.5% of rated)**; end-of-life timing is only good to ±~20 cycles.
- Results come from NASA's public dataset (PCoE). **Our own hardware logs are the next step**, not yet done.
- A deviation alert says *the fingerprint changed*, not *why*.
""")
st.markdown("---")
st.caption("BatteryMRI · UVCE ECE Major Project 2025-26 · Data: NASA Prognostics Center of Excellence Li-ion battery ageing dataset")
