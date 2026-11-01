"""One-off: convert NASA PCoE .mat files into a compact npz (so the project needs no big downloads).
Usage: python tools/extract_nasa.py <folder-with-B0005.mat...> """
import sys, os, numpy as np, scipy.io as sio
src = sys.argv[1]; out = {}
for cell in ["B0005","B0006","B0007","B0018"]:
    m = sio.loadmat(os.path.join(src, cell+".mat"), simplify_cells=True)[cell]["cycle"]
    dis, chg, pair = [], [], []
    for c in m:
        d = c["data"]
        if c["type"] == "discharge":
            pair.append(len(chg)-1)
            dis.append(dict(t=d["Time"], v=d["Voltage_measured"], i=d["Current_measured"],
                            T=d["Temperature_measured"], cap=float(d["Capacity"]), amb=float(c["ambient_temperature"])))
        elif c["type"] == "charge":
            chg.append(dict(t=d["Time"], v=d["Voltage_measured"], i=d["Current_measured"],
                            T=d["Temperature_measured"], amb=float(c["ambient_temperature"])))
    out[f"{cell}_pair"] = np.array(pair)  # index of the charge session just before each discharge
    for k, lst in (("dis", dis), ("chg", chg)):
        out[f"{cell}_{k}_n"] = len(lst)
        for j, r in enumerate(lst):
            for f, a in r.items():
                out[f"{cell}_{k}_{j}_{f}"] = np.asarray(a, dtype=np.float32 if f in "tviT" else np.float64)
np.savez_compressed("data/nasa_cells.npz", **out)
print("saved", os.path.getsize("data/nasa_cells.npz")/1e6, "MB")
