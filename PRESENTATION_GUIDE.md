# BatteryMRI — Presentation Guide

## 1. The project in plain English (say this first, 30 seconds)

> "EV batteries wear out, and owners get no warning. The standard way to measure health is a slow, full discharge test.
> We found that the **charging curve itself carries the information**: the voltage curve has bumps that shrink and
> shift as the cell ages. Our software turns an ordinary charge into a health estimate — no deep-discharge test.
> We built and validated it on **real NASA battery-ageing data (4 cells, 627 real charge cycles)** and it is designed to
> run on logs from our own ESP32 hardware logger."

**Analogy that works:** a doctor's MRI doesn't open the patient — it reads signals from outside. We don't open the battery or
drain it — we read the signal it gives while charging.

## 2. How it works — 5 steps

1. **Measure** voltage, current, temperature while the cell charges at constant current.
2. **Convert** to *incremental capacity*, dQ/dV = "how much charge goes in per volt". Flat spots in the voltage curve become peaks.
3. **Fingerprint:** the main peak (around 4.0 V) gets lower and shifts as the cell ages.
4. **Estimate** capacity from the peak height using a small model trained on *other* cells.
5. **Report:** healthy / ageing / near end-of-life, plus an alert if the fingerprint deviates from the cell's own first 8 cycles.

## 3. Numbers you can quote (all produced by the code in this folder)

| Claim | Number |
|---|---|
| Real data | 4 NASA 18650 cells, 627 usable charge cycles, run to ~30% capacity fade |
| Test method | Leave-one-cell-out: train on 3 cells, predict the 4th (never seen) |
| Error, ICA peak height (our method) | **0.046 Ah** average (about 2.3% of the 2 Ah rating) |
| Error, plain coulomb counting | 0.056 Ah |
| Error, guessing from cycle count | 0.082 Ah |
| 7-feature model | 0.062 Ah — *worse*, over-fits with 3 training cells |
| End-of-life (1.4 Ah) timing | Off by about 17–21 cycles on 3 cells; the 4th never reached it and correctly raised no alarm |
| Sensor-noise finding | With light smoothing the estimate breaks above ~5 mV voltage noise (error ~0.13 Ah); stronger smoothing restores it (~0.06 Ah at 10 mV) |

## 4. Live demo script (about 4 minutes)

1. **Time-machine tab** — pick a cell, press **Play**. Say: "Every cycle is a real measured charge. The red curve is the fingerprint dimming as the cell ages. The cyan line is our estimate from the charge curve only; yellow is the lab-measured capacity, shown just for comparison."
2. **Live charge session tab** — press **Stream**. "This is what the ESP32 would stream: voltage, current, temperature. The fingerprint appears from the green voltage window."
3. **Does it work? tab** — show the bar chart. "Our method beats both baselines, on cells it never saw. The gain is real but modest, and we say so."
4. **Sensor-noise lab** — drag noise to 10 mV with smoothing 3, then raise smoothing. "This tells us what hardware we need: a better ADC and the right smoothing."
5. **Diagnose your data** — download a real charge CSV, upload it back. "Same file format our ESP32 logger produces."
6. **Build the hardware tab** — show parts list (~₹1,200–1,700) and the firmware.

## 5. Questions she will probably ask — honest answers

**"Couldn't I build this with a prompt?"**
A prompt can produce a dashboard. It cannot produce the real measurements. Ours rests on 627 real charge cycles, a held-out validation, baselines that *beat* some of our own ideas (the 7-feature model lost), and a sensor-noise finding that only exists because the data is real. Be ready to explain any chart and the code behind it.

**"Is this real or simulated?"**
Real NASA data (Saha & Goebel, NASA Prognostics Data Repository). Nothing on the dashboard is simulated. **Our own hardware logs are the next milestone and are not done yet** — say this before she asks.

**"Where's the hardware?"**
Honest answer: the ESP32 logger firmware is written but untested on a real cell. Plan for the next weeks: build the logger, log charges of a fresh and a used 18650, check whether we see the same peak behaviour, and calibrate (our charger will not be 1.5 A, so we need our own reference).

**"Why only 4 cells?"** It is the standard public dataset for this kind of study. It is also why we are cautious: one chemistry, one temperature (24 °C), one current (1.5 A). More cells and conditions are future work.

**"Why did you choose the single-feature model?"** We tested four models; the 7-feature one over-fitted. We picked the single peak-height model *after* seeing results, so it needs confirming on more cells. Say this proactively — it earns trust.

**"What is new here?"** Health estimation from the charge (what an EV actually does) with leave-one-cell-out validation, plus a quantified sensor-noise requirement for low-cost hardware. We do not claim a new algorithm — ICA itself is established.

**"Why is the end-of-life prediction off by ~20 cycles?"** Capacity partly recovers after rest periods in the original experiment (the jumps in the data), and only 3 cells train each model. It is a trend indicator, not a precise countdown.

## 6. Limits to state up front

- 4 cells, one chemistry, 24 °C, 1.5 A charge.
- ~0.05 Ah error; end-of-life timing only roughly right.
- A deviation alert says the fingerprint changed, not why.
- No hardware results yet.

## 7. Next steps (what to propose to her)

1. **Weeks 1–2:** build the ESP32 + INA219/ADS1115 + DS18B20 logger; log fresh vs used 18650 cells (protected cells, supervised).
2. **Weeks 3–4:** calibrate for our charger current; compare real-hardware fingerprints with NASA's.
3. **Weeks 5+:** link back to the original idea — a smart-charging unit that uses the health estimate to adapt cut-off voltage and pause charging when hot.

## 8. Reference

B. Saha and K. Goebel (2007). "Battery Data Set", NASA Prognostics Data Repository, NASA Ames Research Center. Verify your copy against the official NASA download before citing.
