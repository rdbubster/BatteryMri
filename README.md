# BatteryMRI

Reads the electrochemical "fingerprint" of a Li-ion cell from an ordinary **charge** (voltage, current, temperature)
and estimates its health. Built and validated on **real** NASA ageing data; designed to take real ESP32 logs unchanged.
UVCE ECE Major Project 2025-26.

## Run it (VS Code terminal, inside this folder)

    py -m pip install -r requirements.txt
    py run.py                                  # ~30 s, rebuilds outputs/ from the real data
    py -m streamlit run src/dashboard.py       # opens http://localhost:8501

Results are pre-built, so the dashboard also works before `run.py`. Windows shortcut: double-click `run_demo.bat`.

## Files

    data/nasa_cells.npz     real charge+discharge logs of 4 cells (B0005, B0006, B0007, B0018), compact format
    src/nasa.py             loads the real data
    src/ica.py              dQ/dV (incremental capacity) + feature extraction  <- the core idea; works on any charge log
    src/pipeline.py         features, leave-one-cell-out validation, baselines, deviation alert -> outputs/
    src/lab.py              sensor-noise robustness experiment on the real data
    src/dashboard.py        Streamlit dashboard (8 tabs)
    firmware/esp32_logger.ino   ESP32 + INA219 + DS18B20 CSV logger (TEMPLATE - not yet tested on a real cell)
    tools/extract_nasa.py   how data/nasa_cells.npz was made from the original .mat files
    PRESENTATION_GUIDE.md   plain-English explanation, demo script, likely questions

## Data provenance (please do this before citing)

The .mat files were taken from a public GitHub mirror of NASA's dataset. Download the official copy from the NASA
Prognostics Data Repository ("Battery Data Set", B. Saha and K. Goebel, 2007) and confirm it matches before citing.

## Using your own hardware logs

CSV with columns `t,v,i,temp` (seconds, volts, amps, deg C) from one constant-current charge crossing 3.92-4.17 V.
Upload it in the "Diagnose your data" tab. Charges at currents other than ~1.5 A give "CALIBRATION NEEDED":
log a few charges of your own healthy cell and use them as its reference.
