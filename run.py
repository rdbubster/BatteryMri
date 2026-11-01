"""Usage:  python run.py        -> processes the real NASA data and writes outputs/
Then:     python -m streamlit run src/dashboard.py"""
import os, sys
ROOT = os.path.dirname(os.path.abspath(__file__)); os.chdir(ROOT); sys.path.insert(0, ROOT)
from src.pipeline import run_pipeline
if __name__ == "__main__":
    print("=" * 56); print("  BatteryMRI - real NASA cell data"); print("=" * 56)
    run_pipeline()
    print("\nNext:  python -m streamlit run src/dashboard.py")
