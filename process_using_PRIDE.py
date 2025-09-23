"""Utility to load a WindBorne `*.pos.stat` file into pandas DataFrames.

This keeps the original `$POS`, `$VELACC`, `$CLK`, and `$SAT` records
separate so you can explore them comfortably in e.g. Spyder or Jupyter.

Example (interactive):

>>> import parse_pos_stat_df as psd
>>> pos, velacc, clk, sat = psd.load("path/to/flight_file2.pos.stat")
>>> pos.head()

Run as a script to print simple summaries:

$ python parse_pos_stat_df.py path/to/file.pos.stat
"""

import datetime
import pathlib
import sys


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

####
from PRIDE_helper_functions import plot_kin_xyz, plot_geometric_ranges
from PRIDE_helper_functions import load_csv, load_kin, receiver_clock_file
from PRIDE_helper_functions import geometric_ranges, apply_corrections_ionosphere
from PRIDE_helper_functions import attach_geometric_residuals, plot_carrier_residuals
from PRIDE_helper_functions import plot_carrier_residuals
import gnsspy as gp

from orbit_propegator_brn import parse_sp3_clock_file


import georinex as gr

from orbit_propegator_brn import load_ecef_from_dataframe

import pickle
from pathlib import Path


def save_object(obj, file_path):
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_object(file_path):
    with Path(file_path).open("rb") as f:
        return pickle.load(f)
# -----------------------------------------------------------------------------
# CLI helper
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) != 2:
        #print("Usage: python parse_pos_stat_df.py path/to/file.pos.stat", file=sys.stderr)
        #sys.exit(1)
        #path_root = r'/Users/brnold/Library/CloudStorage/OneDrive-Personal/WindBorne/data/W-3260_SD/1/'
        path_root = r"C:\Users\Benjamin Nold\OneDrive\WindBorne\data\W-3260_SD\1\\"
        path = path_root + "flight_file1.pos.stat"

        path = "/Users/brnold/src/gpsro/PRIDE_output/"
        path =  r"C:\Users\Benjamin Nold\src\pythonGPSRO\PRIDE_output\\"
        kin_path = path + "kin_2025171_balo"
        res_path = path + "res_all_2025171_balo_modeled.csv"
        obs_path = path + "flight1.obs"
    else:
        path = pathlib.Path(sys.argv[1])
        if not path.is_file():
            sys.exit(f"Error: {path} does not exist or is not a file.")

    # ------------------------------------------------------------------

    pos = load_csv(res_path)
    kin = load_kin(kin_path)

    plot_kin_xyz(kin, title="PRIDE Kinematic ECEF")
    rck_df_1 = receiver_clock_file(r"C:\Users\Benjamin Nold\OneDrive\WindBorne\data\PRIDE-PPP\2025\171\rck_2025171_balo", add_datetime=True)
    rck_df_2 = receiver_clock_file(r"C:\Users\Benjamin Nold\OneDrive\WindBorne\data\PRIDE-PPP\2025\171-172\rck_2025171_balo", add_datetime=True)
    rck_df =  pd.concat([rck_df_1, rck_df_2], axis=0, ignore_index=False)
    fig = plt.figure()
    plt.plot(rck_df.gps_datetime, rck_df.RCK_GPS)
    
   # station = gp.read_obsFile(obs_path)
   ## from gnsspy.funcs import checkif
    #checkif._CWD = r"C:\Users\Benjamin Nold\src\pythonGPSRO\PRIDE_output"

    #orbit = gp.sp3_interp(station.epoch, interval=station.interval, poly_degree=16, sp3_product="gfz", clock_product="gfz")

    # orbit = gp.sp3_interp(station.epoch, interval=station.interval, poly_degree=16, sp3_product="gfz", clock_product="gfz")
    # orbit = gp.sp3_interp(station.epoch, interval=station.interval, poly_degree=16, sp3_product="gfz", clock_product="igs")

    
    # obs = gr.load(obs_path, use='G')
    # obs_E = gr.load(obs_path, use='E')
    obs = load_object("obs_pickel")
    obs_E = load_object("obs_pickel_e")
    
    try:
       _this_file = pathlib.Path(__file__).resolve()
       base_dir = _this_file.parent
    except NameError:
       base_dir = pathlib.Path.cwd()
    sp3_dir = base_dir / "sp3_cache"
    sp3_paths = [p for p in sp3_dir.iterdir() if p.is_file() and p.suffix.lower() == ".sp3"] if sp3_dir.is_dir() else []

    
    sat_clk_df = parse_sp3_clock_file(sp3_paths)
    
    
    sv_all = np.concatenate([obs.sv.values, obs_E.sv.values])          # flat array
    sv_all_unique = np.unique(sv_all)                                   # optional
    
    # Python lists (preserves order, removes dups)
    sv_all_ordered = list(dict.fromkeys([*obs.sv.values, *obs_E.sv.values]))

    ecef_df = load_ecef_from_dataframe(kin[['Mjd','Sod']], sv_all_ordered, cache_dir="sp3_cache")
    
    # for each sv in obs.sv, find the matching times in ecef_df.gps_dataframe and pos.
    # calculate the geometric distances between both of those points and save it in a results dataframe
    rng = geometric_ranges(sv_all_ordered, ecef_df, kin)
    plot_geometric_ranges(rng, title="Geometric ranges by satellite")
    
    
    pr_if_df = apply_corrections_ionosphere(obs, rx_clk=rck_df, sat_clk=sat_clk_df)
    pr_if_df = attach_geometric_residuals(pr_if_df, rng, tolerance_s=0.1)
    
    temp = pr_if_df[pr_if_df.sat == 'G04']
   
    row_interest = pr_if_df.sat == 'G01'
    plt.figure()
    plt.plot(pr_if_df.time[row_interest], pr_if_df.carrier_minus_geom_m[row_interest]) 
   
    ax = plot_carrier_residuals(
        pr_if_df,  # after attach_geometric_residuals
        title="Carrier IF (clk-corrected) - Geometric range"
    )
    plot_carrier_residuals(pr_if_df, title="Carrier IF (clk-corrected) - Geometric range")
    
    ## WTF is going on here
    
    from PRIDE_helper_functions import plot_satellite_ecef_errors
    
    errors_df, axes = plot_satellite_ecef_errors(
        pos, ecef_df, svs=['G04'], title="Selected SV ECEF errors"
        )
