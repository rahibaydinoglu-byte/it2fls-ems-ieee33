"""Prepare per-unit 1-min daily profiles from the PV_Battery_EMS_Operational_Dataset.

Output (npz):
  days      : list of dates (YYYY-MM-DD) with complete 1440-min records
  pv        : (n_days, 1440) PV output per unit of inverter rating (16.3 kW observed max)
  load      : (n_days, 1440) site load per unit of observed peak load
  soc       : (n_days, 1440) measured battery SoC (%)
  prev_idx  : index of previous calendar day in `days` (or -1)
  split     : 0 = train, 1 = validation, 2 = test (interleaved by day index: 2:1:1)
"""
import sys
import numpy as np
import pandas as pd

src, out = sys.argv[1], sys.argv[2]
df = pd.read_pickle(src).set_index("TIME").sort_index()
PV_RATED = float(df.P_PV.max())
LOAD_PEAK = float(df.LOAD.max())

days, pv, load, soc = [], [], [], []
for d, g in df.groupby(df.index.normalize()):
    if len(g) != 1440:
        continue
    days.append(d.strftime("%Y-%m-%d"))
    pv.append(g.P_PV.to_numpy() / PV_RATED)
    load.append(g.LOAD.to_numpy() / LOAD_PEAK)
    soc.append(g.SOC_REAL.to_numpy())
pv, load, soc = map(np.array, (pv, load, soc))
dates = pd.to_datetime(days)
pos = {d: i for i, d in enumerate(dates)}
prev_idx = np.array([pos.get(d - pd.Timedelta(days=1), -1) for d in dates])
split = np.array([[0, 1, 0, 2][i % 4] for i in range(len(days))])
np.savez_compressed(out, days=np.array(days), pv=pv, load=load, soc=soc,
                    prev_idx=prev_idx, split=split,
                    pv_rated_kw=PV_RATED, load_peak_kw=LOAD_PEAK)
print(f"{len(days)} complete days; PV rating {PV_RATED:.2f} kW; load peak {LOAD_PEAK:.2f} kW")
print("split counts (train/val/test):", np.bincount(split))
print("days with previous-day record:", int((prev_idx >= 0).sum()))
