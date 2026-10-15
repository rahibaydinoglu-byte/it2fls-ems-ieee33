# Simulation code: IT2FLS / T1FLS / PI energy management on the IEEE 33-bus feeder

Data: PV_Battery_EMS_Operational_Dataset (Mendeley Data, DOI 10.17632/p4bvdjrgzr.1). Unzip it, then:

    python load.py <dataset_folder>/PV_Battery_EMS_Operational_Dataset data.pkl
    python profiles.py data.pkl profiles.npz
    python run_experiments.py   # preliminary protocol (cost-only criterion) -> results.json (kept as results_prelim.json)
    python protocol2.py         # final protocol reported in the paper -> results2.json, results2_test.npz
    python make_figures.py      # Figures 1-3

Requirements: Python 3.11+, numpy, pandas, scipy, matplotlib.
Files: sim.py (feeder model, load flow, controllers, EKM), setup_data.py (profiles, train/validation/test split,
membership-function identification), results2.json (all numbers reported in the paper), results_prelim.json (preliminary run).
