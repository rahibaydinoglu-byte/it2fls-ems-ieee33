"""Shared data preparation: feeder profiles, reference net load, MF identification on training days."""
import numpy as np
import sim

P = np.load("profiles.npz")
LOAD_KW, PV_AV = sim.feeder_profiles(P["load"], P["pv"])
NL = LOAD_KW - PV_AV
SPLIT = P["split"]
PREV = P["prev_idx"]
DAYS = P["days"]
SOC_DATA = P["soc"]


def ref_index(i):
    """Previous calendar day if recorded, otherwise the previous recorded day (first day: itself)."""
    if PREV[i] >= 0:
        return PREV[i]
    return max(i - 1, 0)


REF_KW = np.array([NL[ref_index(i)].mean() for i in range(len(DAYS))])


def residual_noise(x, win=15):
    """Std of the deviation of a 1-min signal from its centred 15-min moving average (per day rows)."""
    k = np.ones(win) / win
    out = []
    for row in x:
        ma = np.convolve(row, k, mode="same")
        out.append((row - ma)[win:-win])
    return float(np.std(np.concatenate(out)))


def identify(train_mask):
    dev = ((NL[train_mask] - REF_KW[train_mask, None]) / 1000.0)
    # sigma_err combines short-term variability of the measured signal (from data) and sensor noise (assumed)
    dev_noise = float(np.hypot(residual_noise(NL[train_mask] / 1000.0), sim.CFG["sensor_dev_mw"]))
    soc = SOC_DATA[train_mask]
    soc_noise = float(np.hypot(residual_noise(soc), sim.CFG["sensor_soc"]))
    return sim.identify_mfs(dev.ravel(), dev_noise, soc.ravel(), soc_noise)


MF = identify(SPLIT == 0)


def scenarios(day_mask, n, seed):
    """n scenarios: day drawn uniformly from day_mask, measurement noise, initial SoC ~ U(30,70)%."""
    rng = np.random.default_rng(seed)
    pool = np.flatnonzero(day_mask)
    d = rng.choice(pool, n, replace=True) if n != len(pool) else pool.copy()
    nd = rng.normal(0, sim.CFG["sensor_dev_mw"], (n, 288))
    ns = rng.normal(0, sim.CFG["sensor_soc"], (n, 288))
    soc0 = rng.uniform(30, 70, n)
    return dict(day=d, load=LOAD_KW[d], pv=PV_AV[d], ref=REF_KW[d], soc0=soc0, nd=nd, ns=ns)


def run(ctrl, sc, record=False):
    return sim.simulate(ctrl, sc["load"], sc["pv"], sc["ref"], sc["soc0"], sc["nd"], sc["ns"], record=record)
