"""Protocol 2 (fixed before execution).
Criterion J = cost/cost_nc + sd/sd_nc (means over validation scenarios; nc = no control).
PI: 144-point grid (kp, ki, kd, kr). T1FLS: 144-point grid (s_d, g_b, g_d, g_r).
IT2FLS: gains of tuned T1FLS + k in {0, 0.25, ..., 3}. All selection on validation days.
Test: same 10,000 scenarios (seed 2026) as the preliminary run; each final controller run once."""
import itertools, json, time
import numpy as np
from scipy import stats
import sim
from setup_data import MF, SPLIT, scenarios, run

val = scenarios(SPLIT == 1, 512, seed=11)
nc = run(sim.NoControl(), val)
C0, S0 = float(nc["cost"].mean()), float(np.sqrt(nc["pdv"]).mean())
def J(o): return float(o["cost"].mean() / C0 + np.sqrt(o["pdv"]).mean() / S0)
out = dict(protocol=__doc__, nc_val=dict(cost=C0, sd=S0))

t0 = time.time()
pi_grid = list(itertools.product([0.25, 0.5, 1.0, 2.0], [0.0, 0.25, 0.5, 1.0], [0.5, 1.0, 2.0], [0.5, 1.0, 2.0]))
pi_scores = [(J(run(sim.PIController(*g), val)), g) for g in pi_grid]
pj, pg = min(pi_scores)
out["pi"] = dict(J=pj, kp=pg[0], ki=pg[1], kd=pg[2], kr=pg[3], n_grid=len(pi_grid))
print("PI", out["pi"], f"{time.time()-t0:.0f}s", flush=True)

t0 = time.time()
fz_grid = list(itertools.product([0.5, 0.75, 1.0, 1.25], [0.5, 0.75, 1.0, 1.25], [0.5, 1.0, 1.5], [0.5, 1.0, 1.5]))
t1_scores = [(J(run(sim.FuzzyController(MF, 0, False, *g), val)), g) for g in fz_grid]
tj, tg = min(t1_scores)
out["t1"] = dict(J=tj, s_d=tg[0], g_b=tg[1], g_d=tg[2], g_r=tg[3], n_grid=len(fz_grid))
print("T1", out["t1"], f"{time.time()-t0:.0f}s", flush=True)

ks = np.round(np.arange(0.0, 3.01, 0.25), 2)
sweep = []
for k in ks:
    o = run(sim.FuzzyController(MF, float(k), True, *tg), val)
    sweep.append(dict(k=float(k), J=J(o), cost=float(o["cost"].mean()), sd=float(np.sqrt(o["pdv"]).mean()),
                      peak=float(o["peak"].mean()), starts=float(o["starts"].mean()), rur=float(o["rur"].mean())))
    print("k", sweep[-1], flush=True)
kb = min(sweep, key=lambda r: r["J"])["k"]
out["k_sweep"], out["k_best"] = sweep, kb

N = 10000
test = scenarios(SPLIT == 2, N, seed=2026)
ctrls = {"IT2FLS": sim.FuzzyController(MF, kb, True, *tg), "T1FLS": sim.FuzzyController(MF, 0, False, *tg),
         "PI": sim.PIController(*pg), "No control": sim.NoControl()}
res = {}
for n, c in ctrls.items():
    t0 = time.time(); res[n] = run(c, test); print(n, f"{time.time()-t0:.0f}s", flush=True)
np.savez_compressed("results2_test.npz", day=test["day"], soc0=test["soc0"],
                    **{f"{n}_{m}": v for n, r in res.items() for m, v in r.items()})
metrics = ["cost", "pdv", "peak", "rur", "vd", "starts"]
out["test"] = {}
for n, r in res.items():
    d = {m: dict(mean=float(np.mean(r[m])), std=float(np.std(r[m], ddof=1))) for m in metrics}
    sd = np.sqrt(r["pdv"]); d["sd"] = dict(mean=float(sd.mean()), std=float(sd.std(ddof=1)))
    out["test"][n] = d

def paired(a, b):
    dd = a - b
    if np.allclose(dd, 0):
        return dict(mean_diff=0.0, dz=None, W=None, p=None, n=int(len(dd)))
    w = stats.wilcoxon(a, b)
    return dict(mean_diff=float(dd.mean()), rel=float(dd.mean() / np.mean(b)), dz=float(dd.mean() / dd.std(ddof=1)),
                W=float(w.statistic), p=float(w.pvalue), n=int(len(dd)))
day = test["day"]; ud = np.unique(day)
out["stats"] = {}
for base in ("T1FLS", "PI"):
    for m in ("cost", "sd", "peak", "starts", "rur", "vd"):
        a = np.sqrt(res["IT2FLS"]["pdv"]) if m == "sd" else res["IT2FLS"][m]
        b = np.sqrt(res[base]["pdv"]) if m == "sd" else res[base][m]
        da = np.array([a[day == u].mean() for u in ud]); db = np.array([b[day == u].mean() for u in ud])
        out["stats"][f"IT2FLS_vs_{base}_{m}"] = dict(scenario=paired(a, b), day=paired(da, db))
out["n_test_days"] = int(len(ud))
json.dump(out, open("results2.json", "w"), indent=1)
print("DONE")
