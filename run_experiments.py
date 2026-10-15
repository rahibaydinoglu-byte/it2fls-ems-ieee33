"""Full experimental protocol.
1) PI gains: grid search on validation days, criterion = mean daily operational cost.
2) FOU width k: sweep on validation days, criterion = mean daily operational cost.
3) Test: 10,000 Monte Carlo scenarios on held-out test days, run once for all controllers.
4) Statistics, timing, figures data -> results.json / results_test.npz
"""
import itertools, json, time
import numpy as np
from scipy import stats
import sim
from setup_data import MF, SPLIT, DAYS, scenarios, run

out = {"mf": {v: {kk: (np.asarray(vv).tolist() if vv is not None else None) for kk, vv in MF[v].items()} for v in MF}}
out["split_days"] = {"train": int((SPLIT == 0).sum()), "validation": int((SPLIT == 1).sum()), "test": int((SPLIT == 2).sum())}

# ---------------- 1) PI tuning on validation
val = scenarios(SPLIT == 1, 512, seed=11)
grid = list(itertools.product([0.25, 0.5, 1.0, 2.0], [0.0, 0.25, 0.5, 1.0], [0.5, 1.0, 2.0], [0.5, 1.0, 2.0]))
best = None
t0 = time.time()
for g in grid:
    c = float(np.mean(run(sim.PIController(*g), val)["cost"]))
    if best is None or c < best[0]:
        best = (c, g)
out["pi_gains"] = dict(kp=best[1][0], ki=best[1][1], kd=best[1][2], kr=best[1][3], val_cost=best[0], n_grid=len(grid))
print("PI tuned", out["pi_gains"], f"{time.time()-t0:.0f}s", flush=True)

# ---------------- 2) k sweep on validation
ks = np.round(np.arange(0.0, 3.01, 0.25), 2)
sweep = []
for k in ks:
    o = run(sim.FuzzyController(MF, float(k), True), val)
    sweep.append(dict(k=float(k), cost=float(np.mean(o["cost"])), pdv=float(np.mean(o["pdv"])),
                      sd=float(np.mean(np.sqrt(o["pdv"]))), peak=float(np.mean(o["peak"])),
                      starts=float(np.mean(o["starts"])), rur=float(np.mean(o["rur"]))))
    print("k", k, sweep[-1], flush=True)
k_best = min(sweep, key=lambda r: r["cost"])["k"]
out["k_sweep"] = sweep
out["k_best"] = k_best
out["t1_val"] = {kk: float(np.mean(v)) for kk, v in run(sim.FuzzyController(MF, 0, False), val).items()}

# ---------------- 3) test
N = 10000
test = scenarios(SPLIT == 2, N, seed=2026)
ctrls = {"IT2FLS": sim.FuzzyController(MF, k_best, True), "T1FLS": sim.FuzzyController(MF, 0, False),
         "PI": sim.PIController(*best[1])}
res = {}
for name, c in ctrls.items():
    t0 = time.time(); res[name] = run(c, test); print(name, f"{time.time()-t0:.0f}s", flush=True)
np.savez_compressed("results_test.npz", day=test["day"], soc0=test["soc0"],
                    **{f"{n}_{m}": v for n, r in res.items() for m, v in r.items()})

metrics = ["pdv", "peak", "cost", "rur", "vd", "starts"]
out["test"] = {n: {m: dict(mean=float(np.mean(r[m])), std=float(np.std(r[m], ddof=1))) for m in metrics}
               for n, r in res.items()}
for n, r in res.items():
    sd = np.sqrt(r["pdv"]); out["test"][n]["sd"] = dict(mean=float(sd.mean()), std=float(sd.std(ddof=1)))

# ---------------- 4) statistics: scenario level and day level (paired)
def paired(a, b):
    d = a - b
    w = stats.wilcoxon(a, b, alternative="two-sided")
    return dict(mean_diff=float(d.mean()), dz=float(d.mean() / d.std(ddof=1)) if d.std() > 0 else None,
                W=float(w.statistic), p=float(w.pvalue), n=int(len(d)))

day = test["day"]; udays = np.unique(day)
out["stats"] = {}
for base in ("T1FLS", "PI"):
    for m in ("cost", "pdv", "peak", "starts"):
        a, b = res["IT2FLS"][m], res[base][m]
        da = np.array([a[day == u].mean() for u in udays]); db = np.array([b[day == u].mean() for u in udays])
        out["stats"][f"IT2FLS_vs_{base}_{m}"] = dict(scenario=paired(a, b), day=paired(da, db))
out["n_test_days"] = int(len(udays))

# ---------------- 5) timing of one control cycle (single scenario, this machine)
rng = np.random.default_rng(0)
def time_ctrl(c, reps=3000):
    xs = [(np.array([rng.normal()]), np.array([rng.uniform(0, 100)]), np.array([rng.uniform()]), np.array([rng.uniform()])) for _ in range(reps)]
    st = {}
    t0 = time.perf_counter()
    for x in xs:
        c.act(*x, st)
    return (time.perf_counter() - t0) / reps * 1e3

# single-sample IT2 cycle using the EKM routine (as would be deployed)
class IT2Single(sim.FuzzyController):
    def _infer(self, va, xa, vb, xb, table):
        la, ua = self._memb(va, xa); lb, ub = self._memb(vb, xb)
        fl = (la[0][:, None] * lb[0][None, :]).ravel(); fh = (ua[0][:, None] * ub[0][None, :]).ravel()
        yl, yr = sim.ekm_single(fl, fh, table.ravel())
        return np.array([0.5 * (yl + yr)])

out["timing_ms"] = {"IT2FLS (EKM)": time_ctrl(IT2Single(MF, k_best, True)),
                    "T1FLS": time_ctrl(sim.FuzzyController(MF, 0, False)),
                    "PI": time_ctrl(sim.PIController(*best[1]))}
import platform, os
out["platform"] = dict(python=platform.python_version(), numpy=np.__version__, machine=platform.machine(),
                       processor=platform.processor(), cpus=os.cpu_count())
print("timing", out["timing_ms"])
json.dump(out, open("results.json", "w"), indent=1)
print(json.dumps({k: out[k] for k in ("pi_gains", "k_best", "test", "timing_ms")}, indent=1))
