"""Figures from saved results (no new tuning)."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import sim
from setup_data import MF, SPLIT, DAYS, LOAD_KW, PV_AV, REF_KW, run

R = json.load(open("results2.json"))
TG = (R["t1"]["s_d"], R["t1"]["g_b"], R["t1"]["g_d"], R["t1"]["g_r"])
PG = (R["pi"]["kp"], R["pi"]["ki"], R["pi"]["kd"], R["pi"]["kr"])
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
COL = {"IT2FLS": "#1f5fa8", "T1FLS": "#d9822b", "PI": "#3a9a5b"}

# ---------------- Figure 1: architecture (as implemented)
fig, ax = plt.subplots(figsize=(7.4, 3.9)); ax.set_xlim(0, 104); ax.set_ylim(0, 52); ax.axis("off")
def box(x, y, w, h, t, fs=7.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.0", fc="white", ec="black", lw=0.9))
    ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=fs, linespacing=1.25)
def arrow(x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", lw=0.8, color="black"))
for x0, w, t in ((0.5, 31, "Measurement layer"), (36, 31, "Fuzzy inference layer"), (71.5, 32, "Dispatch layer")):
    ax.text(x0 + w / 2, 50, t, ha="center", fontsize=8.5, weight="bold")
    ax.add_patch(plt.Rectangle((x0, 1), w, 47, fill=False, ls="--", lw=0.8))
box(2, 37, 28, 8, "Net-load deviation $d$\n(demand − PV − reference)")
box(2, 26, 28, 8, "Battery state of charge")
box(2, 15, 28, 8, "Time-of-use tariff\n(normalized)")
box(2, 4, 28, 8, "Demand-flexibility index\n(remaining DR energy)")
box(38, 31, 27, 13, "Module A: 25 rules\ninputs: $d$, SoC\noutput: battery power")
box(38, 6, 27, 15, "Module B: 2 × 25 rules\ninputs: residual $r$,\ntariff, flexibility index\noutputs: diesel level, DR")
box(73.5, 35, 28, 8, "Battery 1 MW / 2 MWh\n(bus 6)")
box(73.5, 21, 28, 8, "Diesel generator 300 kW\n(bus 12)")
box(73.5, 7, 28, 8, "Demand response 180 kW\n(bus 25)")
arrow(30, 41, 38, 39.5); arrow(30, 30, 38, 34)
arrow(30, 19, 38, 15); arrow(30, 8, 38, 10)
arrow(51.5, 31, 51.5, 21); ax.text(52.5, 25.5, "$r = d - P_{bat}$", fontsize=7)
arrow(65, 37.5, 73.5, 39); arrow(65, 16, 73.5, 25); arrow(65, 11, 73.5, 11)
fig.savefig("fig1_architecture.png", dpi=600, bbox_inches="tight"); plt.close(fig)

# ---------------- Figure 2: k sensitivity (validation)
sw = R["k_sweep"]; k = [r["k"] for r in sw]
fig, ax1 = plt.subplots(figsize=(6.0, 3.4))
ax1.plot(k, [r["J"] for r in sw], "o-", color=COL["IT2FLS"], label="Selection criterion $J$")
ax1.set_xlabel("FOU width parameter $k$"); ax1.set_ylabel("Criterion $J$ (validation)")
ax2 = ax1.twinx()
ax2.plot(k, [r["cost"] for r in sw], "s--", color="#7a3e9d", label="Operational cost")
ax2.set_ylabel("Mean operational cost (€/day)")
ax1.axvline(R["k_best"], color="grey", ls=":", lw=1)
ax1.text(R["k_best"], ax1.get_ylim()[1], f" selected k = {R['k_best']:g}", va="top", fontsize=8)
h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, loc="upper center", fontsize=8, frameon=False, ncol=2, bbox_to_anchor=(0.5, -0.18))
ax1.grid(alpha=0.25); fig.tight_layout()
fig.savefig("fig2_k_sensitivity.png", dpi=600, bbox_inches="tight"); plt.close(fig)

# ---------------- Figure 3: example test day (median-cost test day of IT2FLS), deterministic single scenario
T = np.load("results2_test.npz")
test_days = np.flatnonzero(SPLIT == 2)
day_cost = {d: T["IT2FLS_cost"][T["day"] == d].mean() for d in np.unique(T["day"])}
dsel = sorted(day_cost, key=day_cost.get)[len(day_cost) // 2]
sc = dict(load=LOAD_KW[[dsel]], pv=PV_AV[[dsel]], ref=REF_KW[[dsel]], soc0=np.array([50.0]),
          nd=np.zeros((1, 288)), ns=np.zeros((1, 288)))
ctrls = {"IT2FLS": sim.FuzzyController(MF, R["k_best"], True, *TG), "T1FLS": sim.FuzzyController(MF, 0, False, *TG),
         "PI": sim.PIController(*PG)}
recs = {n: run(c, sc, record=True) for n, c in ctrls.items()}
hrs = np.arange(1440) / 60
fig, axs = plt.subplots(3, 1, figsize=(7.0, 6.4), sharex=True, gridspec_kw=dict(height_ratios=[2.2, 1.3, 1.3]))
axs[0].plot(hrs, (LOAD_KW[dsel] - PV_AV[dsel]), color="0.6", lw=0.8, label="Net load, no control (before curtailment)")
for n, r in recs.items():
    axs[0].plot(hrs, r["rec"]["p_sub"][0], color=COL[n], lw=0.9, label=n)
axs[0].set_ylabel("Substation import (kW)"); axs[0].legend(fontsize=7.5, ncol=4, frameon=False, loc="upper left")
for n, r in recs.items():
    axs[1].plot(hrs, r["rec"]["p_dg"][0], color=COL[n], lw=0.9, label=f"{n} ({int(r['starts'][0])} starts)")
axs[1].set_ylabel("Diesel output (kW)"); axs[1].set_ylim(-10, 330); axs[1].legend(fontsize=7.5, frameon=False, ncol=3, loc="upper left")
for n, r in recs.items():
    axs[2].plot(hrs, r["rec"]["soc"][0], color=COL[n], lw=0.9)
axs[2].set_ylabel("Battery SoC (%)"); axs[2].set_xlabel("Time (h)"); axs[2].set_xticks(range(0, 25, 3))
for a in axs: a.grid(alpha=0.25)
fig.tight_layout(); fig.savefig("fig3_example_day.png", dpi=600, bbox_inches="tight"); plt.close(fig)
json.dump(dict(example_day=str(DAYS[dsel]), starts={n: int(r["starts"][0]) for n, r in recs.items()},
               cost={n: float(r["cost"][0]) for n, r in recs.items()}), open("fig3_info.json", "w"), indent=1)
print("figures done; example day", DAYS[dsel])
