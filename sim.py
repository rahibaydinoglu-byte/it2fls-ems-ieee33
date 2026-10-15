"""Energy management simulation on the IEEE 33-bus feeder driven by measured PV/load profiles.

All controllers are evaluated in batch over many scenarios (one scenario = one measured day
+ one realization of measurement noise + one initial battery SoC).
Time step: 1 min for the plant; control decisions every 5 min.
Sign convention: battery power > 0 = discharge (injection into the grid).
"""
import numpy as np

# ---------------------------------------------------------------- IEEE 33-bus (Baran & Wu 1989)
LINES = np.array([
    (1, 2, .0922, .0470), (2, 3, .4930, .2511), (3, 4, .3660, .1864), (4, 5, .3811, .1941),
    (5, 6, .8190, .7070), (6, 7, .1872, .6188), (7, 8, .7114, .2351), (8, 9, 1.0300, .7400),
    (9, 10, 1.0440, .7400), (10, 11, .1966, .0650), (11, 12, .3744, .1238), (12, 13, 1.4680, 1.1550),
    (13, 14, .5416, .7129), (14, 15, .5910, .5260), (15, 16, .7463, .5450), (16, 17, 1.2890, 1.7210),
    (17, 18, .7320, .5740), (2, 19, .1640, .1565), (19, 20, 1.5042, 1.3554), (20, 21, .4095, .4784),
    (21, 22, .7089, .9373), (3, 23, .4512, .3083), (23, 24, .8980, .7091), (24, 25, .8960, .7011),
    (6, 26, .2030, .1034), (26, 27, .2842, .1447), (27, 28, 1.0590, .9337), (28, 29, .8042, .7006),
    (29, 30, .5075, .2585), (30, 31, .9744, .9630), (31, 32, .3105, .3619), (32, 33, .3410, .5302)])
LOADS = np.array([  # bus, kW, kvar
    (2, 100, 60), (3, 90, 40), (4, 120, 80), (5, 60, 30), (6, 60, 20), (7, 200, 100), (8, 200, 100),
    (9, 60, 20), (10, 60, 20), (11, 45, 30), (12, 60, 35), (13, 60, 35), (14, 120, 80), (15, 60, 10),
    (16, 60, 20), (17, 60, 20), (18, 90, 40), (19, 90, 40), (20, 90, 40), (21, 90, 40), (22, 90, 40),
    (23, 90, 50), (24, 420, 200), (25, 420, 200), (26, 60, 25), (27, 60, 25), (28, 60, 20),
    (29, 120, 70), (30, 200, 600), (31, 150, 70), (32, 210, 100), (33, 60, 40)])
V_BASE_KV, S_BASE_KVA = 12.66, 1000.0
Z_BASE = V_BASE_KV ** 2 / (S_BASE_KVA / 1000.0)
NB = 33
P_NOM = np.zeros(NB); Q_NOM = np.zeros(NB)
P_NOM[LOADS[:, 0].astype(int) - 1] = LOADS[:, 1]
Q_NOM[LOADS[:, 0].astype(int) - 1] = LOADS[:, 2]

# BIBC: branch k (to-bus b) carries the current of every bus in the subtree of b
_parent = {int(t): int(f) for f, t, _, _ in LINES}
BIBC = np.zeros((NB - 1, NB - 1))
for k, (f, t, _, _) in enumerate(LINES):
    for b in range(2, NB + 1):
        x = b
        while x != 1:
            if x == int(t):
                BIBC[k, b - 2] = 1
                break
            x = _parent[x]
ZBR = (LINES[:, 2] + 1j * LINES[:, 3]) / Z_BASE
BCBV = BIBC.T @ np.diag(ZBR) @ BIBC  # bus-voltage drop per injected bus current


def power_flow(p_kw, q_kvar, iters=4):
    """Batched direct load flow. p_kw, q_kvar: (NB, B) net demand per bus. Returns |V| (NB,B), P_sub kW (B)."""
    s = (p_kw[1:] + 1j * q_kvar[1:]) / S_BASE_KVA
    v = np.ones_like(s)
    for _ in range(iters):
        i_inj = np.conj(s / v)
        v = 1.0 - BCBV @ i_inj
    i_inj = np.conj(s / v)
    i_br = BIBC @ i_inj
    p_sub = np.real(np.conj(i_br[0])) * S_BASE_KVA  # V_slack = 1 p.u.
    vm = np.vstack([np.ones((1, s.shape[1])), np.abs(v)])
    return vm, p_sub


# ---------------------------------------------------------------- system configuration
CFG = dict(
    load_floor=0.3,            # feeder load multiplier = floor + (1-floor) * measured site load (p.u.)
    pv_kw=2000.0, pv_bus=6,
    bat_kw=1000.0, bat_kwh=2000.0, bat_bus=6, soc_min=10.0, soc_max=95.0, eta=0.95,
    dg_kw=300.0, dg_min_frac=0.3, dg_bus=12,
    dr_kw=180.0, dr_kwh_day=360.0, dr_bus=25,
    c_dg=0.28, c_dg_start=3.0, c_dr=0.15, c_bat=0.02,   # EUR/kWh, EUR/start
    load_smooth_min=15,
    dg_min_up=6, dg_min_down=6,                          # control steps (30 min)
    sensor_dev_mw=0.01 * 3715.0 / 1000.0,                # sensor noise: 1% of nominal feeder peak
    sensor_soc=1.0,                                      # SoC estimation noise, percentage points
)
TARIFF_HOURS = np.array([0.10] * 7 + [0.16] * 10 + [0.28] * 4 + [0.16] * 3)  # EUR/kWh by hour
P_MIN, P_MAX = TARIFF_HOURS.min(), TARIFF_HOURS.max()
PRICE_MIN = np.repeat(TARIFF_HOURS, 60)                       # (1440,)
PRICE_N = (PRICE_MIN - P_MIN) / (P_MAX - P_MIN)                # normalized 0..1
DT_H = 1 / 60


def smooth(x, win):
    """Centred moving average along axis 1 with edge padding."""
    k = np.ones(win) / win
    pad = win // 2
    xp = np.pad(x, ((0, 0), (pad, win - 1 - pad)), mode="edge")
    return np.apply_along_axis(lambda r: np.convolve(r, k, mode="valid"), 1, xp)


def feeder_profiles(load_pu, pv_pu, cfg=CFG):
    """load_pu, pv_pu: (D,1440). Returns total feeder demand kW (D,1440), PV available kW (D,1440).
    The measured single-site load is smoothed (15-min moving average) to represent feeder-level aggregation."""
    mult = cfg["load_floor"] + (1 - cfg["load_floor"]) * smooth(load_pu, cfg["load_smooth_min"])
    return mult * P_NOM.sum(), pv_pu * cfg["pv_kw"]


# ---------------------------------------------------------------- fuzzy machinery
LABELS = ["VL", "L", "M", "H", "VH"]


def gauss(x, m, s):
    return np.exp(-0.5 * ((x - m) / s) ** 2)


def rule_table_battery():
    base = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])            # by net-load deviation label
    dis_f = np.array([0.0, 0.25, 0.6, 0.9, 1.0])            # discharge allowed by SoC label
    chg_f = np.array([1.0, 1.0, 0.8, 0.5, 0.0])             # charge allowed by SoC label
    t = np.zeros((5, 5))
    for i in range(5):
        for j in range(5):
            t[i, j] = base[i] * (dis_f[j] if base[i] > 0 else chg_f[j])
    return t                                               # u_bat in [-1,1]


def rule_table_diesel():
    base = np.array([0.0, 0.0, 0.2, 0.7, 1.0])               # by residual-deviation label
    price_f = np.array([0.3, 0.5, 0.8, 1.0, 1.0])            # by tariff label
    return np.outer(base, price_f)                         # u_dg in [0,1]


def rule_table_dr():
    base = np.array([0.0, 0.0, 0.1, 0.6, 1.0])               # by residual-deviation label
    dfi_f = np.array([0.0, 0.4, 0.7, 0.9, 1.0])              # by flexibility label
    return np.outer(base, dfi_f)                           # u_dr in [0,1]


RULES = dict(bat=rule_table_battery(), dg=rule_table_diesel(), dr=rule_table_dr())


def km_batch(f_lo, f_hi, c):
    """Exact centre-of-sets type reduction (all switch points). f_lo,f_hi: (B,M); c: (M,) crisp consequents."""
    o = np.argsort(c); c = c[o]; f_lo = f_lo[:, o]; f_hi = f_hi[:, o]
    # y_l: upper firing on the first L rules (small c), lower on the rest
    cu_hc, cu_h = np.cumsum(f_hi * c, 1), np.cumsum(f_hi, 1)
    cu_lc, cu_l = np.cumsum(f_lo * c, 1), np.cumsum(f_lo, 1)
    tot_lc, tot_l, tot_hc, tot_h = cu_lc[:, -1:], cu_l[:, -1:], cu_hc[:, -1:], cu_h[:, -1:]
    z = np.zeros((f_lo.shape[0], 1))
    A_hc, A_h = np.hstack([z, cu_hc]), np.hstack([z, cu_h])
    A_lc, A_l = np.hstack([z, cu_lc]), np.hstack([z, cu_l])
    num_l = A_hc + (tot_lc - A_lc); den_l = A_h + (tot_l - A_l)
    num_r = A_lc + (tot_hc - A_hc); den_r = A_l + (tot_h - A_h)
    eps = 1e-12
    yl = np.min(num_l / np.maximum(den_l, eps), 1)
    yr = np.max(num_r / np.maximum(den_r, eps), 1)
    return yl, yr


def ekm_single(f_lo, f_hi, c):
    """Enhanced Karnik-Mendel algorithm (Wu & Mendel 2009) for one input vector; returns (yl, yr)."""
    o = np.argsort(c); c = c[o]; fl = f_lo[o]; fh = f_hi[o]; M = len(c)

    def side(left):
        k = int(round(M / 2.4)) if left else int(round(M / 1.7))
        k = min(max(k, 1), M - 1)
        if left:
            a = fh[:k] @ c[:k] + fl[k:] @ c[k:]; b = fh[:k].sum() + fl[k:].sum()
        else:
            a = fl[:k] @ c[:k] + fh[k:] @ c[k:]; b = fl[:k].sum() + fh[k:].sum()
        y = a / b
        for _ in range(M + 2):
            kp = int(np.searchsorted(c, y, side="right"))
            kp = min(max(kp, 1), M - 1)
            if kp == k:
                break
            s = 1 if kp > k else -1
            lo_i, hi_i = (k, kp) if kp > k else (kp, k)
            df = fh[lo_i:hi_i] - fl[lo_i:hi_i]
            if left:
                a += s * (c[lo_i:hi_i] @ df); b += s * df.sum()
            else:
                a -= s * (c[lo_i:hi_i] @ df); b -= s * df.sum()
            y = a / b; k = kp
        return y
    return side(True), side(False)


class FuzzyController:
    """Two-module hierarchical fuzzy EMS. Module A: battery (deviation x SoC).
    Module B: diesel (residual x tariff) and demand response (residual x flexibility).
    type2=False -> type-1 Gaussian sets (sigma_D); type2=True -> IT2 sets with sigma_D +/- k*sigma_err."""

    def __init__(self, mf, k=0.0, type2=True, s_d=1.0, g_b=1.0, g_d=1.0, g_r=1.0):
        self.mf, self.k, self.type2 = mf, k, type2
        self.s_d, self.g_b, self.g_d, self.g_r = s_d, g_b, g_d, g_r

    def _memb(self, var, x):
        m, s, e = self.mf[var]["m"], self.mf[var]["s"], self.mf[var]["err"]
        x = np.clip(x, m[0], m[-1])[:, None]
        if not self.type2:
            g = gauss(x, m, s); return g, g
        su = s + self.k * e
        sl = np.maximum(s - self.k * e, 0.05 * s)
        return gauss(x, m, sl), gauss(x, m, su)

    def _infer(self, va, xa, vb, xb, table):
        la, ua = self._memb(va, xa); lb, ub = self._memb(vb, xb)
        f_lo = (la[:, :, None] * lb[:, None, :]).reshape(len(xa), -1)
        f_hi = (ua[:, :, None] * ub[:, None, :]).reshape(len(xa), -1)
        c = table.reshape(-1)
        if not self.type2:
            return (f_hi @ c) / np.maximum(f_hi.sum(1), 1e-12)
        yl, yr = km_batch(f_lo, f_hi, c)
        return 0.5 * (yl + yr)

    def act(self, dev, soc, price_n, dfi, state):
        u_b = np.clip(self.g_b * self._infer("dev", self.s_d * dev, "soc", soc, RULES["bat"]), -1, 1)
        res = dev - u_b * CFG["bat_kw"] / 1000.0
        u_dg = np.clip(self.g_d * self._infer("dev", self.s_d * res, "price", price_n, RULES["dg"]), 0, 1)
        u_dr = np.clip(self.g_r * self._infer("dev", self.s_d * res, "dfi", dfi, RULES["dr"]), 0, 1)
        return u_b, u_dg, u_dr


class NoControl:
    """Battery, diesel and demand response idle (reference for the selection criterion)."""

    def act(self, dev, soc, price_n, dfi, state):
        z = np.zeros_like(dev)
        return z, z, z


class PIController:
    """PI battery regulator on the net-load deviation; proportional diesel/DR on the residual."""

    def __init__(self, kp, ki, kd, kr):
        self.kp, self.ki, self.kd, self.kr = kp, ki, kd, kr

    def act(self, dev, soc, price_n, dfi, state):
        integ = state.setdefault("integ", np.zeros_like(dev))
        u_b = np.clip(self.kp * dev + self.ki * integ, -1, 1)
        # anti-windup: integrate only when not saturated
        unsat = np.abs(self.kp * dev + self.ki * integ) < 1
        integ += np.where(unsat, dev * 5 / 60, 0.0)
        res = dev - u_b * CFG["bat_kw"] / 1000.0
        u_dg = np.clip(self.kd * res, 0, 1)
        u_dr = np.clip(self.kr * res, 0, 1) * (dfi > 0)
        return u_b, u_dg, u_dr


# ---------------------------------------------------------------- plant simulation
def simulate(ctrl, load_kw, pv_av, ref_kw, soc0, noise_dev, noise_soc, cfg=CFG, record=False):
    """Batch simulation over B scenarios for 1440 minutes.
    load_kw, pv_av: (B,1440) feeder demand and available PV; ref_kw: (B,) reference net load;
    soc0: (B,) initial SoC %; noise_dev, noise_soc: (B,288) measurement noise at each control step."""
    B = load_kw.shape[0]
    soc = soc0.astype(float).copy()
    dr_left = np.full(B, cfg["dr_kwh_day"])
    dg_on = np.zeros(B, bool)
    dg_cnt = np.full(B, 10**6)
    starts = np.zeros(B)
    cost = np.zeros(B); pv_used = np.zeros(B); bat_thr = np.zeros(B)
    p_sub_5 = np.zeros((B, 288)); vd = np.zeros(B)
    state = {}
    ib, idg, idr = cfg["bat_bus"] - 1, cfg["dg_bus"] - 1, cfg["dr_bus"] - 1
    ipv = cfg["pv_bus"] - 1
    share = P_NOM / P_NOM.sum(); qshare = Q_NOM / P_NOM.sum()
    rec = {k: [] for k in ("p_sub", "p_bat", "p_dg", "p_dr", "soc", "curt")} if record else None
    p_bat = p_dg = p_dr = np.zeros(B)
    for t in range(1440):
        if t % 5 == 0:
            s = t // 5
            nl = load_kw[:, t] - pv_av[:, t]
            dev = (nl - ref_kw) / 1000.0 + noise_dev[:, s]
            soc_m = np.clip(soc + noise_soc[:, s], 0, 100)
            dfi = dr_left / cfg["dr_kwh_day"]
            u_b, u_dg, u_dr = ctrl.act(dev, soc_m, np.full(B, PRICE_N[t]), dfi, state)
            # battery limits from (true) SoC
            p_bat = u_b * cfg["bat_kw"]
            e_dis = (soc - cfg["soc_min"]) / 100 * cfg["bat_kwh"] * cfg["eta"]
            e_chg = (cfg["soc_max"] - soc) / 100 * cfg["bat_kwh"] / cfg["eta"]
            p_bat = np.clip(p_bat, -e_chg / (5 / 60), e_dis / (5 / 60))
            # diesel: on only above minimum loading
            p_dg = u_dg * cfg["dg_kw"]
            want = p_dg >= cfg["dg_min_frac"] * cfg["dg_kw"]
            on = np.where(dg_on & (dg_cnt < cfg["dg_min_up"]), True,
                          np.where(~dg_on & (dg_cnt < cfg["dg_min_down"]), False, want))
            p_dg = np.where(on, np.maximum(p_dg, cfg["dg_min_frac"] * cfg["dg_kw"]), 0.0)
            starts += (on & ~dg_on)
            dg_cnt = np.where(on == dg_on, dg_cnt + 1, 1)
            dg_on = on
            p_dr = np.minimum(u_dr * cfg["dr_kw"], dr_left / (5 / 60))
        # per-minute plant
        dem = load_kw[:, t]
        pv = pv_av[:, t]
        inj = pv + p_bat + p_dg + p_dr
        curt = np.maximum(0.0, inj - dem)        # no reverse flow at the substation (losses neglected here)
        curt = np.minimum(curt, pv)
        pv_net = pv - curt
        P = np.outer(share, dem); Q = np.outer(qshare, dem)
        P[ipv] -= pv_net; P[ib] -= p_bat; P[idg] -= p_dg; P[idr] -= p_dr
        vm, p_sub = power_flow(P, Q)
        # state updates
        soc -= np.where(p_bat > 0, p_bat / cfg["eta"], p_bat * cfg["eta"]) * DT_H / cfg["bat_kwh"] * 100
        dr_left = np.maximum(dr_left - p_dr * DT_H, 0)
        cost += (PRICE_MIN[t] * np.maximum(p_sub, 0) + cfg["c_dg"] * p_dg + cfg["c_dr"] * p_dr
                 + cfg["c_bat"] * np.abs(p_bat)) * DT_H
        pv_used += pv_net * DT_H
        bat_thr += np.abs(p_bat) * DT_H
        p_sub_5[:, t // 5] += p_sub / 5
        vd += np.max(np.abs(1 - vm), 0) / 1440
        if record:
            for k, v in (("p_sub", p_sub), ("p_bat", p_bat), ("p_dg", p_dg), ("p_dr", p_dr), ("soc", soc.copy()), ("curt", curt)):
                rec[k].append(v)
    cost += starts * cfg["c_dg_start"]
    pv_tot = pv_av.sum(1) * DT_H
    out = dict(pdv=p_sub_5.var(1), peak=p_sub_5.max(1), cost=cost,
               rur=np.where(pv_tot > 0, pv_used / np.maximum(pv_tot, 1e-9), 1.0),
               vd=vd, starts=starts, soc_end=soc)
    if record:
        out["rec"] = {k: np.array(v).T for k, v in rec.items()}
    return out


# ---------------------------------------------------------------- membership-function identification from data
def identify_mfs(dev_samples, dev_noise, soc_samples, soc_noise):
    """Equal-width partition into five labels; m_D, sigma_D are the sample mean/std of the data in each label."""
    def from_data(x, lo, hi):
        edges = np.linspace(lo, hi, 6)
        xc = np.clip(x, lo, hi)
        idx = np.clip(np.digitize(xc, edges[1:-1]), 0, 4)
        m = np.array([xc[idx == i].mean() for i in range(5)])
        s = np.array([xc[idx == i].std() for i in range(5)])
        n = np.array([(idx == i).sum() for i in range(5)])
        return m, s, n
    lo, hi = np.quantile(dev_samples, [0.005, 0.995])
    m_d, s_d, n_d = from_data(dev_samples, lo, hi)
    m_s, s_s, n_s = from_data(soc_samples, 0.0, 100.0)
    grid = np.linspace(0, 1, 5); sp = 0.25
    return {
        "dev": dict(m=m_d, s=s_d, err=dev_noise, n=n_d, range=(lo, hi)),
        "soc": dict(m=m_s, s=s_s, err=soc_noise, n=n_s, range=(0, 100)),
        "price": dict(m=grid, s=np.full(5, sp / 2), err=0.0, n=None, range=(0, 1)),
        "dfi": dict(m=grid, s=np.full(5, sp / 2), err=0.0, n=None, range=(0, 1)),
    }
