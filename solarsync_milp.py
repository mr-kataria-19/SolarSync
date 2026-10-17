"""
SolarSync: Intelligent Solar Self-Consumption Optimization for Smart Buildings
==============================================================================
MILP (PuLP + CBC) co-scheduling of battery and HVAC for an office building.

Install:   pip install "pulp==2.9.0" pandas numpy matplotlib
Run:       python solarsync_milp.py          (CSV must be next to the script)
Colab:     upload solarsync_synthetic_data.csv, then paste cell by cell
           (every "# %%" marks a cell).

MATHEMATICAL MODEL (t = 0..95, dt = 0.25 h)
-------------------------------------------
Variables (kW unless stated), all >= 0:
  P_ch, P_dis        battery charge / discharge power
  E[t], t=0..96      battery stored energy (kWh)
  P_imp, P_exp       grid import / export
  P_hvac             scheduled HVAC power
  P_pvl, P_pvc, P_pve, P_pvx   PV to load / battery / grid / curtailed
  P_peak             peak grid import over the day (scalar)
  u[t] in {0,1}      1 = battery may charge, 0 = may discharge
  g[t] in {0,1}      1 = grid import allowed, 0 = export allowed

Minimise
  J = sum_t dt*(c_imp[t]*P_imp[t] - c_exp[t]*P_pve[t])     [Rs]   net energy cost
    + lam_peak * P_peak                                    [Rs/kW/day * kW]
    + lam_curt * sum_t dt*P_pvx[t]                         [Rs/kWh * kWh]
    + lam_exp  * sum_t dt*P_pve[t]                         [Rs/kWh * kWh]

Subject to, for every t
  (1) PV allocation   P_pv[t] = P_pvl + P_pvc + P_pve + P_pvx
  (2) Building balance P_pvl + P_imp + P_dis = P_fixed + P_hvac
  (3) No grid charging P_ch = P_pvc
  (4) Battery energy   E[t+1] = E[t] + eta_c*P_ch*dt - P_dis*dt/eta_d
  (5) Bounds           8 <= E[t] <= 36, E[0] = E[96] = 20
  (6) Mode (big-M)     P_ch <= 10*u,  P_dis <= 10*(1-u)
  (7) Grid             P_imp <= 30*g, P_exp <= 30*(1-g), P_exp = P_pve
  (8) HVAC             hvac_min*avail <= P_hvac <= hvac_max*avail
  (9) HVAC energy      (1-tol)*E_base <= sum P_hvac*dt <= (1+tol)*E_base
 (10) Peak             P_peak >= P_imp[t]
"""

# %% 1. Imports and data loading ------------------------------------------------
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pulp

CSV_PATH = "solarsync_synthetic_data.csv"
OUT_DIR = "solarsync_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

if not os.path.exists(CSV_PATH):
    # Colab: upload the file via the Files panel, or run:
    #   from google.colab import files; files.upload()
    # Alternative path used in this project's sandbox:
    alt = "/mnt/user-data/uploads/solarsync_synthetic_data.csv"
    if os.path.exists(alt):
        CSV_PATH = alt
    else:
        raise FileNotFoundError(
            "Upload solarsync_synthetic_data.csv to the working directory "
            "(Colab: Files panel -> Upload, or google.colab.files.upload()).")

df = pd.read_csv(CSV_PATH, parse_dates=["timestamp"])

# %% 2. Data validation ------------------------------------------------------
REQUIRED = ["timestamp", "pv_available_kw", "fixed_load_kw", "hvac_baseline_kw",
            "hvac_min_kw", "hvac_max_kw", "hvac_available", "import_tariff",
            "export_tariff", "outdoor_temp_c", "occupancy", "baseline_total_load_kw"]


def validate(d):
    """Raise on hard errors; print a short report. Units: kW, Rs/kWh, deg C."""
    missing_cols = [c for c in REQUIRED if c not in d.columns]
    assert not missing_cols, f"Missing columns: {missing_cols}"
    assert len(d) == 96, f"Expected 96 rows, got {len(d)}"
    assert d.isna().sum().sum() == 0, "Missing values found"
    steps = d["timestamp"].diff().dropna().unique()
    assert len(steps) == 1 and steps[0] == pd.Timedelta(minutes=15), "Not a regular 15-min grid"
    assert d["timestamp"].dt.date.nunique() == 1, "Data must cover exactly one day"
    for c in ["pv_available_kw", "fixed_load_kw", "hvac_baseline_kw", "hvac_min_kw",
              "hvac_max_kw", "import_tariff", "export_tariff"]:
        assert (d[c] >= 0).all(), f"Negative values in {c}"
    assert set(d["hvac_available"].unique()) <= {0, 1}, "hvac_available must be 0/1"
    assert (d["hvac_min_kw"] <= d["hvac_max_kw"]).all(), "hvac_min > hvac_max"
    on = d["hvac_available"] == 1
    assert (d.loc[on, "hvac_baseline_kw"].between(d.loc[on, "hvac_min_kw"],
                                                  d.loc[on, "hvac_max_kw"])).all(), \
        "Baseline HVAC outside [min,max] while available"
    assert (d.loc[~on, "hvac_baseline_kw"] == 0).all(), "Baseline HVAC > 0 while unavailable"
    resid = (d["baseline_total_load_kw"] - d["fixed_load_kw"] - d["hvac_baseline_kw"]).abs().max()
    assert resid < 1e-6, f"baseline_total_load != fixed + hvac (max residual {resid})"
    assert d["pv_available_kw"].max() <= 20.0 + 1e-9, "PV exceeds 20 kWp"
    print("Data validation passed: 96 rows, 15-min grid, no NaNs, "
          f"PV peak {d['pv_available_kw'].max():.2f} kW (<= 20 kWp), "
          f"tariffs {sorted(map(float, d['import_tariff'].unique()))} Rs/kWh import, "
          f"{sorted(map(float, d['export_tariff'].unique()))} Rs/kWh export")


validate(df)

# %% 3. System parameters ----------------------------------------------------
P = dict(
    dt=0.25, T=96,
    batt_cap=40.0, p_ch_max=10.0, p_dis_max=10.0,
    e_init=20.0, e_min=8.0, e_max=36.0,
    eta_c=0.95, eta_d=0.95,
    p_imp_max=30.0, p_exp_max=30.0,
    lam_peak=100.0,   # Rs per kW of peak import per simulated day
    lam_curt=2.0,     # Rs per kWh of curtailed PV
    lam_exp=0.50,     # Rs per kWh of exported PV (intentional preference for local use)
    hvac_tol=0.05,    # +-5 % of baseline HVAC energy
)
dt, T = P["dt"], P["T"]
idx = range(T)

# %% 4. Baseline (no battery, original HVAC) ----------------------------------
# Rule: PV serves the load first; any surplus is exported; any deficit is imported.
# The battery stays idle at 20 kWh (start = end, so the comparison is fair).


def evaluate(r):
    """Compute all KPIs from a results DataFrame (same definitions for every strategy)."""
    k = {}
    k["cost_net_rs"] = float((dt * (r.import_tariff * r.p_imp - r.export_tariff * r.p_exp)).sum())
    k["peak_import_kw"] = float(r.p_imp.max())
    pv_e = r.pv_available_kw.sum() * dt
    k["pv_available_kwh"] = float(pv_e)
    k["self_consumption_pct"] = float(100 * ((r.p_pvl + r.p_pvc).sum() * dt) / pv_e)
    load_e = (r.fixed_load_kw + r.p_hvac).sum() * dt
    k["load_kwh"] = float(load_e)
    # Self-sufficiency: solar-supplied load = PV direct + battery discharge.
    # Valid attribution because grid charging is disabled and E_end = E_start, so
    # over the day all delivered battery energy originates from PV (see notes).
    k["self_sufficiency_pct"] = float(100 * ((r.p_pvl + r.p_dis).sum() * dt) / load_e)
    k["curtail_kwh"] = float(r.p_pvx.sum() * dt)
    k["import_kwh"] = float(r.p_imp.sum() * dt)
    k["export_kwh"] = float(r.p_exp.sum() * dt)
    k["hvac_kwh"] = float(r.p_hvac.sum() * dt)
    k["batt_charge_kwh"] = float(r.p_ch.sum() * dt)       # AC-side energy into battery
    k["batt_discharge_kwh"] = float(r.p_dis.sum() * dt)   # energy delivered to load
    k["soc_end_kwh"] = float(r.e_next_kwh.iloc[-1])
    return k


def check_physics(r, label, tol=1e-5):
    """Independent sanity checks on any schedule; raises if physics is violated."""
    bal = (r.p_pvl + r.p_imp + r.p_dis - r.fixed_load_kw - r.p_hvac).abs().max()
    alloc = (r.pv_available_kw - r.p_pvl - r.p_pvc - r.p_pve - r.p_pvx).abs().max()
    e = np.append(r.e_kwh.values, r.e_next_kwh.values[-1])   # E[0..96]
    dyn = np.abs(r.e_next_kwh.values - r.e_kwh.values - P["eta_c"] * r.p_ch.values * dt
                 + r.p_dis.values * dt / P["eta_d"]).max()
    both_b = ((r.p_ch > 1e-6) & (r.p_dis > 1e-6)).sum()
    both_g = ((r.p_imp > 1e-6) & (r.p_exp > 1e-6)).sum()
    ok = (bal < tol and alloc < tol and dyn < tol and both_b == 0 and both_g == 0
          and e.min() >= P["e_min"] - tol and e.max() <= P["e_max"] + tol
          and r.p_imp.max() <= P["p_imp_max"] + tol)
    print(f"  [{label}] checks: balance {bal:.1e}, PV alloc {alloc:.1e}, "
          f"battery dynamics {dyn:.1e}, simultaneous ch/dis {both_b}, "
          f"simultaneous imp/exp {both_g} -> {'OK' if ok else 'FAILED'}")
    assert ok, f"Physics check failed for {label}"


def build_frame(d, hvac, ch, dis, e, imp, exp, pvl, pvc, pve, pvx):
    r = d[["timestamp", "pv_available_kw", "fixed_load_kw", "hvac_baseline_kw",
           "import_tariff", "export_tariff", "outdoor_temp_c"]].copy()
    r["p_hvac"], r["p_ch"], r["p_dis"] = hvac, ch, dis
    r["p_imp"], r["p_exp"] = imp, exp
    r["p_pvl"], r["p_pvc"], r["p_pve"], r["p_pvx"] = pvl, pvc, pve, pvx
    e = np.asarray(e, dtype=float)
    r["e_kwh"] = e[:T]        # stored energy at the START of each interval (kWh)
    r["e_next_kwh"] = e[1:]   # stored energy at the END of each interval (kWh)
    return r


def simulate_dispatch(d, hvac, battery_rule=None, soc_ceiling=None):
    """Forward-simulate a rule-based dispatch for a given HVAC profile.
    battery_rule=None     -> battery idle at 20 kWh.
    battery_rule='greedy' -> simple documented self-consumption rule:
      * charge from surplus PV (after serving the load), up to a SOC ceiling
        (default 36 kWh) and 10 kW;
      * discharge to cover any load that PV cannot, but never below 20 kWh
        (the start level = terminal target), up to 10 kW.
    PV surplus that cannot be stored is exported (no curtailment needed, as
    export is allowed). The floor equals the start energy, so E_end >= 20 kWh. A
    ceiling that is too high leaves unused energy at midnight (E_end > 20), which would
    be unfair to the other strategies; tune_ceiling() below finds the highest ceiling
    that still drains back to 20 kWh.
    """
    pv = d.pv_available_kw.values
    fixed = d.fixed_load_kw.values
    hv = np.asarray(hvac, dtype=float)
    tar = d.import_tariff.values
    ceiling = P["e_max"] if soc_ceiling is None else soc_ceiling
    E = P["e_init"]
    out = {k: np.zeros(T) for k in ["ch", "dis", "imp", "exp", "pvl", "pvc", "pve", "pvx"]}
    e_series = [E]
    for t in idx:
        load = fixed[t] + hv[t]
        pvl = min(pv[t], load)
        surplus, deficit = pv[t] - pvl, load - pvl
        ch = dis = 0.0
        if battery_rule == "greedy":
            if surplus > 1e-9:
                ch = max(0.0, min(surplus, P["p_ch_max"], (ceiling - E) / (P["eta_c"] * dt)))
            elif deficit > 1e-9 and E > P["e_init"] + 1e-9:
                dis = min(deficit, P["p_dis_max"], (E - P["e_init"]) * P["eta_d"] / dt)
        E = E + P["eta_c"] * ch * dt - dis * dt / P["eta_d"]
        out["pvl"][t], out["pvc"][t], out["ch"][t], out["dis"][t] = pvl, ch, ch, dis
        out["pve"][t] = surplus - ch
        out["imp"][t] = deficit - dis
        out["exp"][t] = out["pve"][t]
        e_series.append(E)
    return build_frame(d, hv, out["ch"], out["dis"], np.array(e_series), out["imp"],
                       out["exp"], out["pvl"], out["pvc"], out["pve"], out["pvx"])


def tune_ceiling(d, hvac, tol=1e-4):
    """Bisection: highest SOC ceiling (kWh) for which the greedy rule ends the day at 20 kWh."""
    lo, hi = P["e_init"], P["e_max"]
    r = simulate_dispatch(d, hvac, "greedy", hi)
    if r.e_next_kwh.iloc[-1] <= P["e_init"] + tol:
        return hi
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        r = simulate_dispatch(d, hvac, "greedy", mid)
        if r.e_next_kwh.iloc[-1] <= P["e_init"] + tol:
            lo = mid
        else:
            hi = mid
    return lo


baseline = simulate_dispatch(df, df.hvac_baseline_kw.values, battery_rule=None)
kpi_baseline = evaluate(baseline)
print("\nBASELINE (original HVAC, no battery operation)")
check_physics(baseline, "Baseline")
print(f"  net cost {kpi_baseline['cost_net_rs']:.2f} Rs/day, "
      f"peak import {kpi_baseline['peak_import_kw']:.2f} kW")

# %% 5-9. MILP: variables, constraints, objective, solve, status handling ---------


def solve_milp(d, hvac_flexible=True, battery_on=True, hvac_tol=None,
               lam_exp=None, lam_peak=None, lam_curt=None, time_limit=120, verbose=False):
    """Build and solve the MILP. Flags let one model represent several strategies:
       hvac_flexible=False -> P_hvac fixed to the baseline schedule
       battery_on=False    -> battery idle at 20 kWh (P_ch = P_dis = 0)
    Returns (results DataFrame, objective value, solver info dict)."""
    hvac_tol = P["hvac_tol"] if hvac_tol is None else hvac_tol
    lam_exp = P["lam_exp"] if lam_exp is None else lam_exp
    lam_peak = P["lam_peak"] if lam_peak is None else lam_peak
    lam_curt = P["lam_curt"] if lam_curt is None else lam_curt

    pv, fixed = d.pv_available_kw.values, d.fixed_load_kw.values
    c_imp, c_exp = d.import_tariff.values, d.export_tariff.values
    h_base = d.hvac_baseline_kw.values
    h_lo = d.hvac_min_kw.values * d.hvac_available.values
    h_hi = d.hvac_max_kw.values * d.hvac_available.values
    e_hvac_base = h_base.sum() * dt

    m = pulp.LpProblem("SolarSync", pulp.LpMinimize)
    # ---- decision variables (explicit bounds encode the physical limits) ----
    ch = pulp.LpVariable.dicts("ch", idx, 0, P["p_ch_max"])
    dis = pulp.LpVariable.dicts("dis", idx, 0, P["p_dis_max"])
    E = pulp.LpVariable.dicts("E", range(T + 1), P["e_min"], P["e_max"])
    imp = pulp.LpVariable.dicts("imp", idx, 0, P["p_imp_max"])
    exp = pulp.LpVariable.dicts("exp", idx, 0, P["p_exp_max"])
    hv = pulp.LpVariable.dicts("hv", idx, 0)
    pvl = pulp.LpVariable.dicts("pvl", idx, 0)
    pvc = pulp.LpVariable.dicts("pvc", idx, 0)
    pve = pulp.LpVariable.dicts("pve", idx, 0)
    pvx = pulp.LpVariable.dicts("pvx", idx, 0)
    u = pulp.LpVariable.dicts("u", idx, cat="Binary")   # 1 = charge mode
    g = pulp.LpVariable.dicts("g", idx, cat="Binary")   # 1 = import mode
    peak = pulp.LpVariable("peak", 0, P["p_imp_max"])

    # ---- objective ----
    m += (pulp.lpSum(dt * (c_imp[t] * imp[t] - c_exp[t] * pve[t]) for t in idx)  # Rs
          + lam_peak * peak                                                       # Rs
          + lam_curt * pulp.lpSum(dt * pvx[t] for t in idx)                       # Rs
          + lam_exp * pulp.lpSum(dt * pve[t] for t in idx)), "total_cost"

    # ---- constraints ----
    m += E[0] == P["e_init"], "E_start"
    m += E[T] == P["e_init"], "E_end"
    for t in idx:
        m += pv[t] == pvl[t] + pvc[t] + pve[t] + pvx[t], f"pv_alloc_{t}"
        m += pvl[t] + imp[t] + dis[t] == fixed[t] + hv[t], f"balance_{t}"
        m += ch[t] == pvc[t], f"no_grid_charge_{t}"
        m += E[t + 1] == E[t] + P["eta_c"] * ch[t] * dt - dis[t] * dt / P["eta_d"], f"soc_{t}"
        m += ch[t] <= P["p_ch_max"] * u[t], f"mode_ch_{t}"
        m += dis[t] <= P["p_dis_max"] * (1 - u[t]), f"mode_dis_{t}"
        m += imp[t] <= P["p_imp_max"] * g[t], f"mode_imp_{t}"
        m += exp[t] <= P["p_exp_max"] * (1 - g[t]), f"mode_exp_{t}"
        m += exp[t] == pve[t], f"export_def_{t}"
        m += peak >= imp[t], f"peak_{t}"
        if hvac_flexible:
            m += hv[t] >= h_lo[t], f"hvac_lo_{t}"
            m += hv[t] <= h_hi[t], f"hvac_hi_{t}"
        else:
            m += hv[t] == h_base[t], f"hvac_fixed_{t}"
        if not battery_on:
            m += ch[t] == 0, f"batt_off_ch_{t}"
            m += dis[t] == 0, f"batt_off_dis_{t}"
    e_hvac = pulp.lpSum(dt * hv[t] for t in idx)
    m += e_hvac >= (1 - hvac_tol) * e_hvac_base, "hvac_energy_lo"
    m += e_hvac <= (1 + hvac_tol) * e_hvac_base, "hvac_energy_hi"

    solver = pulp.PULP_CBC_CMD(msg=verbose, timeLimit=time_limit, gapRel=0.0)
    m.solve(solver)
    status = pulp.LpStatus[m.status]
    info = {"status": status, "objective": pulp.value(m.objective)}

    # ---- status handling ----
    if status == "Infeasible":
        raise RuntimeError(
            "MILP infeasible. Typical causes: HVAC energy band cannot be met inside "
            "[hvac_min, hvac_max]; E_end != E_start unreachable; or PV/load limits "
            "violated. Relax one constraint at a time to find the culprit.")
    if status != "Optimal":
        raise RuntimeError(f"Solver did not reach optimality (status: {status}).")

    val = lambda var, rng: np.array([var[i].value() for i in rng], dtype=float)
    r = build_frame(d, val(hv, idx), val(ch, idx), val(dis, idx), val(E, range(T + 1)),
                    val(imp, idx), val(exp, idx), val(pvl, idx), val(pvc, idx),
                    val(pve, idx), val(pvx, idx))
    r["p_ch_mode"] = val(u, idx)
    r["p_imp_mode"] = val(g, idx)
    # clean tiny solver noise (|x| < 1e-7) so checks and plots are tidy
    num = r.select_dtypes("number").columns
    r[num] = r[num].mask(r[num].abs() < 1e-7, 0.0)
    return r, info["objective"], info


def objective_of(r, lam_exp=None):
    """Recompute the MILP objective from a schedule (to verify the solver and compare strategies)."""
    lam_exp = P["lam_exp"] if lam_exp is None else lam_exp
    return float((dt * (r.import_tariff * r.p_imp - r.export_tariff * r.p_pve)).sum()
                 + P["lam_peak"] * r.p_imp.max()
                 + P["lam_curt"] * r.p_pvx.sum() * dt
                 + lam_exp * r.p_pve.sum() * dt)


# %% 8. Compare the four strategies -------------------------------------------------
print("\nSolving strategies ...")
# 2) Battery-only: HVAC fixed to baseline, battery optimised
s2, obj2, _ = solve_milp(df, hvac_flexible=False, battery_on=True)
# 3) Flexible HVAC: HVAC rescheduled by the same model with the battery idle,
#    then the simple greedy self-consumption battery rule is applied on top of that HVAC profile.
s3_hvac, _, _ = solve_milp(df, hvac_flexible=True, battery_on=False)
ceil3 = tune_ceiling(df, s3_hvac.p_hvac.values)
s3 = simulate_dispatch(df, s3_hvac.p_hvac.values, battery_rule="greedy", soc_ceiling=ceil3)
print(f"  Strategy 3 rule: SOC ceiling tuned to {ceil3:.2f} kWh so that E_end = 20 kWh")
# 4) Full SolarSync: joint MILP
s4, obj4, info4 = solve_milp(df, hvac_flexible=True, battery_on=True)

strategies = {"1 Baseline": baseline, "2 Battery-only": s2,
              "3 Flexible-HVAC + rule": s3, "4 Full SolarSync": s4}
rows = []
for name, r in strategies.items():
    check_physics(r, name)
    k = evaluate(r)
    k["objective_rs"] = objective_of(r)
    k["strategy"] = name
    rows.append(k)
kpi = pd.DataFrame(rows).set_index("strategy")
assert abs(kpi.loc["4 Full SolarSync", "objective_rs"] - obj4) < 1e-3, "Objective mismatch"

# HVAC energy compliance for every strategy
e_base = df.hvac_baseline_kw.sum() * dt
kpi["hvac_pct_of_baseline"] = 100 * kpi.hvac_kwh / e_base
kpi["hvac_within_95_105"] = kpi.hvac_pct_of_baseline.between(95 - 1e-6, 105 + 1e-6)

pd.set_option("display.width", 200, "display.max_columns", 30)
show = ["cost_net_rs", "objective_rs", "peak_import_kw", "self_consumption_pct",
        "self_sufficiency_pct", "curtail_kwh", "import_kwh", "export_kwh", "hvac_kwh",
        "hvac_pct_of_baseline", "batt_charge_kwh", "batt_discharge_kwh", "soc_end_kwh"]
print("\nKPI comparison (same input data, E_start = E_end = 20 kWh for all):")
print(kpi[show].round(2).T.to_string())
kpi.round(4).to_csv(os.path.join(OUT_DIR, "solarsync_kpis.csv"))

# %% 10. Export optimised schedule (Full SolarSync) --------------------------------
cols = ["timestamp", "pv_available_kw", "fixed_load_kw", "hvac_baseline_kw", "p_hvac",
        "p_ch", "p_dis", "e_kwh", "e_next_kwh", "p_imp", "p_exp", "p_pvl", "p_pvc", "p_pve", "p_pvx",
        "import_tariff", "export_tariff"]
sched = s4[cols].copy()
sched["soc_pct_start"] = 100 * sched.e_kwh / P["batt_cap"]
sched.to_csv(os.path.join(OUT_DIR, "solarsync_optimized_schedule.csv"), index=False)

# %% 11. Sensitivity analysis -------------------------------------------------------
print("\nSENSITIVITY 1: export penalty (Full SolarSync). Does the penalty distort the schedule?")
sens = []
for lam in [0.0, 0.5, 1.0, 2.0, 5.0]:
    r, _, _ = solve_milp(df, lam_exp=lam)
    k = evaluate(r)
    sens.append(dict(lam_exp=lam, cost_rs=k["cost_net_rs"], peak_kw=k["peak_import_kw"],
                     export_kwh=k["export_kwh"], curtail_kwh=k["curtail_kwh"],
                     self_cons_pct=k["self_consumption_pct"], hvac_kwh=k["hvac_kwh"]))
sens = pd.DataFrame(sens)
print(sens.round(2).to_string(index=False))
sens.round(4).to_csv(os.path.join(OUT_DIR, "sensitivity_export_penalty.csv"), index=False)

print("\nSENSITIVITY 2: HVAC energy tolerance (Full SolarSync). 0 % = pure load shifting.")
sens2 = []
for tol in [0.0, 0.02, 0.05, 0.10]:
    r, _, _ = solve_milp(df, hvac_tol=tol)
    k = evaluate(r)
    sens2.append(dict(hvac_tol_pct=100 * tol, cost_rs=k["cost_net_rs"],
                      peak_kw=k["peak_import_kw"], hvac_kwh=k["hvac_kwh"],
                      self_cons_pct=k["self_consumption_pct"]))
sens2 = pd.DataFrame(sens2)
print(sens2.round(2).to_string(index=False))
sens2.round(4).to_csv(os.path.join(OUT_DIR, "sensitivity_hvac_tolerance.csv"), index=False)

# %% 12. Plots -----------------------------------------------------------------------
t = df.timestamp
hours = pd.to_datetime(t).dt.strftime("%H:%M")


def style(ax, title, ylabel):
    ax.set_title(title)
    ax.set_xlabel("Time of day (15-min intervals)")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_locator(plt.matplotlib.dates.HourLocator(interval=3))
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%H:%M"))
    ax.legend(loc="best", fontsize=8)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, name), dpi=140)


fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(t, df.pv_available_kw, label="Available PV (kW)", color="orange")
ax.plot(t, df.baseline_total_load_kw, label="Baseline building load (kW)", color="tab:blue")
style(ax, "1. Available PV generation vs building load", "Power (kW)"); save(fig, "01_pv_vs_load.png")

fig, ax = plt.subplots(figsize=(9, 4))
ax.step(t, baseline.p_imp, where="post", label="Baseline import", color="tab:red")
ax.step(t, s4.p_imp, where="post", label="Optimised import (SolarSync)", color="tab:green")
style(ax, "2. Baseline vs optimised grid import", "Grid import (kW)"); save(fig, "02_import_compare.png")

fig, ax = plt.subplots(figsize=(9, 4))
ax.step(t, s4.p_imp, where="post", label="Grid import", color="tab:red")
ax.step(t, -s4.p_exp, where="post", label="Grid export (shown negative)", color="tab:green")
ax.axhline(0, color="k", lw=0.6)
style(ax, "3. Grid imports and exports (SolarSync)", "Power (kW), import > 0"); save(fig, "03_grid_import_export.png")

fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(pd.to_datetime(list(t) + [t.iloc[-1] + pd.Timedelta(minutes=15)]),
        100 * np.append(s4.e_kwh.values, s4.e_next_kwh.values[-1]) / P["batt_cap"],
        label="SOC (%)", color="purple")
ax.axhline(20, ls="--", color="gray", lw=0.8, label="Min SOC 20 %")
ax.axhline(90, ls="--", color="gray", lw=0.8, label="Max SOC 90 %")
style(ax, "4. Battery state of charge", "SOC (% of 40 kWh)"); save(fig, "04_battery_soc.png")

fig, ax = plt.subplots(figsize=(9, 4))
ax.step(t, s4.p_ch, where="post", label="Charging (kW)", color="tab:green")
ax.step(t, -s4.p_dis, where="post", label="Discharging (shown negative, kW)", color="tab:red")
ax.axhline(0, color="k", lw=0.6)
style(ax, "5. Battery charging and discharging power", "Power (kW)"); save(fig, "05_battery_power.png")

fig, ax = plt.subplots(figsize=(9, 4))
ax.step(t, df.hvac_baseline_kw, where="post", label="Baseline HVAC", color="tab:blue")
ax.step(t, s4.p_hvac, where="post", label="Optimised HVAC (SolarSync)", color="tab:orange")
ax.step(t, df.hvac_max_kw * df.hvac_available, where="post", ls=":", color="gray", label="HVAC max")
style(ax, "6. Baseline vs optimised HVAC schedule", "HVAC power (kW)"); save(fig, "06_hvac_schedule.png")

fig, ax = plt.subplots(figsize=(9, 4))
ax.step(t, df.import_tariff, where="post", label="Import tariff (Rs/kWh)", color="tab:red")
ax.step(t, df.export_tariff, where="post", label="Export tariff (Rs/kWh)", color="tab:green")
style(ax, "7. Electricity tariffs", "Tariff (Rs/kWh)"); save(fig, "07_tariffs.png")

fig, axes = plt.subplots(1, 4, figsize=(14, 4))
for ax, (col, lab) in zip(axes, [("cost_net_rs", "Net cost (Rs/day)"),
                                 ("peak_import_kw", "Peak import (kW)"),
                                 ("self_consumption_pct", "Self-consumption (%)"),
                                 ("self_sufficiency_pct", "Self-sufficiency (%)")]):
    ax.bar(range(4), kpi[col], color=["gray", "tab:blue", "tab:orange", "tab:green"])
    ax.set_xticks(range(4)); ax.set_xticklabels(["Base", "Batt", "HVAC+rule", "Full"], fontsize=8)
    ax.set_ylabel(lab); ax.set_title(lab, fontsize=9); ax.grid(axis="y", alpha=0.3)
fig.suptitle("8. KPI comparison across strategies")
save(fig, "08_kpi_summary.png")

if "agg" not in plt.get_backend().lower():
    plt.show()
print(f"\nOutputs written to ./{OUT_DIR}/ (schedule CSV, KPI CSV, sensitivity CSVs, 8 PNG plots)")
