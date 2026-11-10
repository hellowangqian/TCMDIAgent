"""
Final TCMAgent experiment grid: 4 splits x 3 seeds.

Agent-P (tool-augmented agent predictor):
  composite-1408 interaction features [T, D, |T-D|, T*D]
  + T3 substructure alerts (drug 12, TCM 10)
  + T4 physicochemical descriptors (drug 4, TCM 4, cross 8)
  -> LightGBM per fold.

Ablation (seed42 only): composite-only (no tool features).
All splits/caches read-only; results -> TCMAgent/results/.
"""
import os, sys, pickle, time
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_experiments as R
from agent_tools import SubstructureAlertTool, PhysicochemicalDescriptorTool

t0 = time.time()
BENCH = R.BENCH
with open(os.path.join(BENCH, "data", "composite_feature_cache.pkl"), "rb") as f:
    comp = pickle.load(f)
ctf, cdf = comp["tcm_fp"], comp["drug_fp"]
DIM = len(next(iter(ctf.values())))
print(f"composite dim {DIM}  ({time.time()-t0:.0f}s)", flush=True)

# ---- entity-level tool features ----
al = SubstructureAlertTool(); de = PhysicochemicalDescriptorTool()
sm = pd.read_csv(R.SMILES_CSV); sm = sm[sm["valid"] == True]
_sm0 = next(iter(sm["drugbank_id"]))
N_DA, N_DD = len(al.drug_alerts("")), 4
drug_alert = {}; drug_desc = {}
for d, s in zip(sm["drugbank_id"], sm["smiles"]):
    drug_alert[d] = al.drug_alerts(s); drug_desc[d] = de.describe(s)
N_DA = len(drug_alert[_sm0])
tcm_comp = R.load_tcm_components()
tcms_all = sorted(ctf.keys())
tcm_alert = {}; tcm_desc = {}
for t in tcms_all:
    comps = tcm_comp.get(t, [])
    if comps:
        tcm_alert[t] = np.mean([al.comp_alerts(s) for s in comps], axis=0)
        tcm_desc[t] = np.mean([de.describe(s) for s in comps], axis=0)
    else:
        tcm_alert[t] = np.zeros(N_TA, np.float32); tcm_desc[t] = np.zeros(N_DD, np.float32)
_t0 = next(iter(tcm_alert))
N_TA = len(tcm_alert[_t0])
print(f"tool features cached (N_DA={N_DA}, N_TA={N_TA})  ({time.time()-t0:.0f}s)", flush=True)

# entity matrices aligned to composite dicts
tcms_all = sorted(ctf.keys()); drugs_all = sorted(cdf.keys())
tix = {t: i for i, t in enumerate(tcms_all)}; dix = {d: i for i, d in enumerate(drugs_all)}
T = np.vstack([np.asarray(ctf[t], np.float32) for t in tcms_all])
D = np.vstack([np.asarray(cdf[d], np.float32) for d in drugs_all])
TA = np.vstack([tcm_alert[t] for t in tcms_all]); TD = np.vstack([tcm_desc[t] for t in tcms_all])
DA = np.vstack([drug_alert.get(d, np.zeros(N_DA, np.float32)) for d in drugs_all])
DD = np.vstack([drug_desc.get(d, np.zeros(N_DD, np.float32)) for d in drugs_all])
ND_T = TD.shape[1]

TOOL_DIM = N_DA + N_TA + N_DD + ND_T + 4 + 4   # alerts + desc + cross
FULL_DIM = 4 * DIM + TOOL_DIM

def build_X(pairs, with_tools=True):
    n = len(pairs)
    ti = np.array([tix[t] for t, _ in pairs], np.int64)
    di = np.array([dix[d] for _, d in pairs], np.int64)
    dim = FULL_DIM if with_tools else 4 * DIM
    X = np.empty((n, dim), np.float32)
    CH = 40000
    for s in range(0, n, CH):
        e = min(s + CH, n)
        tv, dv = T[ti[s:e]], D[di[s:e]]
        o = s * (4 * DIM)
        X[s:e, 0:DIM] = tv
        X[s:e, DIM:2*DIM] = dv
        X[s:e, 2*DIM:3*DIM] = np.abs(tv - dv)
        X[s:e, 3*DIM:4*DIM] = tv * dv
        if with_tools:
            td, dd = TD[ti[s:e]], DD[di[s:e]]
            base = 4 * DIM
            X[s:e, base:base+N_DA] = DA[di[s:e]]
            X[s:e, base+N_DA:base+N_DA+N_TA] = TA[ti[s:e]]
            b2 = base + N_DA + N_TA
            X[s:e, b2:b2+N_DD] = dd
            X[s:e, b2+N_DD:b2+N_DD+ND_T] = td
            b3 = b2 + N_DD + ND_T
            X[s:e, b3:b3+4] = np.abs(td - dd)
            X[s:e, b3+4:b3+8] = td * dd
    return X

def lgbm(s):
    return lgb.LGBMClassifier(n_estimators=150, max_depth=6, learning_rate=0.1,
                              subsample=0.8, colsample_bytree=0.8, random_state=s,
                              n_jobs=-1, verbose=-1)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
os.makedirs(OUT, exist_ok=True)
rows = []
for split in ["tcmDrugCold", "tcmCold", "drugCold", "random"]:
    for seed in [42, 43, 44]:
        f = os.path.join(R.SPLIT_DIR, f"DB-{split}_seed{seed}.pkl")
        df = pickle.load(open(f, "rb"))
        train = df[df["split"] == "train"]; test = df[df["split"] == "test"]
        tok = train["tcm_name"].isin(ctf) & train["drugbank_id"].isin(cdf)
        teok = test["tcm_name"].isin(ctf) & test["drugbank_id"].isin(cdf)
        train_pairs = list(zip(train[tok]["tcm_name"], train[tok]["drugbank_id"]))
        y_tr = train[tok]["label"].values.astype(int)
        test_pairs = list(zip(test[teok]["tcm_name"], test[teok]["drugbank_id"]))
        y_te = test[teok]["label"].values.astype(int)
        print(f"\n[{split} s{seed}] train {len(train_pairs)} test {len(test_pairs)} "
              f"({time.time()-t0:.0f}s)", flush=True)
        X_tr = build_X(train_pairs, with_tools=True)
        X_te = build_X(test_pairs, with_tools=True)
        m = lgbm(seed); m.fit(X_tr, y_tr)
        p = m.predict_proba(X_te)[:, 1]
        auc_full = roc_auc_score(y_te, p) * 100
        del X_tr
        res = dict(split=split, seed=seed, n_test=len(test_pairs),
                   agent_full=round(auc_full, 1))
        if seed == 42:  # ablation: composite-only
            X_tr2 = build_X(train_pairs, with_tools=False)
            X_te2 = build_X(test_pairs, with_tools=False)
            m2 = lgbm(seed); m2.fit(X_tr2, y_tr)
            p2 = m2.predict_proba(X_te2)[:, 1]
            res["agent_core"] = round(roc_auc_score(y_te, p2) * 100, 1)
            del X_tr2, X_te2
        del X_te
        rows.append(res)
        print(f"  agent_full {auc_full:.1f}  agent_core {res.get('agent_core')}  "
              f"({time.time()-t0:.0f}s)", flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(OUT, "final_grid_progress.csv"), index=False)

res = pd.DataFrame(rows)
res.to_csv(os.path.join(OUT, "final_grid.csv"), index=False)
print("\n=== Summary ===")
print(res.groupby("split")[["agent_full", "agent_core"]].agg(["mean", "std"]).round(2))
print(f"\ntotal {time.time()-t0:.0f}s")
