"""Run agent_core (composite-only, no tool features) for seeds 43,44 on all 4 splits."""
import os, sys, pickle, time
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_experiments as R

t0 = time.time()
with open(os.path.join(R.BENCH, "data", "composite_feature_cache.pkl"), "rb") as f:
    comp = pickle.load(f)
ctf, cdf = comp["tcm_fp"], comp["drug_fp"]
DIM = len(next(iter(ctf.values())))

tcms_all = sorted(ctf.keys()); drugs_all = sorted(cdf.keys())
tix = {t: i for i, t in enumerate(tcms_all)}; dix = {d: i for i, d in enumerate(drugs_all)}
T = np.vstack([np.asarray(ctf[t], np.float32) for t in tcms_all])
D = np.vstack([np.asarray(cdf[d], np.float32) for d in drugs_all])

def build_X(pairs):
    n = len(pairs)
    ti = np.array([tix[t] for t, _ in pairs], np.int64)
    di = np.array([dix[d] for _, d in pairs], np.int64)
    X = np.empty((n, 4 * DIM), np.float32)
    for s in range(0, n, 40000):
        e = min(s + 40000, n)
        tv, dv = T[ti[s:e]], D[di[s:e]]
        X[s:e, 0:DIM] = tv
        X[s:e, DIM:2*DIM] = dv
        X[s:e, 2*DIM:3*DIM] = np.abs(tv - dv)
        X[s:e, 3*DIM:4*DIM] = tv * dv
    return X

def lgbm(s):
    return lgb.LGBMClassifier(n_estimators=150, max_depth=6, learning_rate=0.1,
                              subsample=0.8, colsample_bytree=0.8, random_state=s,
                              n_jobs=-1, verbose=-1)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
os.makedirs(OUT, exist_ok=True)
rows = []
for split in ["tcmDrugCold", "tcmCold", "drugCold", "random"]:
    for seed in [43, 44]:
        df = pickle.load(open(os.path.join(R.SPLIT_DIR, f"DB-{split}_seed{seed}.pkl"), "rb"))
        train = df[df["split"] == "train"]; test = df[df["split"] == "test"]
        tok = train["tcm_name"].isin(ctf) & train["drugbank_id"].isin(cdf)
        teok = test["tcm_name"].isin(ctf) & test["drugbank_id"].isin(cdf)
        train_pairs = list(zip(train[tok]["tcm_name"], train[tok]["drugbank_id"]))
        y_tr = train[tok]["label"].values.astype(int)
        test_pairs = list(zip(test[teok]["tcm_name"], test[teok]["drugbank_id"]))
        y_te = test[teok]["label"].values.astype(int)
        X_tr = build_X(train_pairs); X_te = build_X(test_pairs)
        m = lgbm(seed); m.fit(X_tr, y_tr)
        p = m.predict_proba(X_te)[:, 1]
        auc = roc_auc_score(y_te, p) * 100
        rows.append(dict(split=split, seed=seed, agent_core=round(auc, 1)))
        print(f"[{split} s{seed}] agent_core {auc:.1f}  ({time.time()-t0:.0f}s)", flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(OUT, "core_grid_progress.csv"), index=False)
        del X_tr, X_te
pd.DataFrame(rows).to_csv(os.path.join(OUT, "core_grid.csv"), index=False)
print("done", time.time()-t0)
