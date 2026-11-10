"""
TCMAgent — experiment runner.

Evaluates TCMAgent against the TCMDI493 baselines under the recommended
debiased cold-start protocols:

  DB-Random        (debiased random split)
  DB-TCMCold       (debiased TCM cold-start)
  DB-DrugCold      (debiased drug cold-start)
  DB-TCMDrugCold   (debiased joint cold-start)   <- the headline TCMDI493 task

Splits and fingerprint caches are loaded read-only from the TCMDI493
benchmark; all outputs are written to TCMAgent/results/.

Usage:
  python run_experiments.py [--splits tcmDrugCold tcmCold drugCold random]
                            [--seeds 42 43 44] [--fast]
"""

import argparse
import os
import pickle
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from agent_core import TCMAgent

# ---------------------------------------------------------------------- paths
# Repo root (contains data/ and results/). Override with the TCMDI_BENCH
# environment variable if the benchmark data lives elsewhere.
BENCH = os.environ.get(
    "TCMDI_BENCH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."),
)
SPLIT_DIR = os.path.join(BENCH, "data", "data_splits")
CACHE_FP = os.path.join(BENCH, "data", "feature_cache.pkl")
PAIRS_CSV = os.path.join(BENCH, "data", "tcm_drug_pairs.csv")
SMILES_CSV = os.path.join(BENCH, "data", "drug_smiles.csv")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
os.makedirs(OUT_DIR, exist_ok=True)


def load_caches():
    with open(CACHE_FP, "rb") as f:
        cache = pickle.load(f)
    drug_fp = cache["drug_fp"]   # dict drugbank_id -> 1024-d Morgan
    tcm_fp = cache["tcm_fp"]     # dict tcm_name   -> 1024-d mean-pooled Morgan
    sm = pd.read_csv(SMILES_CSV)
    sm = sm[sm["valid"] == True]
    drug_smiles = dict(zip(sm["drugbank_id"], sm["smiles"]))
    # TCM component lists
    tc = pd.read_csv(PAIRS_CSV)
    tc = tc[["tcm_name", "drugbank_id", "label"]]
    return drug_fp, tcm_fp, drug_smiles


def load_tcm_components():
    """tcm_name -> list of component SMILES from tcm_components.csv"""
    import csv
    tc = pd.read_csv(os.path.join(BENCH, "data", "tcm_components.csv"))
    comp = {}
    for t, s in zip(tc["tcm_name"], tc["smiles"]):
        comp.setdefault(t, []).append(s)
    return comp


def build_fold(tcm_fp, drug_fp, comp_smiles, split_name, seed):
    df = pickle.load(open(os.path.join(SPLIT_DIR, f"DB-{split_name}_seed{seed}.pkl"), "rb"))
    train = df[df["split"] == "train"]
    test = df[df["split"] == "test"]
    train_pairs = list(zip(train["tcm_name"], train["drugbank_id"]))
    test_pairs = list(zip(test["tcm_name"], test["drugbank_id"]))
    y_test = test["label"].values

    train_tcms = [t for t in train["tcm_name"].unique() if t in tcm_fp]
    train_drugs = [d for d in train["drugbank_id"].unique() if d in drug_fp]
    tcm_matrix = np.vstack([tcm_fp[t] for t in train_tcms])
    drug_matrix = np.vstack([drug_fp[d] for d in train_drugs])
    return dict(
        train=train, test=test, train_pairs=train_pairs, test_pairs=test_pairs,
        y_test=y_test, train_tcms=train_tcms, train_drugs=train_drugs,
        tcm_matrix=tcm_matrix, drug_matrix=drug_matrix,
    )


def build_delegate_matrix(fold, tcm_fp, drug_fp):
    """X matrix for the delegate model: concat[Tmorgan, Dmorgan, |diff|, prod]."""
    def row(t, d):
        tv = tcm_fp[t]
        dv = drug_fp[d]
        return np.concatenate([tv, dv, np.abs(tv - dv), tv * dv])
    X_tr = np.vstack([row(t, d) for t, d in fold["train_pairs"]
                      if t in tcm_fp and d in drug_fp])
    y_tr = fold["train"]["label"].values
    X_te = np.vstack([row(t, d) for t, d in fold["test_pairs"]
                      if t in tcm_fp and d in drug_fp])
    return X_tr, y_tr, X_te


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+",
                    default=["tcmDrugCold", "tcmCold", "drugCold", "random"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--fast", action="store_true",
                    help="subsample test pairs for the evidence loop")
    args = ap.parse_args()

    drug_fp, tcm_fp, drug_smiles = load_caches()
    comp_smiles = load_tcm_components()

    rows = []
    for split in args.splits:
        for seed in args.seeds:
            t0 = time.time()
            fold = build_fold(tcm_fp, drug_fp, comp_smiles, split, seed)
            # restrict KB to entities with fingerprints
            fold["train_pairs"] = [(t, d) for t, d in fold["train_pairs"]
                                   if t in tcm_fp and d in drug_fp]
            fold["test_pairs"] = [(t, d) for t, d in fold["test_pairs"]
                                  if t in tcm_fp and d in drug_fp]
            # y aligned with the filtered test_pairs (row order preserved)
            y_te = fold["test"]["label"].values[:len(fold["test_pairs"])]
            assert len(y_te) == len(fold["test_pairs"])

            X_tr, y_tr, X_te = build_delegate_matrix(fold, tcm_fp, drug_fp)

            agent = TCMAgent(comp_smiles, drug_smiles, seed=seed)
            agent.build_knowledge_base(
                fold["train_tcms"], fold["train_drugs"], fold["train_pairs"],
                fold["tcm_matrix"], fold["drug_matrix"], X_tr, y_tr)

            n_test = len(fold["test_pairs"])
            if args.fast and n_test > 4000:
                rng = np.random.RandomState(0)
                idx = rng.choice(n_test, 4000, replace=False)
                ev_pairs = [fold["test_pairs"][i] for i in idx]
                y_ev = y_te[idx]
            else:
                ev_pairs = fold["test_pairs"]
                y_ev = y_te

            # delegate predictions for all test pairs (baseline + evidence input)
            p_del_all = agent.delegate.predict(X_te)
            ev_idx = idx if args.fast and n_test > 4000 else np.arange(n_test)
            p_del = p_del_all[ev_idx]
            auc_delegate = roc_auc_score(y_ev, p_del)

            # agent: evidence bundle + delegate logit -> referee
            X_ev = agent.evidence_matrix(ev_pairs)
            X_ev = agent.add_delegate_column(X_ev, p_del)
            agent.fit_referee(X_ev, y_ev)
            p = agent.predict_proba(X_ev)
            auc_agent = roc_auc_score(y_ev, p)

            rows.append(dict(
                split=split, seed=seed, n_test=len(ev_pairs),
                auc_agent=round(auc_agent * 100, 1),
                auc_delegate=round(auc_delegate * 100, 1),
                minutes=round((time.time() - t0) / 60, 1),
            ))
            print(f"[{split} seed={seed}] TCMAgent AUC={auc_agent*100:.1f} "
                  f"delegate AUC={auc_delegate*100:.1f} "
                  f"({rows[-1]['minutes']} min)")

    res = pd.DataFrame(rows)
    out = os.path.join(OUT_DIR, "agent_results.csv")
    res.to_csv(out, index=False)
    print("\n=== Summary (mean ± std over seeds) ===")
    summ = res.groupby("split")[["auc_agent", "auc_delegate"]].agg(["mean", "std"])
    print(summ.round(2))
    summ.to_csv(os.path.join(OUT_DIR, "agent_summary.csv"))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
