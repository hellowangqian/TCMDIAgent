"""
TCMAgent — the agent core.

Architecture (search -> reason -> tool use -> referee):

  Stage 1 SEARCH   : retrieve analogs of the query TCM and drug from the
                     TRAIN-split knowledge base (T1 similarity + T2 profile).
  Stage 2 REASON   : fuse retrieval evidence into a structured evidence vector
                     (weighted profile statistics + chemical descriptors).
  Stage 3 TOOL USE : run RDKit alert tools (T3) and descriptor tools (T4),
                     then invoke the delegate predictor (T5) on the fused view.
  Stage 4 REFEREE  : logistic referee maps the evidence bundle to the final
                     contraindication probability.

The design mirrors ReAct / TxAgent-style agentic reasoning, but every step is
deterministic and reproducible: no external LLM calls are required at
inference time. The LLMs are replaced by learned light-weight modules that
play the same roles (searcher, reasoner, tool orchestrator, referee).
"""

import numpy as np
import pandas as pd

from agent_tools import (
    FingerprintSimilarityTool, InteractionProfileTool,
    SubstructureAlertTool, PhysicochemicalDescriptorTool,
    MLPredictorTool,
    DRUG_ALERT_NAMES, TCM_ALERT_NAMES,
)


class TCMAgent:
    def __init__(self, tcm_smiles, drug_smiles, n_analogs_tcm=10, n_analogs_drug=10,
                 min_sim=0.05, seed=42):
        """
        tcm_smiles   : dict tcm_name -> list of component SMILES (train KB only)
        drug_smiles  : dict drugbank_id -> SMILES
        """
        self.tcm_smiles = tcm_smiles
        self.drug_smiles = drug_smiles
        self.k = n_analogs_tcm
        self.kd = n_analogs_drug
        self.min_sim = min_sim
        self.seed = seed
        self.alert_tool = SubstructureAlertTool()
        self.desc_tool = PhysicochemicalDescriptorTool()
        self.sim_tcm = None
        self.sim_drug = None
        self.profile = None
        self.delegate = MLPredictorTool(seed=seed)
        self.referee = None

    # ------------------------------------------------------------------
    # KB construction (per fold; uses ONLY the train split)
    # ------------------------------------------------------------------
    def build_knowledge_base(self, train_tcms, train_drugs, train_pairs,
                             tcm_matrix, drug_matrix, X_train, y_train):
        """
        train_tcms / train_drugs : ordered lists of entity ids in the train KB
        tcm_matrix / drug_matrix : (n, d) fingerprint matrices aligned with lists
        train_pairs              : iterable of (tcm, drug) positive pairs in train
        X_train, y_train         : delegate-model training matrix and labels
        """
        self.sim_tcm = FingerprintSimilarityTool(train_tcms, tcm_matrix)
        self.sim_drug = FingerprintSimilarityTool(train_drugs, drug_matrix)
        self.profile = InteractionProfileTool(train_pairs, train_tcms, train_drugs)
        self.delegate.fit(X_train, y_train)

    # ------------------------------------------------------------------
    # Stage 1 SEARCH : analog retrieval over the train KB
    # ------------------------------------------------------------------
    def search(self, tcm_name, drug_id):
        tcm_vec = self._tcm_vector(tcm_name)
        tcm_analogs = self.sim_tcm.analogs(tcm_vec, k=self.k, min_sim=self.min_sim)
        # query drug against the train drug KB (zero vector if unseen)
        j = self.sim_drug._index.get(drug_id)
        if j is None:
            dvec = np.zeros(self.sim_drug.M.shape[1], dtype=self.sim_drug.M.dtype)
        else:
            dvec = self.sim_drug.M[j]
        drug_analogs = self.sim_drug.analogs(dvec, k=self.kd, min_sim=self.min_sim)
        return tcm_analogs, drug_analogs

    # ------------------------------------------------------------------
    # Stage 2 REASON : weighted profile statistics from retrieval evidence
    # ------------------------------------------------------------------
    def reason(self, tcm_name, drug_id):
        tcm_analogs, drug_analogs = self.search(tcm_name, drug_id)
        # TCM side: weighted positive-rate of analogs against the query drug
        num = 0.0
        den = 0.0
        for a, s in tcm_analogs:
            i = self.profile.tcm_index.get(a)
            j = self.profile.drug_index.get(drug_id)
            if i is not None and j is not None:
                num += s * self.profile.P[i, j]
                den += s
        tcm_evidence = num / den if den > 0 else self.profile.pos_rate
        # Drug side: weighted positive-rate of analogs against the query TCM
        num = 0.0
        den = 0.0
        j = self.profile.drug_index.get(drug_id)
        if j is not None:
            for a, s in drug_analogs:
                i = self.profile.tcm_index.get(tcm_name)
                if i is not None:
                    num += s * self.profile.P[i, self.profile.drug_index.get(a, -1)] \
                        if self.profile.drug_index.get(a, -1) >= 0 else 0.0
                    den += s
        drug_evidence = num / den if den > 0 else self.profile.pos_rate
        # Global profile statistics
        global_rate = self.profile.pos_rate
        return tcm_analogs, drug_analogs, tcm_evidence, drug_evidence, global_rate

    # ------------------------------------------------------------------
    # Stage 3 TOOL USE + Stage 4 REFEREE : evidence bundle -> probability
    # ------------------------------------------------------------------
    def build_evidence(self, tcm_name, drug_id, p_delegate=None):
        tcm_analogs, drug_analogs, tcm_ev, drug_ev, g = self.reason(tcm_name, drug_id)
        # similarity statistics: top-1 and mean similarity of retrieved analogs
        t_sim1 = tcm_analogs[0][1] if tcm_analogs else 0.0
        t_simm = float(np.mean([s for _, s in tcm_analogs])) if tcm_analogs else 0.0
        d_sim1 = drug_analogs[0][1] if drug_analogs else 0.0
        d_simm = float(np.mean([s for _, s in drug_analogs])) if drug_analogs else 0.0
        # T3 alerts
        dsmiles = self.drug_smiles.get(drug_id, "")
        comps = self.tcm_smiles.get(tcm_name, [])
        d_alert = self.alert_tool.drug_alerts(dsmiles)
        c_alerts = [self.alert_tool.comp_alerts(s) for s in comps] or [np.zeros(len(TCM_ALERT_NAMES), np.float32)]
        c_alert = np.mean(c_alerts, axis=0)
        # T4 descriptors
        d_desc = self.desc_tool.describe(dsmiles)
        c_descs = [self.desc_tool.describe(s) for s in comps] or [np.zeros(4, np.float32)]
        c_desc = np.mean(c_descs, axis=0)
        # T5 delegate logit
        del_logit = np.log(np.clip(p_delegate, 1e-6, 1 - 1e-6) /
                           np.clip(1 - p_delegate, 1e-6, 1)) if p_delegate is not None else 0.0
        feats = np.concatenate([
            np.array([tcm_ev, drug_ev, g, len(tcm_analogs), len(drug_analogs),
                      t_sim1, t_simm, d_sim1, d_simm,
                      del_logit], np.float32),
            d_alert, c_alert, d_desc, c_desc,
        ])
        return feats

    # ------------------------------------------------------------------
    # Delegate features for ALL pairs (train + test) — evidence bundle
    # ------------------------------------------------------------------
    def evidence_matrix(self, pairs, verbose=False):
        rows = []
        for n, (t, d) in enumerate(pairs):
            rows.append(self.build_evidence(t, d))
            if verbose and (n + 1) % 20000 == 0:
                print(f"  evidence {n+1}/{len(pairs)}")
        return np.vstack(rows)

    def add_delegate_column(self, X_ev, p_del):
        """Append the delegate logit column to a precomputed evidence matrix."""
        p = np.clip(np.asarray(p_del, dtype=np.float64), 1e-6, 1 - 1e-6)
        col = np.log(p / (1 - p)).astype(np.float32).reshape(-1, 1)
        return np.hstack([X_ev, col]).astype(np.float32)

    # ------------------------------------------------------------------
    # Referee training (gradient-boosted referee on evidence bundles)
    # ------------------------------------------------------------------
    def fit_referee(self, X_ev, y):
        import lightgbm as lgb
        self.referee = lgb.LGBMClassifier(
            n_estimators=150, max_depth=4, learning_rate=0.08,
            subsample=0.8, colsample_bytree=0.8,
            num_leaves=15, random_state=self.seed, n_jobs=-1, verbose=-1,
        )
        self.referee.fit(X_ev, y)
        return self

    def predict_proba(self, X_ev):
        return self.referee.predict_proba(X_ev)[:, 1]

    # ------------------------------------------------------------------
    def _tcm_vector(self, tcm_name):
        i = self.sim_tcm._index.get(tcm_name)
        if i is None:
            # entity not in the train KB (unseen TCM): zero query vector
            return np.zeros(self.sim_tcm.M.shape[1], dtype=self.sim_tcm.M.dtype)
        return self.sim_tcm.M[i]
