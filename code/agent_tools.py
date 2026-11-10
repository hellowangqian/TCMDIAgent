"""
TCMAgent — Tool library.

Implements the atomic tools invoked by the TCMAgent reasoning loop:
  T1  FingerprintSimilarityTool      : cosine/Tanimoto retrieval over the train KB
  T2  InteractionProfileTool         : train-set interaction profiles (label propagation)
  T3  SubstructureAlertTool          : RDKit toxicophore / pharmacophore alerts
  T4  PhysicochemicalDescriptorTool  : RDKit MW / logP / TPSA / HBD ...
  T5  MLPredictorTool                : delegate model (LightGBM surrogate)

All tools are deterministic and operate only on the TRAIN split knowledge base.
"""

import os
import sys

import numpy as np
import pandas as pd

# rdkit bootstrap: some sandboxed Windows environments block the DLL of newer
# rdkit builds; set RDKIT_ALT_PATH to a locally extracted rdkit wheel if needed.
try:
    from rdkit import Chem
except Exception:  # ImportError (missing) or OSError (DLL blocked by policy)
    if "RDKIT_ALT_PATH" not in os.environ:
        raise
    _ALT = os.path.normpath(os.environ["RDKIT_ALT_PATH"])
    # purge any partially-imported rdkit modules, then re-import from _ALT
    for _m in [m for m in list(sys.modules) if m == "rdkit" or m.startswith("rdkit.")]:
        del sys.modules[_m]
    sys.path = [p for p in sys.path
                if "site-packages" not in p.lower() or "rdkit" not in p.lower()]
    sys.path.insert(0, _ALT)
    from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

# ----------------------------------------------------------------------
# Alert SMARTS library
# ----------------------------------------------------------------------
# drug-side: pharmacophore classes known to drive herb-drug interactions
DRUG_ALERTS = {
    "coumarin_4hydroxy":    ["Oc1cc(=O)oc2ccccc12", "O=c1oc2ccccc2cc1O"],
    "sulfonylurea_sulfonam":["S(=O)(=O)N"],
    "azole_antifungal":     ["c1nc[nH]n1", "c1nnc[nH]1", "c1ncc[nH]1"],
    "nitro":                ["[N+](=O)[O-]"],
    "catechol":             ["c(O)c(O)"],
    "furan":                ["o1cccc1"],
    "aniline":              "[NX3;H2,H1;!$(N[!#6])][c]",
    "epoxide":              "C1OC1",
    "quinoline":            ["c1ccc2ncccc2c1", "n1ccc2ccccc12"],
    "carboxylic_acid":      "C(=O)[OX2H1]",
    "steroid_nucleus":      ["C1CC2CCC1(C)C2", "C1CCC2(C)C1CCC2"],
    "tertiary_amine":       "[NX3;H0;!$(N[!#6]);!$(N[!#6])]",
}
# TCM-component-side: structural classes of herbal constituents
TCM_ALERTS = {
    "anthraquinone":        "O=C1c2ccccc2C(=O)c2ccccc21",
    "quinone":              ["O=C1C=CC(=O)C=C1", "O=C1CC(=O)c2ccccc21"],
    "flavonoid":            ["O=c1oc2ccccc2cc1", "O=c1cc(-c2ccccc2)oc2cc(O)ccc12"],
    "coumarin_4hydroxy":    ["Oc1cc(=O)oc2ccccc12"],
    "polyphenol_diOH":      "c(O)c(O)",
    "basic_amine":          "[NX3;H2,H1;!$(N[!#6]);!$(N*C=O)]",
    "tertiary_amine":       "[NX3;H0;!$(N[!#6]);!$(N[!#6])]",
    "glycoside_ether":      "[OX2][CX4][CX4]",
    "furan":                "o1cccc1",
    "lactone":              "O=C1OCC1",
    "ester":                "C(=O)O[C@H]",
}

def _compile(pat_list):
    pats = []
    if isinstance(pat_list, str):
        pat_list = [pat_list]
    for s in pat_list:
        try:
            m = Chem.MolFromSmarts(s)
            if m is not None:
                pats.append(m)
        except Exception:
            pass
    return pats

_DRUG_PATS = {k: _compile(v) for k, v in DRUG_ALERTS.items()}
_TCM_PATS = {k: _compile(v) for k, v in TCM_ALERTS.items()}
DRUG_ALERT_NAMES = list(DRUG_ALERTS.keys())
TCM_ALERT_NAMES = list(TCM_ALERTS.keys())


class SubstructureAlertTool:
    """T3: returns binary alert vectors for a drug SMILES and component SMILES."""

    def __init__(self):
        self._drug_cache = {}
        self._comp_cache = {}

    @staticmethod
    def _hit(mol, pats):
        if mol is None:
            return 0
        for p in pats:
            if mol.HasSubstructMatch(p):
                return 1
        return 0

    def drug_alerts(self, smiles):
        if smiles in self._drug_cache:
            return self._drug_cache[smiles]
        mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) and smiles else None
        vec = np.array([self._hit(mol, p) for p in _DRUG_PATS.values()], dtype=np.float32)
        self._drug_cache[smiles] = vec
        return vec

    def comp_alerts(self, smiles):
        if smiles in self._comp_cache:
            return self._comp_cache[smiles]
        mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) and smiles else None
        vec = np.array([self._hit(mol, p) for p in _TCM_PATS.values()], dtype=np.float32)
        self._comp_cache[smiles] = vec
        return vec


class PhysicochemicalDescriptorTool:
    """T4: RDKit descriptors with caching."""

    def __init__(self):
        self.cache = {}

    def describe(self, smiles):
        if smiles in self.cache:
            return self.cache[smiles]
        out = np.zeros(4, dtype=np.float32)
        try:
            mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) and smiles else None
            if mol is not None:
                out[:] = [
                    Descriptors.MolWt(mol),
                    Descriptors.MolLogP(mol),
                    rdMolDescriptors.CalcTPSA(mol),
                    Descriptors.NumHDonors(mol),
                ]
        except Exception:
            pass
        self.cache[smiles] = out
        return out


class FingerprintSimilarityTool:
    """T1: cosine-similarity retrieval of analog entities from the train KB."""

    def __init__(self, entity_ids, entity_matrix):
        # entity_matrix: (n_entities, d), L2-normalised rows
        self.ids = list(entity_ids)
        self.M = entity_matrix / (np.linalg.norm(entity_matrix, axis=1, keepdims=True) + 1e-9)
        self._index = {e: i for i, e in enumerate(self.ids)}

    def analogs(self, query_vec, k=10, min_sim=0.05):
        q = query_vec / (np.linalg.norm(query_vec) + 1e-9)
        sims = self.M @ q
        order = np.argsort(-sims)[:k]
        out = []
        for i in order:
            if sims[i] >= min_sim:
                out.append((self.ids[i], float(sims[i])))
        return out


class InteractionProfileTool:
    """T2: train-set positive-pair knowledge base P[tcm, drug] in {0,1}."""

    def __init__(self, train_pairs, tcm_ids, drug_ids):
        self.tcm_index = {t: i for i, t in enumerate(tcm_ids)}
        self.drug_index = {d: i for i, d in enumerate(drug_ids)}
        self.P = np.zeros((len(tcm_ids), len(drug_ids)), dtype=np.float32)
        for t, d in train_pairs:
            i, j = self.tcm_index.get(t), self.drug_index.get(d)
            if i is not None and j is not None:
                self.P[i, j] = 1.0
        self.pos_rate = float(self.P.mean())

    def profile_tcm(self, tcm_weights):
        """Given weight vector over train TCMs, return per-drug interaction propensity."""
        return tcm_weights @ self.P                       # (n_drugs,)

    def profile_drug(self, drug_weights):
        """Given weight vector over train drugs, return per-TCM interaction propensity."""
        return self.P @ drug_weights                      # (n_tcms,)


class MLPredictorTool:
    """T5: LightGBM surrogate trained on the train split of the current fold."""

    def __init__(self, seed=42):
        self.seed = seed
        self.model = None

    def fit(self, X, y):
        import lightgbm as lgb
        self.model = lgb.LGBMClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, random_state=self.seed,
            n_jobs=-1, verbose=-1,
        )
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict_proba(X)[:, 1]
