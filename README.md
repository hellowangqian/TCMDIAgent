# TCMDIAgent

A zero-training, tool-augmented LLM agent for **TCM–drug interaction
(contraindication) prediction** on the [TCMDI493](https://github.com/hellowangqian/TCMDI)
benchmark. For each query pair, the agent grounds its judgment in verifiable
train-split evidence — the *K* most similar labeled training cases (entity-masked
fingerprint retrieval) and a trained ML tool's contraindication probability —
and a reasoning LLM weighs this evidence against its pharmacological knowledge,
with *T*-sample self-consistency averaging producing the final score. No agent
training or fine-tuning is involved.

This repository contains the code, the released test subsets with selection
indices, and the result tables accompanying the paper

> **TCMDIAgent: A Machine Learning Tool-Augmented LLM Agent for Traditional
> Chinese Medicine Drug Interaction Prediction**

## Repository layout

```
TCMDIAgent/
├── code/
│   ├── agent_tools.py        # atomic evidence tools: case retrieval, ML
│   │                         #   probability, substructure alerts, descriptors
│   ├── agent_core.py         # agent core (knowledge base, search, evidence)
│   ├── run_experiments.py    # agent + evidence pipelines over the debiased splits
│   ├── run_core.py           # composite-fingerprint (Agent-Core) grid
│   └── run_final.py          # full agent grid (Agent-Full, 4 splits x 3 seeds)
├── data/
│   ├── tcm_drug_pairs.csv            # 279,826 positive pairs (12.0 MB)
│   ├── drug_smiles.csv               # DrugBank SMILES for the 4,503 drugs
│   ├── tcm_components.csv            # TCM -> active-component SMILES
│   ├── unique_tcms.csv / unique_drugs.csv
│   ├── feature_cache.pkl             # precomputed 1024-d Morgan fingerprints
│   │                                 #   (drug_fp / tcm_fp dicts, 19.9 MB)
│   └── composite_feature_cache.pkl   # precomputed 1,408-d composite fingerprints
│                                     #   (27.2 MB)
└── results/
    ├── test_subsets/         # the 36 released evaluation subsets
    │   │                     #   (4 splits x 3 data seeds x 3 selection seeds,
    │   │                     #   600 stratified pairs each), the selection
    │   │                     #   indices (subset_indices.json) and a summary
    │   └── subset_summary.csv
    └── *.csv / *.md          # result tables (see below)
```

The two fingerprint caches are derived from `tcm_drug_pairs.csv` +
`tcm_components.csv` via the TCMDI493 feature pipeline; they are included so
the agent runs out of the box. Delete them if you prefer to regenerate.

## Requirements

- Python 3.10+
- `pip install -r requirements.txt` (numpy, pandas, scikit-learn, lightgbm,
  rdkit, openai)
- LLM judging uses a locally served OpenAI-compatible endpoint (vLLM);
  see `code/` for the judge configuration.

## Data splits

The debiased evaluation uses the benchmark's four protocols — `Random`,
`TCMCold`, `DrugCold`, `TCMDrugCold` — each generated with three data seeds
(42/43/44), stored as `DB-{split}_seed{seed}.pkl`. Obtain them from the
[TCMDI493 benchmark repository](https://github.com/hellowangqian/TCMDI) and
place them under `data/data_splits/`.

## Reproducing the evaluation

```bash
cd code

# Agent evidence/ablation pipelines on the debiased splits
python run_experiments.py --splits tcmDrugCold --seeds 42

# Composite-fingerprint (Agent-Core) grid
python run_core.py

# Full agent grid, 4 splits x 3 seeds
python run_final.py
```

Every LLM evaluation in the paper uses the released 600-pair subsets in
`results/test_subsets/`; selection indices are fixed and released, so each
reported number can be reproduced exactly, and the agent and the retrained ML
baselines are compared on identical subsets.

## Result tables

Key files in `results/` and the paper artifacts they back:

| File(s) | Paper artifact |
|---|---|
| `ml_baseline_600pair_all.csv`, `crossattn_600pair.csv`, `unified_multi_split.csv` | main comparison (Tab. 1) |
| `tcmdi_agent_sc_{lgb,xgb,crossattn}_subsets*.csv` | per-subset agent results (Fig. 2, per-seed) |
| `sel_seed_sensitivity.csv` | selection-seed stability analysis |
| `ablation_components_{lgb,crossattn}*.csv` | component ablations (Tab. 2/3) |
| `sc_n_sensitivity_summary.csv`, `sc_n_table2_*.csv`, `ablation_sc_n_jointcold.csv` | self-consistency *T* sweep (Fig. 3a) |
| `agent_grounding_600.csv`, `ablation_agent_sc_noml_600.csv`, `prompt_sensitivity.csv` | prompt-evidence ablation |
| `agent_summary.csv`, `agent_results.csv` | aggregate agent scores |
| `calibration_diagnosis.csv` | leakage / calibration diagnostics |
| `ablation_summary_for_paper.md`, `experiment_summary_table.md` | human-readable summaries |

## TODO before release

- [ ] Add `code/llm_judge.py` (judge prompt template, vLLM client,
  self-consistency loop, score extraction) from the experiment server.

## License

Released under the MIT License; see [LICENSE](LICENSE).
The underlying TCMDI493 benchmark is governed by its own license at the
[benchmark repository](https://github.com/hellowangqian/TCMDI).

## Citation

If you use this code or the released subsets, please cite the paper above and
the TCMDI493 benchmark.
