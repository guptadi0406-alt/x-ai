# AI Usage Documentation

## Tool Used
**Antigravity IDE with Claude Opus** (Google DeepMind Advanced Agentic Coding)

## Purpose of Assistance
- Structured implementation of the complete research replication pipeline
- Code generation for data loading, preprocessing, model training, SHAP/LIME explainers, NMR/MIP metrics, and the research extension
- Test suite generation
- Documentation drafting (README, replication report, research gap, paper draft)
- Visualization code for publication-quality figures

## Key Prompts Used
1. Master prompt requesting full replication of Salih et al. (2024) with:
   - Repository audit
   - Dataset pipeline for Airline Passenger Satisfaction
   - Four classifier implementations (DT, LR, LGBM, SVC)
   - SHAP and LIME explanation generation
   - NMR and MIP metric implementations
   - Correlation-Aware XAI Robustness Framework extension
   - Documentation and paper draft

## Files Generated/Modified with AI Assistance

| File | Purpose | AI Contribution |
|------|---------|----------------|
| `src/data/loader.py` | Dataset loading and validation | Generated, reviewed by team |
| `src/data/download_dataset.py` | Dataset download helper | Generated |
| `src/preprocessing/pipeline.py` | Preprocessing pipeline | Generated, reviewed by team |
| `src/models/classifiers.py` | Classifier training/evaluation | Generated, reviewed by team |
| `src/explainers/shap_explainer.py` | SHAP explanation module | Generated, reviewed by team |
| `src/explainers/lime_explainer.py` | LIME explanation module | Generated, reviewed by team |
| `src/metrics/nmr_mip.py` | NMR and MIP implementations | Generated from paper/repo descriptions |
| `src/metrics/ranking_metrics.py` | Ranking comparison metrics | Generated |
| `src/extension/correlation_aware.py` | Research extension | Generated, reviewed by team |
| `src/experiments/run_all.py` | Main experiment runner | Generated, reviewed by team |
| `src/visualization/plots.py` | Visualization module | Generated |
| `tests/test_pipeline.py` | Test suite | Generated, validated by running |
| `configs/experiment_config.yaml` | Experiment configuration | Generated |
| `README.md` | Project documentation | Generated, team should review |
| `docs/*.md` | Research documentation | Generated, team should review |
| `AI_USAGE.md` | This file | Generated |

## How the Team Reviewed and Validated Output

*[Team members should update this section with their actual review activities]*

- [ ] Reviewed all source code for correctness
- [ ] Verified preprocessing pipeline prevents data leakage
- [ ] Confirmed SHAP explainer types match model requirements
- [ ] Validated NMR algorithm against original repository description
- [ ] Validated MIP algorithm against original repository description
- [ ] Ran test suite and confirmed all tests pass
- [ ] Ran full experiment pipeline and reviewed results
- [ ] Compared reproduced results against paper-reported values
- [ ] Reviewed extension methodology for scientific soundness
- [ ] Verified all citations are genuine and accessible
- [ ] Proofread documentation and paper draft

## Additional Tools Used
- Python 3.14 with virtual environment
- scikit-learn 1.9.1, SHAP 0.49.1, LIME 0.2.0.1, LightGBM 4.7.0
- matplotlib 3.11.2, seaborn 0.13.2
- pytest 9.2.0 for testing

## Limitations of AI-Generated Content
- NMR and MIP algorithms were reimplemented from README examples and paper descriptions, not from full original source code
- The research extension is a proposed framework, not an established contribution
- All experimental results should be independently verified by the team
- Citations should be cross-checked for accuracy
- The paper draft is a starting point for team revision, not a finished submission
