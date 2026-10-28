# XAI Research Replication: SHAP and LIME Perspective

## Research Paper

**Title:** A Perspective on Explainable Artificial Intelligence Methods: SHAP and LIME  
**Authors:** Ahmed M. Salih, Zahra Raisi-Estabragh, Ilaria Boscolo Galazzo, Petia Radeva, Steffen E. Petersen, Karim Lekadir, Gloria Menegaz  
**Publication:** Advanced Intelligent Systems, Vol. 7, Issue 1, January 2025  
**DOI:** [https://doi.org/10.1002/aisy.202400304](https://doi.org/10.1002/aisy.202400304)  
**Course:** DA351 — Explainable AI (XAI)

## Research Problem

How do machine-learning model choice and feature collinearity affect the reliability, ranking, and interpretation of SHAP and LIME explanations?

## Dataset

**Airline Passenger Satisfaction** from Kaggle  
Source: https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction

Per the paper's case study (Section 3):
- 500 subjects (stratified sample from the full dataset)
- 250 satisfied, 250 not satisfied
- 22 features for binary classification
- 80% training / 20% testing split

## Experimental Methodology

### Phase 1: Paper Replication

1. **Four Classifiers** with default parameters:
   - Decision Tree (DT)
   - Logistic Regression (LR)
   - LightGBM (LGBM)
   - Support Vector Classifier (SVC)

2. **SHAP Analysis:**
   - Global feature importance rankings per model
   - Cross-model ranking comparison (Table 2)
   - Local explanations for selected instances

3. **LIME Analysis:**
   - Local explanations for same instances across models (Figure 3)
   - Stability analysis across random seeds

4. **NMR (Normalized Movement Rate):**
   - Measures ranking stability when top features are iteratively removed
   - Paper reports: LGBM=0.231, LR=0.275, DT=0.445, SVC=0.273

5. **MIP (Modified Index Position):**
   - Adjusts rankings to account for feature collinearity
   - Applied to the model with lowest NMR

### Phase 2: Research Extension

**Correlation-Aware XAI Robustness Framework:**
- Correlation-aware feature grouping via hierarchical clustering
- Group-level SHAP importance aggregation
- Multi-seed explanation robustness analysis
- Comparison: Baseline SHAP vs MIP vs Correlation-Aware approach

## Repository Structure

```
├── configs/                    # Experiment configuration
│   └── experiment_config.yaml
├── data/                       # Dataset (not committed)
├── docs/                       # Documentation
│   ├── replication_report.md
│   ├── research_gap.md
│   └── experimental_results.md
├── results/                    # Generated results
│   ├── figures/                # All plots
│   └── tables/                 # CSV tables
├── src/                        # Source code
│   ├── data/                   # Dataset loading & validation
│   ├── preprocessing/          # Feature engineering pipeline
│   ├── models/                 # Classifier implementations
│   ├── explainers/             # SHAP and LIME modules
│   ├── metrics/                # NMR, MIP, ranking metrics
│   ├── extension/              # Research extension
│   ├── experiments/            # Experiment runners
│   └── visualization/          # Plot generation
├── tests/                      # Test suite
├── AI_USAGE.md                 # AI assistance documentation
├── README.md                   # This file
└── requirements.txt            # Dependencies
```

## Installation

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download dataset
python -m src.data.download_dataset
# Follow the instructions to place the CSV in data/
```

## Running Experiments

```bash
# Activate virtual environment
source venv/bin/activate

# Run the complete pipeline
python -m src.experiments.run_all

# Or specify a custom data path
python -m src.experiments.run_all --data data/train.csv

# Run tests
python -m pytest tests/ -v
```

## Original vs. Reproduced Results

| Metric | Paper | Ours | Difference |
|--------|-------|------|------------|
| DT Accuracy | 0.84 | See results/ | — |
| LR Accuracy | 0.85 | See results/ | — |
| LGBM Accuracy | 0.91 | See results/ | — |
| SVC Accuracy | 0.86 | See results/ | — |
| DT NMR | 0.445 | See results/ | — |
| LR NMR | 0.275 | See results/ | — |
| LGBM NMR | 0.231 | See results/ | — |
| SVC NMR | 0.273 | See results/ | — |

Results are populated after running the experiment pipeline.

## Limitations

1. **Sample selection:** The paper does not disclose the exact random seed or sampling procedure for the 500-subject subset. Our sampling is an approximation.
2. **NMR/MIP algorithms:** Reimplemented from the README examples and paper descriptions, as the full original source code was not accessible. Results may differ from the original implementation.
3. **SVC SHAP:** Uses KernelSHAP approximation, which is computationally expensive and may produce slightly different results depending on background sample selection.

## References

1. Salih, A.M. et al. (2024). "A Perspective on Explainable Artificial Intelligence Methods: SHAP and LIME." *Advanced Intelligent Systems*, 7(1), 2400304. https://doi.org/10.1002/aisy.202400304
2. Lundberg, S.M. & Lee, S.I. (2017). "A Unified Approach to Interpreting Model Predictions." *NeurIPS*.
3. Ribeiro, M.T. et al. (2016). "'Why Should I Trust You?': Explaining the Predictions of Any Classifier." *KDD*.
4. Salih, A.M. et al. (2022). "Normalized Movement Rate." *IEEE ICHI*. https://doi.org/10.1109/ICHI54592.2022.00040
5. Salih, A.M. et al. (2024). "Characterizing the Contribution of Dependent Features in XAI Methods." *IEEE JBHI*.
6. Original NMR code: https://github.com/amaa11/NMR
7. Original MIP code: https://github.com/amaa11/MIP

## License

This project is for academic purposes (DA351 course project).
