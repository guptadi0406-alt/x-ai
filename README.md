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
- 80% training / 20% testing split (fitted on train only to prevent data leakage)

## Experimental Methodology

### Phase 1: Paper Replication

1. **Four Classifiers** with default parameters:
   - Decision Tree (DT)
   - Logistic Regression (LR) *(Note: `max_iter=1000` applied to guarantee convergence)*
   - LightGBM (LGBM)
   - Support Vector Classifier (SVC) *(Note: `probability=True` enabled for LIME and ROC-AUC)*

2. **SHAP Analysis:**
   - Global feature importance rankings per model (TreeSHAP, LinearSHAP, KernelSHAP)
   - Cross-model ranking comparison (reproducing Table 2)
   - Local explanations for selected instances

3. **LIME Analysis:**
   - Local explanations for same instances across models (Figure 3)
   - Stability analysis across random seeds for all classifiers

4. **NMR (Normalized Movement Rate):**
   - Measures ranking stability when top features are iteratively removed
   - Paper reports: LGBM=0.231, LR=0.275, DT=0.445, SVC=0.273

5. **MIP (Modified Index Position):**
   - Adjusts rankings to account for feature collinearity
   - Applied to the model with lowest NMR

### Phase 2: Research Extension

**Correlation-Aware XAI Robustness Framework:**
- Correlation-aware feature grouping via complete-linkage hierarchical clustering with distance $d = 1 - |r|$
- Group-level SHAP importance aggregation
- Multi-seed explanation robustness analysis
- Comparison: Baseline SHAP vs MIP vs Correlation-Aware approach
- Explicit reporting and analysis of contradicting evidence

## Repository Structure

```
├── configs/                    # Experiment configuration
│   └── experiment_config.yaml
├── data/                       # Dataset (not committed)
├── docs/                       # Documentation & paper drafts
│   ├── replication_report.md
│   ├── research_gap.md
│   ├── experimental_results.md
│   └── paper_draft.md
├── results/                    # Generated results
│   ├── figures/                # All publication plots
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
├── tests/                      # Pytest test suite
├── AI_USAGE.md                 # AI assistance documentation
├── README.md                   # This file
└── requirements.txt            # Dependencies
```

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd x-ai

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/macOS:
source venv/bin/activate
# On Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# On Windows (CMD):
.\venv\Scripts\activate.bat

# Install dependencies
pip install -r requirements.txt

# Download dataset (or place CSV in data/airline_satisfaction.csv)
python -m src.data.download_dataset
```

## Running Experiments

```bash
# Run the complete experimental pipeline
python -m src.experiments.run_all

# Or specify custom config / data path
python -m src.experiments.run_all --config configs/experiment_config.yaml --data data/airline_satisfaction.csv

# Run test suite
python -m pytest tests/test_pipeline.py -v
```

## Original vs. Reproduced Results

| Metric | Paper Value | Reproduced (Ours) | Difference ($\Delta$) | Status |
|---|---|---|---|---|
| **DT Accuracy** | 0.840 | 0.830 | -0.010 | Closely matched |
| **LR Accuracy** | 0.850 | 0.830 | -0.020 | Closely matched |
| **LGBM Accuracy** | 0.910 | 0.950 | +0.040 | Sample variance |
| **SVC Accuracy** | 0.860 | 0.880 | +0.020 | Closely matched |
| **DT NMR** | 0.445 | 0.422 | -0.023 | Closely matched |
| **LR NMR** | 0.275 | 0.324 | +0.049 | Reasonable match |
| **LGBM NMR** | 0.231 | 0.360 | +0.129 | Deviates (see Discussion) |
| **SVC NMR** | 0.273 | 0.342 | +0.069 | Reasonable match |

### Extension Findings & Method Comparison

| Model | Baseline vs. CA Spearman | Model Stability (Mean Spearman across Seeds) |
|---|---|---|
| **DT** | 0.986 | 0.912 |
| **LR** | 0.968 | 1.000 |
| **LGBM** | 0.975 | 1.000 |
| **SVC** | 0.997 | 0.996 |

**Cross-Model Agreement:**
- Baseline SHAP: **0.569**
- Correlation-Aware (CA): **0.555**

> **Honest Scientific Evaluation:** While Correlation-Aware rankings demonstrate high within-model consistency ($r_s > 0.96$) and group attribution clarifies multi-feature blocks (such as Departure/Arrival delays), CA ranking slightly decreases cross-model agreement ($0.555$ vs $0.569$). This occurs because tree-based and linear models distribute credit across correlated groups fundamentally differently. See `docs/paper_draft.md` Section 8 for complete analysis.

## Limitations

1. **Sample selection:** The original paper does not disclose the random seed used to sample the 500-subject cohort from Kaggle's 130K records.
2. **Reimplementation:** NMR and MIP are reimplemented following the methodological description from Salih et al. (2022, 2024b) and GitHub specifications.
3. **KernelSHAP:** Model-agnostic explanations for SVC rely on background sampling approximations.

## References

1. Salih, A.M. et al. (2024). "A Perspective on Explainable Artificial Intelligence Methods: SHAP and LIME." *Advanced Intelligent Systems*, 7(1), 2400304. https://doi.org/10.1002/aisy.202400304
2. Lundberg, S.M. & Lee, S.I. (2017). "A Unified Approach to Interpreting Model Predictions." *NeurIPS*.
3. Ribeiro, M.T. et al. (2016). "'Why Should I Trust You?': Explaining the Predictions of Any Classifier." *KDD*.
4. Salih, A.M. et al. (2022). "Normalized Movement Rate." *IEEE ICHI*. https://doi.org/10.1109/ICHI54592.2022.00040
5. Salih, A.M. et al. (2024b). "Characterizing the Contribution of Dependent Features in XAI Methods." *IEEE JBHI*.

## License

This project is developed for academic purposes (DA351 course project).
