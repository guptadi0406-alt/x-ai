# Research Gap Analysis

## 1. What Limitation Exists

Salih et al. (2024) demonstrate two critical limitations of SHAP and LIME:

**Model dependency:** The same XAI method produces different feature rankings when applied to different ML models trained on the same data (Section 2, Figures 2-3, Table 2). Even when classifier accuracies are comparable, the "most important" features differ substantially.

**Feature collinearity:** Both SHAP and LIME assume feature independence. When features are correlated, SHAP may assign artificially low importance to genuinely relevant features because their predictive contribution is already captured by correlated features. LIME's local linear approximation similarly fails to account for simultaneous feature changes.

## 2. What Previous Research Has Attempted

### NMR (Salih et al., IEEE ICHI 2022)
- Measures the *stability* of feature rankings when top features are iteratively removed
- Identifies which model produces the most stable XAI output
- **Limitation:** Diagnoses instability but does not correct it

### MIP (Salih et al., IEEE JBHI 2024)
- Post-hoc adjustment of XAI rankings to account for multicollinearity
- Reorders features based on positional shifts when correlated features are removed
- **Limitation:** Requires retraining models multiple times; computationally expensive

### Conditional SHAP / SHAP with correlated features
- Aas et al. (2021) proposed conditional SHAP using the joint distribution
  - DOI: https://doi.org/10.1007/s10994-021-06016-y
- **Limitation:** Limited to local explanation; user-dependent configurations

### shapr package (Aas et al., 2021)
- R package implementing approximate conditional Shapley values
- Accounts for feature dependencies in the SHAP computation itself
- **Limitation:** R-only; limited to local explanations; computationally intensive

## 3. What Remains Unresolved

Despite these advances, several gaps persist in our selected setting:

1. **No unified framework** combines collinearity detection, ranking adjustment, and stability assessment into a single pipeline
2. **Group-level explanations** are rarely provided — users see individual feature importance but not which features are functionally interchangeable due to correlation
3. **Stability quantification across random seeds** is not standard practice — LIME explanations can vary significantly across runs, but this is rarely measured systematically
4. **Cross-method comparison** (baseline SHAP vs MIP vs grouped explanations) lacks controlled experimental evaluation on the same benchmark

## 4. What Our Extension Changes

We propose a **Correlation-Aware XAI Robustness Framework** with four components:

### Component 1: Correlation-Aware Feature Grouping
- Hierarchical clustering on the training-data correlation matrix
- Groups features that are correlated above a defensible threshold (|r| > 0.6)
- Uses only training data (no test-set leakage)

### Component 2: Group-Level SHAP Importance
- Aggregates individual SHAP values within correlated groups
- Provides a higher-level view: "this group of features contributes X" rather than splitting credit among correlated features
- More robust to collinearity artifacts

### Component 3: Correlation-Aware Ranking
- Adjusts individual feature rankings by accounting for group membership
- Features in larger correlated groups get redistributed credit
- Complements (not replaces) standard SHAP rankings

### Component 4: Multi-Seed Stability Analysis
- Repeats SHAP analysis across multiple random seeds
- Measures Spearman rank correlation, Kendall's tau, and Jaccard top-k overlap
- Quantifies how stable each model's explanations are

## 5. How We Experimentally Evaluate It

### Controlled Comparison
We compare four approaches on the same dataset and models:
- **A.** Baseline SHAP rankings
- **B.** Baseline LIME rankings
- **C.** MIP-adjusted rankings (paper's method)
- **D.** Our correlation-aware rankings

### Metrics
- **Prediction performance:** Accuracy, F1, ROC-AUC (should be unchanged — our method is post-hoc)
- **Ranking stability:** Spearman correlation across seeds, Jaccard@5
- **Cross-model agreement:** Do different models agree more when using our method?
- **Group interpretability:** Are grouped explanations easier to reason about?
- **Computational cost:** Runtime comparison

### Expectations
- Our method should NOT improve prediction accuracy (it's post-hoc)
- It SHOULD improve ranking stability and provide clearer grouped explanations
- It should offer a useful complement to MIP, not necessarily replace it

## 6. What Evidence Would Support or Contradict Our Hypothesis

**Supporting evidence:**
- Higher Spearman correlation across seeds (more stable rankings)
- Higher cross-model Jaccard@5 (more agreement between models)
- Correlated features are grouped together rather than arbitrarily split
- Computational overhead is reasonable

**Contradicting evidence:**
- Rankings are already stable without grouping
- Grouping obscures genuinely distinct features
- The threshold selection is too sensitive (different thresholds → very different results)
- No improvement over MIP in any metric

## References

1. Salih, A.M. et al. (2024). "A Perspective on Explainable AI Methods: SHAP and LIME." *Advanced Intelligent Systems*. https://doi.org/10.1002/aisy.202400304
2. Lundberg, S.M. & Lee, S.I. (2017). "A Unified Approach to Interpreting Model Predictions." *NeurIPS*.
3. Ribeiro, M.T. et al. (2016). "'Why Should I Trust You?'" *KDD*.
4. Aas, K. et al. (2021). "Explaining individual predictions when features are dependent." *Machine Learning*, 110, 1379–1428. https://doi.org/10.1007/s10994-021-06016-y
5. Salih, A.M. et al. (2022). "A Perspective on XAI for Biomedical Data." *IEEE ICHI*. https://doi.org/10.1109/ICHI54592.2022.00040
6. Salih, A.M. et al. (2024). "Characterizing the Contribution of Dependent Features in XAI Methods." *IEEE JBHI*.
7. Molnar, C. (2022). *Interpretable Machine Learning*. https://christophm.github.io/interpretable-ml-book/
