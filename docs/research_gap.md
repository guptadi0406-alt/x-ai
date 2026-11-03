# Research Gap Analysis

## 1. What Limitation Exists

Salih et al. (2024a) demonstrate two critical limitations of standard local and global XAI techniques (SHAP and LIME):

**Model dependency:** The same XAI method produces disparate feature importance rankings when applied to different machine learning models trained on identical data (Section 2, Figures 2–3, Table 2). Even when classifier predictive accuracies are statistically comparable, the top-ranked "most important" features differ substantially across architectures.

**Feature collinearity:** Both SHAP (marginal sampling formulation) and LIME assume feature independence when evaluating counterfactual subsets or perturbing inputs. When features exhibit multicollinearity ($|r| > 0.5$), this independence assumption leads to off-manifold / out-of-distribution sampling, evaluating unfeasible feature combinations. Consequently, SHAP arbitrarily splits attribution credit among collinear features, artificially depressing the importance of genuinely predictive variables. LIME's local linear surrogate similarly fails to capture simultaneous, correlated feature movements.

---

## 2. What Previous Research Has Attempted

### Normalized Movement Rate (NMR) (Salih et al., IEEE ICHI 2022)
- Quantifies the *stability* of feature rankings when top-ranked features are iteratively ablated from the model.
- Measures rank volatility to identify which model produces the most consistent XAI output.
- **Limitation:** Functions strictly as a descriptive diagnostic of instability; does not adjust rankings, rectify collinearity distortions, or group interchangeable features.

### Modified Index Position (MIP) (Salih et al., IEEE JBHI 2024b)
- Post-hoc adjustment of XAI rankings designed to recover features suppressed by multicollinearity.
- Tracks upward positional shifts of ablated feature indices to reorder and elevate masked predictors.
- **Limitation:** Requires iterative model retraining ($O(K)$ retraining runs for $K$ ablated features), which is computationally prohibitive on larger datasets. Furthermore, MIP outputs an altered single-feature ranking rather than providing joint group-level attribution or solving off-manifold evaluation.

### Conditional SHAP / SHAP with Correlated Features
- Aas et al. (2021) proposed conditional SHAP using empirical copulas and joint distributions.
  - DOI: https://doi.org/10.1007/s10994-021-06016-y
- **Limitation:** Computationally intensive ($O(2^M)$ conditional expectation calculations); primarily constrained to local explanations; sensitive to user hyperparameter configurations.

### shapr Package (Aas et al., 2021)
- R implementation approximating conditional Shapley values accounting for feature dependencies in the game-theoretic formulation itself.
- **Limitation:** R-only ecosystem; limited to local explanations; computationally expensive for standard tabular pipelines.

---

## 3. What Remains Unresolved

Despite these advances, several critical gaps persist in the literature:

1. **Absence of a unified, lightweight framework:** No existing pipeline integrates collinearity detection, group-level attribution, rank adjustment, and multi-seed stability quantification in a single leakage-free Python workflow without requiring expensive model retraining.
2. **Lack of explicit group-level explanations:** Prior methods force individual-feature rankings, concealing functional redundancy and collinear clusters from end-users and practitioners.
3. **Inconsistent seed-stability quantification:** Explanation variance across perturbation seeds (e.g., LIME perturbation samples, KernelSHAP background subsets) is rarely benchmarked systematically across diverse model families.
4. **Controlled benchmark comparison:** Direct head-to-head empirical evaluations comparing baseline SHAP/LIME, MIP, and correlation-aware group adjustments on identical train/test splits are sparse.

---

## 4. What Our Extension Changes

We propose a **Correlation-Aware XAI Robustness Framework** comprising four modular components:

### Component 1: Correlation-Aware Feature Grouping
- Complete-linkage hierarchical clustering on the training-data Pearson correlation matrix using distance metric $d = 1 - |r|$.
- Features with correlation $|r| \ge \theta$ (default threshold $\theta = 0.6$, distance cutoff $1 - \theta = 0.4$) are partitioned into disjoint functional groups $G_k$.
- Executed strictly on training splits to guarantee zero test-set leakage.

### Component 2: Group-Level SHAP Importance
- Computes aggregated attribution for each correlated cluster:
  $$\Phi(G_k) = \sum_{j \in G_k} |\phi_j|$$
- Provides users with an interpretable, cluster-level summary ("Cluster $G_k$ accounts for $\Phi(G_k)$ total attribution") that eliminates arbitrary credit-splitting artifacts.

### Component 3: Correlation-Aware Ranking Adjustment
- Blends individual feature attribution with group weight to adjust individual feature positions:
  $$\bar{\phi}_j = \alpha \cdot |\phi_j| + (1 - \alpha) \cdot \frac{\Phi(G_k)}{|G_k|}, \quad \text{for } j \in G_k$$
- Elevates suppressed features that belong to high-impact collinear clusters without requiring model retraining.
- Serves as an efficient complement to, rather than a full replacement for, standard point attributions.

### Component 4: Multi-Seed Perturbation Stability Analysis
- Repeats explanation generation across multiple stochastic seeds (evaluating LIME surrogate variance and KernelSHAP background sampling sensitivity).
- Systematically calculates rank consistency metrics:
  - Spearman rank correlation ($r_s$)
  - Kendall's rank correlation ($\tau$)
  - Top-$k$ Jaccard index ($\text{Jaccard}@5$)

---

## 5. How We Experimentally Evaluate It

### Controlled Comparison
We benchmark four approaches across four distinct model families (Decision Tree, Logistic Regression, LightGBM, and Support Vector Classifier) on the identical 22-feature Airline Passenger Satisfaction dataset:
- **A.** Baseline SHAP rankings (TreeSHAP / LinearSHAP / KernelSHAP)
- **B.** Baseline LIME rankings
- **C.** MIP-adjusted rankings (Salih et al., 2024b)
- **D.** Correlation-Aware rankings & group attributions (Our Extension)

### Metrics
- **Prediction performance:** Accuracy, Macro-F1, ROC-AUC (invariant, as our method operates strictly post-hoc).
- **Ranking stability:** Multi-seed Spearman $r_s$, Kendall's $\tau$, and $\text{Jaccard}@5$.
- **Cross-model agreement:** Pairwise rank correlation between differing classifier architectures on identical test instances.
- **Interpretability & redundancy reduction:** Dimensionality reduction from individual features to collinear clusters.
- **Computational efficiency:** Wall-clock execution time compared to MIP's retraining loop.

### Hypotheses & Expectations
- **$H_1$ (Replication):** Different classifier architectures yield divergent feature rankings, and high-correlation feature pairs experience significant rank instability under ablation.
- **$H_2$ (Extension):** Correlation-aware grouping stabilizes attributions against credit splitting. However, because tree ensembles and linear models project collinear manifolds onto fundamentally different decision boundaries, cross-model agreement may remain constrained by architectural inductive biases.

---

## 6. What Evidence Supports or Contradicts Our Hypothesis

**Supporting evidence:**
- High intra-model stability across random perturbation seeds ($r_s > 0.90$).
- Consistent identification and clustering of known collinear pairs (e.g., *Departure Delay* $\leftrightarrow$ *Arrival Delay*, *Inflight entertainment* $\leftrightarrow$ *Cleanliness*).
- Near-zero computational overhead relative to $O(K)$ iterative retraining required by MIP.
- Meaningful recovery of collinear features previously suppressed in baseline SHAP rankings.

**Contradicting or nuanced evidence:**
- Grouping does not necessarily enforce cross-model consensus if underlying model families rely on genuinely distinct decision surfaces (e.g., tree threshold splits vs. hyperplane margins).
- Excessive threshold values ($\theta > 0.8$) leave collinear features ungrouped, whereas overly permissive thresholds ($\theta < 0.4$) merge distinct conceptual features.

---

## References

1. Salih, A.M. et al. (2024a). "A Perspective on Explainable AI Methods: SHAP and LIME." *Advanced Intelligent Systems*, 6(12), 2400304. https://doi.org/10.1002/aisy.202400304
2. Salih, A.M. et al. (2024b). "Characterizing the Contribution of Dependent Features in XAI Methods." *IEEE Journal of Biomedical and Health Informatics (JBHI)*. https://doi.org/10.1109/JBHI.2024.3418579
3. Salih, A.M. et al. (2022). "A Perspective on XAI for Biomedical Data." *IEEE International Conference on Healthcare Informatics (ICHI)*. https://doi.org/10.1109/ICHI54592.2022.00040
4. Lundberg, S.M. & Lee, S.I. (2017). "A Unified Approach to Interpreting Model Predictions." *NeurIPS 2017*, 30, 4765–4774.
5. Ribeiro, M.T., Singh, S., & Guestrin, C. (2016). "'Why Should I Trust You?': Explaining the Predictions of Any Classifier." *ACM KDD 2016*, 1135–1144.
6. Aas, K., Jullum, M., & Løland, A. (2021). "Explaining individual predictions when features are dependent: More accurate approximations to Shapley values." *Machine Learning*, 110(6), 1379–1428. https://doi.org/10.1007/s10994-021-06016-y
7. Molnar, C. (2022). *Interpretable Machine Learning: A Guide for Making Black Box Models Explainable* (2nd ed.). https://christophm.github.io/interpretable-ml-book/
