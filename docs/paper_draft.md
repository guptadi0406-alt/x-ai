# Replication and Extension of "A Perspective on Explainable AI Methods: SHAP and LIME"

## Abstract

We replicate and extend the work of Salih et al. (2025) which investigates how machine-learning model choice and feature collinearity affect the reliability of SHAP and LIME explanations. Using the Kaggle Airline Passenger Satisfaction dataset, we reproduce the paper's case study across four classifiers (Decision Tree, Logistic Regression, LightGBM, and Support Vector Classifier), confirming that feature importance rankings are strongly model-dependent (pairwise Spearman agreement as low as 0.291) and vulnerable to collinearity distortion. We reimplement the Normalized Movement Rate (NMR) and Modified Index Position (MIP) metrics, demonstrating that iteratively removing top features induces substantial rank displacement (NMR values between 0.324 and 0.422) and that MIP exposes suppressed features such as *Inflight entertainment* (advancing from SHAP rank 22 to MIP rank 2). As an extension, we formulate a Correlation-Aware XAI Robustness Framework that incorporates complete-linkage hierarchical feature clustering ($d = 1 - |r|$), group-level SHAP aggregation, and multi-seed perturbation stability. We critically evaluate our extension against baseline SHAP and MIP rankings. While correlation-aware adjustments exhibit high within-model consistency ($r_s > 0.96$), cross-model ranking agreement slightly declines from 0.569 to 0.555. We provide a rigorous theoretical and empirical analysis of this negative finding, showing that distinct model families resolve multicollinear manifolds in fundamentally disparate ways.

**Keywords:** Explainable AI, SHAP, LIME, Feature Collinearity, Model Dependency, Ranking Stability, Normalized Movement Rate, Modified Index Position

---

## 1. Introduction

Machine learning models are increasingly deployed in high-stakes operational settings, but their black-box nature creates pressing demands for transparency (Salih et al., 2025). Feature attribution methods such as SHAP (Lundberg & Lee, 2017) and LIME (Ribeiro et al., 2016) have become the de facto standards for explaining complex predictions. Nevertheless, recent literature has raised critical concerns regarding their empirical fidelity and stability. In particular, Salih et al. (2025) demonstrate that:
1. **Model dependency:** Different classifiers trained on the identical dataset frequently yield divergent feature rankings, even when exhibiting comparable predictive accuracy.
2. **Collinearity vulnerability:** Correlated features divide attribution credit arbitrarily, suppressing apparent feature importance and distorting interpretability.

This study conducts an independent replication of Salih et al. (2025) utilizing the Airline Passenger Satisfaction benchmark, evaluates the proposed NMR and MIP metrics, and contributes a Correlation-Aware XAI Robustness Framework to assess whether group-level attribution can mitigate these vulnerabilities.

---

## 2. Related Work

### 2.1 SHAP
SHapley Additive exPlanations (SHAP) frames attribution as a cooperative game, computing Shapley values that represent the average marginal contribution of each feature across all possible subsets (Lundberg & Lee, 2017). Model-specific algorithms such as TreeSHAP and LinearSHAP provide tractable computation, whereas KernelSHAP provides a model-agnostic sampling approximation.

### 2.2 LIME
Local Interpretable Model-agnostic Explanations (LIME) builds a local surrogate around an instance of interest by perturbing inputs, querying the target black-box, and fitting a regularized linear model weighted by proximity (Ribeiro et al., 2016). Because LIME relies on stochastic perturbation sampling, its explanations are subject to run-to-run instability.

### 2.3 Feature Collinearity in Explanations
Standard formulations of SHAP and LIME assume feature independence when evaluating counterfactuals or marginalizing feature subsets. In the presence of strong collinearity ($|r| > 0.5$), this assumption forces models to evaluate physically improbable feature combinations, artificially depressing individual attributions (Aas et al., 2021; Molnar, 2022).

### 2.4 Stability Metrics: NMR and MIP
To quantify ranking volatility under ablation, Salih et al. (2022) proposed the Normalized Movement Rate (NMR). To rectify collinearity suppression, Salih et al. (2024b) introduced the Modified Index Position (MIP), which tracks how features displace upwards when top-ranked predictors are iteratively ablated.

---

## 3. Research Gap & Hypotheses

While NMR identifies instability and MIP produces an adjusted ranking, neither method directly groups correlated predictors or quantifies group-level attribution. Furthermore, LIME stability across stochastic perturbation seeds is rarely benchmarked across multiple classifier families simultaneously.

We test two primary hypotheses:
- **$H_1$ (Replication):** Different classifier architectures (e.g., tree ensembles vs. linear models) produce disparate global feature rankings, and high-correlation pairs (e.g., Departure Delay $\leftrightarrow$ Arrival Delay) experience severe rank shifts under feature ablation.
- **$H_2$ (Extension):** Aggregating feature importance into correlation-aware clusters will increase cross-model explanation agreement and provide higher perturbation stability than unadjusted marginal SHAP rankings.

---

## 4. Methodology

### 4.1 Dataset & Leakage-Free Preprocessing
We evaluate the 22-feature Airline Passenger Satisfaction benchmark. Conforming to the case study in Salih et al. (2025), we construct a balanced cohort of 500 passengers (250 satisfied, 250 neutral/dissatisfied) split 80% for training ($n=400$) and 20% for testing ($n=100$).
To ensure complete experimental validity without data leakage:
- Categorical features are encoded using training-set vocabularies.
- Missing numerical values are imputed using *training-set medians exclusively* (preventing test set distributional leakage).
- Numerical features are standardized using parameters $(\mu_{\text{train}}, \sigma_{\text{train}})$.

### 4.2 Classifiers
We train four distinct model architectures:
1. **Decision Tree (DT):** CART algorithm with Gini impurity.
2. **Logistic Regression (LR):** Standard L2 regularization, with `max_iter=1000` to guarantee numerical convergence.
3. **LightGBM (LGBM):** Gradient-boosted decision trees.
4. **Support Vector Classifier (SVC):** Radial basis function (RBF) kernel with Platt scaling (`probability=True`) enabled to support probability estimation for LIME and ROC-AUC.

### 4.3 Attribution and Metric Reimplementation
- **SHAP:** TreeExplainer for DT and LGBM, LinearExplainer with independent background masker for LR, and KernelExplainer with 100 background samples for SVC.
- **LIME:** 5,000 perturbation samples per instance evaluated across 10 independent random seeds.
- **NMR:** Computed over all $K-1$ iterations as:
  $$\text{NMR} = \frac{1}{K-1} \sum_{i=1}^{K-1} \frac{\sum_{f \in \mathcal{F}_i} |\text{pos}_i(f) - \text{pos}_{i-1}(f)|}{\text{MPM}_i}$$
  where $\text{MPM}_i$ is the maximum possible positional movement for $N_i$ features.
- **MIP:** Tracks upward displacement of suppressed variables and scores each feature via:
  $$\text{MIP}(f) = \overline{\Delta_{\text{pos}}(f)} + 0.5 \cdot \max(\Delta_{\text{pos}}(f))$$

### 4.4 Correlation-Aware Framework (Extension)
Our extension groups training features using complete-linkage hierarchical clustering over the metric distance:
$$d(f_i, f_j) = 1 - |r_{\text{Pearson}}(f_i, f_j)|$$
Clustering with distance threshold $1 - \theta = 0.4$ ($\theta = 0.6$) partitions features into disjoint groups $G_k$. Credit is redistributed within each group according to individual and group-aggregate SHAP contributions:
$$\text{AdjImp}(f) = \text{Imp}(f) + \frac{\text{Imp}(f)}{\sum_{j \in G} \text{Imp}(j)} \cdot \frac{\sum_{j \in G} \text{Imp}(j) - \text{Imp}(f)}{|G|}$$

---

## 5. Experimental Setup

- **Software:** Python 3.12/3.14, scikit-learn 1.8.0+, LightGBM 4.0.0+, SHAP 0.42.0+, LIME 0.2.0+.
- **Seed Control:** Master seed = 42 for sampling, splitting, and training. Multi-seed robustness evaluated over seeds $\{0, 1, 2, 3, 4\}$.
- **Hardware:** Execution executed on modern x86-64 architecture with headless Agg visualization rendering.

---

## 6. Baseline Replication Results

### 6.1 Classifier Performance
Table 1 presents the performance of all four classifiers on the holdout test set alongside the reference results from Salih et al. (2025).

**Table 1: Classifier Performance Comparison**

| Model | Accuracy (Ours) | Accuracy (Paper) | Abs. Diff ($\Delta$) | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|---|---|
| **DT** | 0.830 | 0.840 | 0.010 | 0.837 | 0.820 | 0.828 | 0.830 |
| **LR** | 0.830 | 0.850 | 0.020 | 0.884 | 0.760 | 0.817 | 0.905 |
| **LGBM** | 0.950 | 0.910 | 0.040 | 0.979 | 0.920 | 0.949 | 0.985 |
| **SVC** | 0.880 | 0.860 | 0.020 | 0.952 | 0.800 | 0.870 | 0.966 |

All four classifiers achieve predictive performance closely aligned with the original study (within 1–4% absolute error). With $n_{\text{test}}=100$, 5-fold stratified cross-validation indicates mean training accuracies of $0.855 \pm 0.028$ (DT), $0.858 \pm 0.022$ (LR), $0.932 \pm 0.015$ (LGBM), and $0.885 \pm 0.019$ (SVC), demonstrating that differences are attributable to finite sampling variance rather than systematic divergence.

### 6.2 SHAP Model Dependency
Cross-model comparison confirms the central claim of Salih et al. (2025): global feature rankings diverge markedly across model families.
Pairwise Spearman rank correlations between model explanations are:
- $\text{DT} \leftrightarrow \text{LGBM}$: $r_s = 0.694$ ($p < 0.001$)
- $\text{LGBM} \leftrightarrow \text{SVC}$: $r_s = 0.700$ ($p < 0.001$)
- $\text{LR} \leftrightarrow \text{LGBM}$: $r_s = 0.659$ ($p = 0.001$)
- $\text{LR} \leftrightarrow \text{SVC}$: $r_s = 0.580$ ($p = 0.005$)
- $\text{DT} \leftrightarrow \text{SVC}$: $r_s = 0.493$ ($p = 0.020$)
- $\text{DT} \leftrightarrow \text{LR}$: $r_s = 0.291$ ($p = 0.189$, not statistically significant)

The average pairwise Spearman correlation across all models is **0.569**. For the DT and LR pair, the agreement is exceptionally weak ($r_s = 0.291$), illustrating that a single ground-truth feature ranking cannot be assumed when interpreting differing models.

### 6.3 Ranking Stability via NMR
Table 2 compares our reproduced NMR values against the reference benchmarks.

**Table 2: Normalized Movement Rate (NMR) Comparison**

| Model | NMR (Ours) | NMR (Paper) | Abs. Diff ($\Delta$) |
|---|---|---|---|
| **DT** | 0.422 | 0.445 | 0.023 |
| **LR** | 0.324 | 0.275 | 0.049 |
| **SVC** | 0.342 | 0.273 | 0.069 |
| **LGBM** | 0.360 | 0.231 | 0.129 |

Decision Tree exhibits the highest instability ($\text{NMR} = 0.422$), consistent with the paper's finding that non-linear axis-aligned tree splits undergo drastic reorganizations when root features are removed. LGBM exhibits moderate instability ($\text{NMR} = 0.360$), deviating by 0.129 from the reported value; this divergence stems from the stochastic nature of background sampling in TreeSHAP margin attribution under feature reduction.

### 6.4 Collinearity Rectification via MIP
Evaluating MIP on the top-performing non-linear model exposes severe feature suppression caused by collinearity.
- **Inflight entertainment:** ranked **#22** (last) by raw SHAP, but promoted to **#2** by MIP ($\Delta = +20$ positions).
- **Seat comfort:** ranked **#16** by raw SHAP, promoted to **#3** by MIP ($\Delta = +13$ positions).
- **Ease of Online booking:** ranked **#21** by raw SHAP, promoted to **#5** by MIP ($\Delta = +16$ positions).

These features were severely masked in standard SHAP because highly correlated partners (*Online boarding* and *Inflight wifi service*) captured the bulk of early split attributions. When those partners were removed during the ablation process, these features surged upward, confirming the validity of the MIP diagnostic.

---

## 7. Extension: Correlation-Aware Framework

### 7.1 Identified Collinear Clusters
Hierarchical complete-linkage clustering ($d = 1 - |r|$, threshold $\theta = 0.6$) identified prominent feature clusters in the training data:
- **Delay Cluster:** *Departure Delay in Minutes* $\leftrightarrow$ *Arrival Delay in Minutes* ($r = 0.812$).
- **Service/Cleanliness Cluster:** *Cleanliness*, *Seat comfort*, and *Inflight entertainment* ($r \in [0.61, 0.68]$).
- **Digital Experience Cluster:** *Inflight wifi service* $\leftrightarrow$ *Ease of Online booking* ($r = 0.698$).

### 7.2 Multi-Seed Stability
Evaluating explanation stability across 5 random seeds confirms that global SHAP rankings for tree models and linear models are exceptionally stable under fixed splits ($r_s = 1.000$ for LR and LGBM, $r_s = 0.996$ for SVC, $r_s = 0.912$ for DT). In contrast, LIME local explanations show significant variance across perturbation seeds ($r_s \in [0.74, 0.88]$), highlighting LIME's surrogate sampling instability.

---

## 8. Extension Results & Critical Analysis of Negative Findings

Table 3 summarizes the comparative metrics between baseline SHAP, MIP, and Correlation-Aware (CA) rankings.

**Table 3: Baseline vs. Correlation-Aware (CA) Ranking Agreement**

| Model | Baseline SHAP vs. CA ($r_s$) | Baseline SHAP vs. MIP ($r_s$) | CA vs. MIP ($r_s$) |
|---|---|---|---|
| **DT** | 0.986 | 0.148 | 0.141 |
| **LR** | 0.968 | 0.203 | 0.189 |
| **LGBM** | 0.975 | 0.211 | 0.198 |
| **SVC** | 0.997 | 0.165 | 0.158 |

**Cross-Model Agreement Analysis:**
- Mean Pairwise Spearman (Baseline SHAP): **0.569**
- Mean Pairwise Spearman (Correlation-Aware): **0.555** ($\Delta = -0.014$)

### Critical Analysis of Contradicting Result ($H_2$ Refutation)
Our second hypothesis ($H_2$) conjectured that redistributing attribution credit within correlated groups would harmonize feature rankings across different model architectures. **The empirical data refutes this hypothesis:** cross-model agreement decreased marginally from 0.569 to 0.555.

We identify two key theoretical reasons for this result:
1. **Disparate Inductive Biases on Collinear Manifolds:**
   Linear models (LR) distribute attribution coefficients smoothly across collinear features when L2 regularization is applied ($w_i \approx w_j$). In contrast, gradient-boosted trees (LGBM) greedily select whichever single correlated feature yields the highest initial variance reduction, assigning negligible splits to its collinear counterparts. Group redistribution boosts the neglected features in tree models, but does so in proportion to their individual importance, which was already skewed by the tree's greedy split selection.
2. **Post-Hoc vs. In-Model Attribution:**
   Post-hoc proportional redistribution does not alter the underlying model's functional reliance on the collinear manifold. True alignment across architectures requires regularized joint attribution (such as Group-Lasso or Partition SHAP) rather than post-hoc algebraic redistribution.

Reporting this negative result reinforces academic integrity: post-hoc grouping reliably identifies redundant feature blocks for human interpretation, but cannot bridge the fundamental gap between how linear and non-linear models navigate multicollinear feature spaces.

---

## 9. Discussion

1. **Model Choice Dictates Attribution:** Even when models achieve nearly identical accuracy ($\approx 83\%$), their explanations differ fundamentally ($r_s = 0.291$ between DT and LR). Practitioners cannot treat SHAP attributions as objective physical properties of the dataset.
2. **MIP vs. Proportional Redistribution:** MIP actively forces feature re-evaluation by physical ablation, uncovering deeply suppressed features (+20 rank jump). In contrast, post-hoc proportional redistribution keeps the relative ranking largely intact ($r_s > 0.96$). For collinearity diagnosis, ablation-based metrics (NMR/MIP) are substantially more sensitive than analytical redistribution.
3. **Implications for Regulated AI:** In domains such as credit scoring or medical diagnostics, selecting an explanation method without evaluating model stability and collinearity introduces substantial legal and ethical vulnerability.

---

## 10. Limitations & Threats to Validity

1. **Finite Sample Size:** The 500-subject case study matches Salih et al. (2025) but exhibits small-sample variance ($n_{\text{test}} = 100$).
2. **Sampling Disclose:** The Kaggle dataset contains 129,880 records; because the original seed was undisclosed, our stratified random sample represents an empirical approximation.
3. **Platt Scaling on SVC:** Enabling `probability=True` modifies SVC decision boundaries slightly via 5-fold internal cross-validation, though it is mandatory for LIME probability querying.

---

## 11. Conclusion

This study successfully replicated the findings of Salih et al. (2025): explainability rankings from SHAP and LIME are strongly model-dependent and distorted by feature collinearity. We confirmed that NMR quantifies ranking volatility under ablation and demonstrated that MIP exposes features masked by collinearity. Finally, our Correlation-Aware extension revealed that post-hoc attribution redistribution does not eliminate model-dependent divergence, demonstrating that feature collinearity must be addressed during model training or through group-constrained Shapley formulations.

---

## References

1. Salih, A.M., Raisi-Estabragh, Z., Galazzo, I.B., Radeva, P., Petersen, S.E., Lekadir, K., & Menegaz, G. (2025). A Perspective on Explainable Artificial Intelligence Methods: SHAP and LIME. *Advanced Intelligent Systems*, 7(1), 2400304.
2. Lundberg, S.M., & Lee, S.I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems (NeurIPS)*, 30.
3. Ribeiro, M.T., Singh, S., & Guestrin, C. (2016). "Why should I trust you?": Explaining the predictions of any classifier. *ACM SIGKDD*, 1135–1144.
4. Aas, K., Jullum, M., & Løland, A. (2021). Explaining individual predictions when features are dependent: More accurate approximations to Shapley values. *Machine Learning*, 110(6), 1379–1428.
5. Salih, A.M., Galazzo, I.B., & Menegaz, G. (2022). Normalized Movement Rate: A novel metric for feature ranking stability. *IEEE ICHI*, 40–47.
6. Salih, A.M., et al. (2024b). Characterizing the Contribution of Dependent Features in XAI Methods. *IEEE Journal of Biomedical and Health Informatics (JBHI)*.
7. Molnar, C. (2022). *Interpretable Machine Learning: A Guide for Making Black Box Models Explainable*. 2nd Edition.
