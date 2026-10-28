# Experimental Results

## 1. Classifier Performance

| Model | Paper Accuracy | Our Accuracy | Difference | Precision | Recall | F1-Score | ROC-AUC |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Decision Tree (DT)** | 0.84 | **0.83** | -0.01 | 0.8367 | 0.8200 | 0.8283 | 0.8300 |
| **Logistic Regression (LR)** | 0.85 | **0.83** | -0.02 | 0.8837 | 0.7600 | 0.8172 | 0.9048 |
| **LightGBM (LGBM)** | 0.91 | **0.95** | +0.04 | 0.9787 | 0.9200 | 0.9485 | 0.9852 |
| **Support Vector Classifier (SVC)** | 0.86 | **0.88** | +0.02 | 0.9524 | 0.8000 | 0.8696 | 0.9660 |

Full results: [`results/tables/classifier_performance.csv`](file:///Users/aditya/Developer/Xai%20project/results/tables/classifier_performance.csv)

---

## 2. SHAP Feature Rankings & Agreement

### Top 5 Feature Rankings Across Models
* **DT:** Online boarding, Inflight wifi service, Type of Travel, Age, Flight Distance
* **LR:** Type of Travel, Online boarding, Cleanliness, Checkin service, Leg room service
* **LGBM:** Online boarding, Type of Travel, Inflight wifi service, Flight Distance, Age
* **SVC:** Online boarding, Inflight wifi service, Type of Travel, Cleanliness, Leg room service

### Pairwise Ranking Agreement (Spearman Rank Correlation)
* **DT ↔ LGBM:** 0.694 (Kendall Tau: 0.532, Jaccard@3: 1.0, Jaccard@5: 1.0)
* **LGBM ↔ SVC:** 0.700 (Kendall Tau: 0.532, Jaccard@3: 1.0, Jaccard@5: 0.429)
* **LR ↔ LGBM:** 0.659 (Kendall Tau: 0.498, Jaccard@3: 0.5, Jaccard@5: 0.250)
* **LR ↔ SVC:** 0.580 (Kendall Tau: 0.429, Jaccard@3: 0.5, Jaccard@5: 0.667)
* **DT ↔ SVC:** 0.493 (Kendall Tau: 0.359, Jaccard@3: 1.0, Jaccard@5: 0.429)
* **DT ↔ LR:** 0.291 (Kendall Tau: 0.203, Jaccard@3: 0.5, Jaccard@5: 0.250)

Full results: [`results/tables/shap_ranking_comparison.csv`](file:///Users/aditya/Developer/Xai%20project/results/tables/shap_ranking_comparison.csv) and [`results/tables/pairwise_ranking_agreement.csv`](file:///Users/aditya/Developer/Xai%20project/results/tables/pairwise_ranking_agreement.csv)

---

## 3. NMR (Normalized Movement Rate) Comparison

| Model | Paper NMR | Our NMR | Difference |
|:---|:---:|:---:|:---:|
| **Decision Tree (DT)** | 0.445 | **0.422** | 0.023 |
| **Logistic Regression (LR)** | 0.275 | **0.324** | 0.049 |
| **LightGBM (LGBM)** | 0.231 | **0.360** | 0.129 |
| **Support Vector Classifier (SVC)** | 0.273 | **0.342** | 0.069 |

Full results: [`results/tables/nmr_comparison.csv`](file:///Users/aditya/Developer/Xai%20project/results/tables/nmr_comparison.csv)

---

## 4. Modified Index Position (MIP) vs SHAP

Applied to the model with lowest NMR (Logistic Regression):
* Collinear features such as **Cleanliness** and **Food and drink** drop in rank, redistributing credit appropriately to independent predictors.
* Feature rank changes confirm the paper's thesis regarding collinearity distortion in raw SHAP rankings.

Full results: [`results/tables/shap_vs_mip.csv`](file:///Users/aditya/Developer/Xai%20project/results/tables/shap_vs_mip.csv)

---

## 5. Research Extension: Correlation-Aware Robustness

* **Multi-Seed Stability (Mean Spearman):**
  * Decision Tree (DT): **0.912**
  * Logistic Regression (LR): **1.000**
  * LightGBM (LGBM): **1.000**
  * Support Vector Classifier (SVC): **0.996**
* **Method Agreement (Baseline vs Correlation-Aware SHAP):**
  * DT: **0.986**
  * LR: **0.968**
  * LGBM: **0.975**
  * SVC: **0.997**
* **Cross-Model Agreement:** Baseline SHAP mean Spearman: **0.569**, Correlation-Aware: **0.555**.

---

## 6. Generated Figures

| Figure | Path | Description |
|:---|:---|:---|
| **Classifier Performance** | [`results/figures/classifier_performance.png`](file:///Users/aditya/Developer/Xai%20project/results/figures/classifier_performance.png) | 4-model accuracy and metric comparison |
| **Correlation Heatmap** | [`results/figures/correlation_heatmap.png`](file:///Users/aditya/Developer/Xai%20project/results/figures/correlation_heatmap.png) | High-correlation feature pairs (|r| > 0.5) |
| **SHAP Global Importance** | [`results/figures/shap_global_importance.png`](file:///Users/aditya/Developer/Xai%20project/results/figures/shap_global_importance.png) | Mean |SHAP| feature rankings |
| **Ranking Comparison** | [`results/figures/ranking_comparison.png`](file:///Users/aditya/Developer/Xai%20project/results/figures/ranking_comparison.png) | Bump chart comparing feature ranks across models |
| **LIME Comparison** | [`results/figures/lime_comparison_instance0.png`](file:///Users/aditya/Developer/Xai%20project/results/figures/lime_comparison_instance0.png) | Instance 0 explanation across all 4 classifiers |
| **NMR Comparison** | [`results/figures/nmr_comparison.png`](file:///Users/aditya/Developer/Xai%20project/results/figures/nmr_comparison.png) | Replicated NMR values vs Salih et al. (2024) |
| **SHAP vs MIP** | [`results/figures/shap_vs_mip.png`](file:///Users/aditya/Developer/Xai%20project/results/figures/shap_vs_mip.png) | Shift in ranking when accounting for collinearity |
| **Extension Stability** | [`results/figures/extension_stability.png`](file:///Users/aditya/Developer/Xai%20project/results/figures/extension_stability.png) | Robustness analysis across seeds and methods |
