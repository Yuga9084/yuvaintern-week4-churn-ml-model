# Week 4: Machine Learning Model Development and Evaluation (Telecom Churn)

Internship: **Virtual Data Science with Python Apprentice Intern** at YuvaIntern
Intern: **Yugank Charnotia**

## Problem
Binary classification: predict whether a telecom customer will churn (leave), so a retention team can contact the right customers.

## Dataset
IBM Telco Customer Churn (7,043 customers, 21 columns, 26.5% churn), public on GitHub: `IBM/telco-customer-churn-on-icp4d`.
Data-quality fix: `TotalCharges` is text with 11 blanks (all customers with tenure 0), filled with 0.

## Pipeline
1. Clean data, stratified 80/20 train/test split (test set used once, at the end)
2. `ColumnTransformer`: StandardScaler for numeric, OneHotEncoder for categorical (inside the pipeline, so no leakage)
3. Models: majority-class baseline, Logistic Regression, Decision Tree, Random Forest
4. Hyperparameter tuning: GridSearchCV, stratified 5-fold CV, scoring ROC AUC (training data only)
5. Evaluation: accuracy, precision, recall, F1, ROC AUC, PR AUC, confusion matrices, ROC and PR curves
6. Overfitting analysis: validation curve (tree depth) and learning curve
7. Threshold selection with an illustrative cost model (missed churner = 5x wasted offer), chosen on out-of-fold training predictions
8. Interpretation (coefficients, feature importance) and error analysis

## Results (test set, threshold 0.50)
| Model | Accuracy | Precision | Recall | F1 | ROC AUC |
|---|---|---|---|---|---|
| Baseline (majority) | 0.735 | 0.000 | 0.000 | 0.000 | 0.500 |
| Logistic Regression | 0.806 | 0.659 | 0.559 | 0.605 | 0.841 |
| Decision Tree | 0.796 | 0.641 | 0.524 | 0.576 | 0.829 |
| Random Forest | 0.805 | 0.675 | 0.511 | 0.581 | 0.843 |

- An untuned decision tree overfits badly: training AUC 1.00 vs cross-validation AUC 0.65.
- Models are practically tied (bootstrap 95% CI for random forest AUC: 0.82 to 0.86).
- Lowering the threshold to 0.18 raises recall from 51% to 87% and cuts the assumed cost by 35% (the cost ratio is an illustrative assumption).
- Highest-risk customers: month-to-month contracts, short tenure, fiber-optic internet.

## Folder structure
```
analysis.py          full pipeline, evaluation and figures
results_output.txt   printed output of the script
requirements.txt
data/telco_churn.csv
figures/             7 charts
```

## How to run
```bash
pip install -r requirements.txt
python analysis.py
```

## Charts
![fig1](figures/fig1_eda.png)
![fig2](figures/fig2_overfitting.png)
![fig3](figures/fig3_confusion_matrices.png)
![fig4](figures/fig4_roc_pr.png)
![fig5](figures/fig5_metric_comparison.png)
![fig6](figures/fig6_threshold.png)
![fig7](figures/fig7_coefficients.png)

## Tools
Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
