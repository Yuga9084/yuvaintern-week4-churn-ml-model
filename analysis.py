"""
YuvaIntern - Virtual Data Science with Python
Week 4: Machine Learning Model Development and Evaluation
Problem: predict which telecom customers will churn (binary classification)
"""
import os, warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_predict, validation_curve, learning_curve
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score,
                             confusion_matrix, roc_curve, precision_recall_curve, ConfusionMatrixDisplay)
warnings.filterwarnings("ignore")
os.makedirs("figures", exist_ok=True); os.makedirs("data", exist_ok=True)
SEED = 42; np.random.seed(SEED)
sns.set_theme(style="whitegrid"); plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})
BLUE, ORANGE, GREEN, RED, GREY = "#2E86AB", "#F3A712", "#59A96A", "#E4572E", "#9AA0A6"

# ------------------------------------------------ 1. DATA
URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
try:
    raw = pd.read_csv(URL); raw.to_csv("data/telco_churn.csv", index=False)
except Exception:
    raw = pd.read_csv("data/telco_churn.csv")
print("shape", raw.shape, "| missing", raw.isnull().sum().sum(), "| duplicate rows", raw.duplicated().sum(), "| duplicate IDs", raw.customerID.duplicated().sum())
blank = raw.TotalCharges.str.strip() == ""
print("blank TotalCharges:", blank.sum(), "| their tenure:", raw.loc[blank, "tenure"].unique())

# ------------------------------------------------ 2. CLEANING
df = raw.copy()
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].str.strip().replace("", np.nan))
df["TotalCharges"] = df["TotalCharges"].fillna(0)           # tenure 0 = not billed yet
df["Churn"] = (df["Churn"] == "Yes").astype(int)
df = df.drop(columns=["customerID"])
y = df.pop("Churn"); X = df
print("churn rate", round(y.mean(), 4), "| n =", len(y))
num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
cat_cols = [c for c in X.columns if X[c].dtype == "object" or str(X[c].dtype) == "str"]
print("numeric:", num_cols, "| categorical:", len(cat_cols), "| passthrough: SeniorCitizen")

# EDA figure
fig, ax = plt.subplots(1, 3, figsize=(13, 4.2))
cr = pd.concat([X, y], axis=1).groupby("Contract").Churn.mean().mul(100).loc[["Month-to-month", "One year", "Two year"]]
ax[0].bar(cr.index, cr.values, color=[RED, ORANGE, GREEN]); ax[0].set_ylabel("Churn rate (%)"); ax[0].set_ylim(0, 50); ax[0].set_title("Churn by contract type", loc="left", fontweight="bold")
for i, v in enumerate(cr.values): ax[0].text(i, v+.8, f"{v:.0f}%", ha="center", fontweight="bold")
ax[0].tick_params(axis="x", labelsize=9)
for lab, col in [(0, BLUE), (1, RED)]: sns.kdeplot(X.tenure[y == lab], ax=ax[1], color=col, fill=True, alpha=.3, label="Stayed" if lab == 0 else "Churned", cut=0)
ax[1].set_xlabel("Tenure (months)"); ax[1].set_title("New customers churn most", loc="left", fontweight="bold"); ax[1].legend(frameon=False)
for lab, col in [(0, BLUE), (1, RED)]: sns.kdeplot(X.MonthlyCharges[y == lab], ax=ax[2], color=col, fill=True, alpha=.3, label="Stayed" if lab == 0 else "Churned", cut=0)
ax[2].set_xlabel("Monthly charges ($)"); ax[2].set_title("Churners pay more per month", loc="left", fontweight="bold"); ax[2].legend(frameon=False)
plt.tight_layout(); plt.savefig("figures/fig1_eda.png", dpi=160); plt.close()

# ------------------------------------------------ 3. SPLIT + PREPROCESSING
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
print("train", X_tr.shape, "test", X_te.shape, "| churn rate train/test", round(y_tr.mean(), 4), round(y_te.mean(), 4))
pre = ColumnTransformer([("num", StandardScaler(), num_cols),
                         ("cat", OneHotEncoder(drop="if_binary", handle_unknown="ignore"), cat_cols)], remainder="passthrough")
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)

# ------------------------------------------------ 4. MODELS + TUNING (on training data only)
def pipe(model): return Pipeline([("pre", pre), ("clf", model)])
specs = {
    "Logistic Regression": (pipe(LogisticRegression(max_iter=2000, random_state=SEED)), {"clf__C": [0.01, 0.1, 1, 10]}),
    "Decision Tree": (pipe(DecisionTreeClassifier(random_state=SEED)), {"clf__max_depth": [2, 3, 4, 5, 6, 8, 10], "clf__min_samples_leaf": [1, 10, 30, 60]}),
    "Random Forest": (pipe(RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1)), {"clf__max_depth": [4, 8, 12, None], "clf__min_samples_leaf": [1, 5, 15]}),
}
baseline = pipe(DummyClassifier(strategy="most_frequent")).fit(X_tr, y_tr)
best = {}; cv_rows = []
for name, (p, grid) in specs.items():
    gs = GridSearchCV(p, grid, scoring="roc_auc", cv=cv, n_jobs=-1, return_train_score=True).fit(X_tr, y_tr)
    best[name] = gs.best_estimator_
    r = gs.cv_results_; i = gs.best_index_
    print(f"{name}: best params {gs.best_params_} | CV AUC {gs.best_score_:.4f} (+/- {r['std_test_score'][i]:.4f}) | train AUC {r['mean_train_score'][i]:.4f}")
    cv_rows.append((name, gs.best_score_, r["std_test_score"][i], r["mean_train_score"][i], gs.best_params_))
# untuned default tree to show overfitting
dt_def = pipe(DecisionTreeClassifier(random_state=SEED)).fit(X_tr, y_tr)
from sklearn.model_selection import cross_val_score
cv_def = cross_val_score(pipe(DecisionTreeClassifier(random_state=SEED)), X_tr, y_tr, cv=cv, scoring="roc_auc").mean()
print("Untuned tree: depth", dt_def.named_steps["clf"].get_depth(), "leaves", dt_def.named_steps["clf"].get_n_leaves(),
      "| train AUC", round(roc_auc_score(y_tr, dt_def.predict_proba(X_tr)[:, 1]), 4), "| CV AUC", round(cv_def, 4))

# overfitting figure: validation curve for tree depth + learning curve for LR
depths = [1, 2, 3, 4, 5, 6, 8, 10, 14, 20]
tr_s, va_s = validation_curve(pipe(DecisionTreeClassifier(random_state=SEED)), X_tr, y_tr, param_name="clf__max_depth", param_range=depths, cv=cv, scoring="roc_auc", n_jobs=-1)
print("depth, train, cv:", [(d, round(a, 3), round(b, 3)) for d, a, b in zip(depths, tr_s.mean(1), va_s.mean(1))])
sizes, tr_l, va_l = learning_curve(best["Logistic Regression"], X_tr, y_tr, cv=cv, scoring="roc_auc", train_sizes=np.linspace(.1, 1, 8), n_jobs=-1, random_state=SEED)
print("learning curve LR train/cv at full:", round(tr_l[-1].mean(), 4), round(va_l[-1].mean(), 4))
fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
ax[0].plot(depths, tr_s.mean(1), "o-", color=ORANGE, label="Training AUC"); ax[0].plot(depths, va_s.mean(1), "o-", color=BLUE, label="Cross-validation AUC")
ax[0].fill_between(depths, va_s.mean(1)-va_s.std(1), va_s.mean(1)+va_s.std(1), color=BLUE, alpha=.15)
bi = int(np.argmax(va_s.mean(1))); ax[0].axvline(depths[bi], color=GREY, ls="--"); ax[0].text(depths[bi]+.4, .70, f"best depth = {depths[bi]}", color="#555", fontsize=9)
ax[0].text(13, .86, "overfitting zone:\ntraining keeps rising,\nvalidation falls", fontsize=9, color=RED); ax[0].set_xlabel("Tree depth (model complexity)"); ax[0].set_ylabel("ROC AUC")
ax[0].set_title("Decision tree: from underfitting to overfitting", loc="left", fontweight="bold"); ax[0].legend(frameon=False, loc="center right")
ax[1].plot(sizes, tr_l.mean(1), "o-", color=ORANGE, label="Training AUC"); ax[1].plot(sizes, va_l.mean(1), "o-", color=BLUE, label="Cross-validation AUC")
ax[1].set_xlabel("Training set size"); ax[1].set_ylabel("ROC AUC"); ax[1].set_ylim(.78, .9); ax[1].legend(frameon=False)
ax[1].set_title("Logistic regression: small gap, no overfitting", loc="left", fontweight="bold")
plt.tight_layout(); plt.savefig("figures/fig2_overfitting.png", dpi=160); plt.close()

# ------------------------------------------------ 5. TEST-SET EVALUATION (touched once)
def metrics(model, thr=0.5):
    pr = model.predict_proba(X_te)[:, 1]; yh = (pr >= thr).astype(int)
    return dict(Accuracy=accuracy_score(y_te, yh), Precision=precision_score(y_te, yh, zero_division=0), Recall=recall_score(y_te, yh),
                F1=f1_score(y_te, yh), ROC_AUC=roc_auc_score(y_te, pr), PR_AUC=average_precision_score(y_te, pr)), pr
res = {}; probs = {}
res["Baseline (majority)"], probs["Baseline (majority)"] = metrics(baseline)
for n, m in best.items(): res[n], probs[n] = metrics(m)
tab = pd.DataFrame(res).T.round(3); print("\nTEST METRICS (threshold 0.5)\n", tab)
tab.to_csv("data/test_metrics.csv")
for n in ["Logistic Regression", "Decision Tree", "Random Forest"]:
    tn, fp, fn, tp = confusion_matrix(y_te, probs[n] >= .5).ravel(); print(n, "TN FP FN TP", tn, fp, fn, tp)

# confusion matrices fig
fig, ax = plt.subplots(1, 3, figsize=(13, 4.2))
for a, n in zip(ax, ["Logistic Regression", "Decision Tree", "Random Forest"]):
    ConfusionMatrixDisplay.from_predictions(y_te, probs[n] >= .5, display_labels=["Stayed", "Churned"], cmap="Blues", ax=a, colorbar=False, values_format="d")
    a.set_title(f"{n}\nrecall {res[n]['Recall']:.2f} | precision {res[n]['Precision']:.2f}", fontsize=11, fontweight="bold"); a.grid(False)
plt.tight_layout(); plt.savefig("figures/fig3_confusion_matrices.png", dpi=160); plt.close()

# ROC + PR fig
cols = {"Logistic Regression": BLUE, "Decision Tree": ORANGE, "Random Forest": GREEN, "Baseline (majority)": GREY}
fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
for n, c in cols.items():
    fpr, tpr, _ = roc_curve(y_te, probs[n]); ax[0].plot(fpr, tpr, color=c, lw=2.2, label=f"{n} (AUC {res[n]['ROC_AUC']:.3f})")
    if n.startswith("Baseline"): continue
    pr_, rc_, _ = precision_recall_curve(y_te, probs[n]); ax[1].plot(rc_, pr_, color=c, lw=2.2, label=f"{n} (AP {res[n]['PR_AUC']:.3f})")
ax[0].plot([0, 1], [0, 1], ls=":", color="#888"); ax[0].set_xlabel("False positive rate"); ax[0].set_ylabel("True positive rate (recall)"); ax[0].set_title("ROC curves (test set)", loc="left", fontweight="bold"); ax[0].legend(frameon=False, loc="lower right", fontsize=9)
ax[1].axhline(y_te.mean(), ls=":", color="#888", label=f"Baseline (churn rate {y_te.mean():.2f})"); ax[1].set_xlabel("Recall"); ax[1].set_ylabel("Precision"); ax[1].set_title("Precision-recall curves (test set)", loc="left", fontweight="bold"); ax[1].legend(frameon=False, loc="upper right", fontsize=9)
plt.tight_layout(); plt.savefig("figures/fig4_roc_pr.png", dpi=160); plt.close()

# metric bar comparison
mm = tab.drop(columns=["PR_AUC"]).loc[["Baseline (majority)", "Logistic Regression", "Decision Tree", "Random Forest"]]
fig, ax = plt.subplots(figsize=(11, 4.6)); mm.plot(kind="bar", ax=ax, color=[BLUE, GREEN, ORANGE, RED, "#8E5BA6"], width=.8); ax.set_ylim(0, 1.05)
ax.set_xticklabels(mm.index, rotation=0); ax.legend(frameon=False, ncol=5, loc="upper center", bbox_to_anchor=(.5, 1.12)); ax.set_ylabel("Score")
ax.set_title("Accuracy alone is misleading: the do-nothing baseline scores 73% but finds no churners", loc="left", fontweight="bold", y=1.12, fontsize=11)
plt.tight_layout(); plt.savefig("figures/fig5_metric_comparison.png", dpi=160); plt.close()

# ------------------------------------------------ 6. SELECT MODEL, THRESHOLD, BOOTSTRAP CI
champ = max(cv_rows, key=lambda r: r[1])[0]; print("\nCHAMPION by CV AUC:", champ)
cvp = cross_val_predict(best[champ], X_tr, y_tr, cv=cv, method="predict_proba")[:, 1]       # out-of-fold on TRAIN only
FN_COST, FP_COST = 5.0, 1.0     # ILLUSTRATIVE assumption: missing a churner costs 5x a wasted retention offer
ths = np.linspace(.05, .95, 91)
def cost(yt, pr, t): yh = pr >= t; return (FN_COST*((yt == 1) & ~yh).sum() + FP_COST*((yt == 0) & yh).sum())
costs = np.array([cost(y_tr.values, cvp, t) for t in ths]); t_best = float(ths[costs.argmin()])
f1s = np.array([f1_score(y_tr, cvp >= t) for t in ths]); t_f1 = float(ths[f1s.argmax()])
print("cost-optimal threshold (train CV):", round(t_best, 2), "| F1-optimal:", round(t_f1, 2))
m_def, pr_ch = metrics(best[champ], .5); m_thr, _ = metrics(best[champ], t_best)
cost_def, cost_thr, cost_all = cost(y_te.values, pr_ch, .5), cost(y_te.values, pr_ch, t_best), cost(y_te.values, np.zeros(len(y_te)), .5)
cost_none = cost(y_te.values, np.ones(len(y_te)), 0.0)
print("test at 0.5:", {k: round(v, 3) for k, v in m_def.items()}); print("test at thr:", {k: round(v, 3) for k, v in m_thr.items()})
tn, fp, fn, tp = confusion_matrix(y_te, pr_ch >= t_best).ravel(); print("threshold CM TN FP FN TP", tn, fp, fn, tp)
print("test cost: default 0.5 =", cost_def, "| tuned threshold =", cost_thr, "| flag nobody =", cost_all, "| flag everyone =", cost_none)
rng = np.random.default_rng(SEED); aucs = []
for _ in range(1000):
    idx = rng.integers(0, len(y_te), len(y_te)); 
    if y_te.values[idx].min() != y_te.values[idx].max(): aucs.append(roc_auc_score(y_te.values[idx], pr_ch[idx]))
print("bootstrap AUC 95% CI:", np.round(np.percentile(aucs, [2.5, 97.5]), 3))
fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
pt, rt, tt = precision_recall_curve(y_tr, cvp)
ax[0].plot(tt, pt[:-1], color=BLUE, lw=2, label="Precision"); ax[0].plot(tt, rt[:-1], color=RED, lw=2, label="Recall")
ax[0].axvline(.5, color=GREY, ls=":"); ax[0].axvline(t_best, color="#222", ls="--"); ax[0].text(t_best+.01, .08, f"chosen\nthreshold {t_best:.2f}", fontsize=9)
ax[0].set_xlabel("Decision threshold"); ax[0].set_title("The threshold trades precision for recall", loc="left", fontweight="bold"); ax[0].legend(frameon=False)
ax[1].plot(ths, costs / len(y_tr), color=ORANGE, lw=2.2); ax[1].axvline(t_best, color="#222", ls="--"); ax[1].set_xlabel("Decision threshold"); ax[1].set_ylabel("Average cost per customer (cost units)")
ax[1].set_title(f"Cost curve (missed churner = {FN_COST:.0f}x wasted offer)", loc="left", fontweight="bold")
plt.tight_layout(); plt.savefig("figures/fig6_threshold.png", dpi=160); plt.close()

# ------------------------------------------------ 7. INTERPRETATION
lr = best["Logistic Regression"]; names = lr.named_steps["pre"].get_feature_names_out()
coef = pd.Series(lr.named_steps["clf"].coef_[0], index=[n.split("__")[1] for n in names]).sort_values()
top = pd.concat([coef.head(8), coef.tail(8)])
print("\nTop LR coefficients:\n", top.round(3)); print("LR best C:", lr.named_steps["clf"].C)
fig, ax = plt.subplots(figsize=(9, 5.6)); ax.barh(top.index, top.values, color=[GREEN if v < 0 else RED for v in top.values])
ax.axvline(0, color="#333", lw=.8); ax.set_xlabel("Coefficient (log-odds, standardized numeric features)")
ax.set_title("What the logistic regression learned\n(red raises churn risk, green lowers it)", loc="left", fontweight="bold", fontsize=11)
plt.tight_layout(); plt.savefig("figures/fig7_coefficients.png", dpi=160); plt.close()
rf = best["Random Forest"]; rfn = [n.split("__")[1] for n in rf.named_steps["pre"].get_feature_names_out()]
imp = pd.Series(rf.named_steps["clf"].feature_importances_, index=rfn).sort_values(ascending=False).head(8); print("RF importance top8:\n", imp.round(3))

# error analysis on champion at chosen threshold
ev = X_te.copy(); ev["y"] = y_te.values; ev["pred"] = (pr_ch >= t_best).astype(int)
ev["outcome"] = np.select([(ev.y == 1) & (ev.pred == 1), (ev.y == 1) & (ev.pred == 0), (ev.y == 0) & (ev.pred == 1)], ["TP", "FN", "FP"], "TN")
print("\nError analysis\n", ev.groupby("outcome").agg(n=("y", "size"), tenure=("tenure", "mean"), monthly=("MonthlyCharges", "mean")).round(1))
print(pd.crosstab(ev.outcome, ev.Contract, normalize="index").round(2))
fn_df = ev[ev.outcome == "FN"]; print("FN: share month-to-month", round((fn_df.Contract == "Month-to-month").mean(), 2), "| n", len(fn_df))
# overall cv summary table
pd.DataFrame(cv_rows, columns=["model", "cv_auc", "cv_auc_std", "train_auc", "params"]).to_csv("data/cv_results.csv", index=False)
print("\nfigures saved")
