import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV
from sklearn.metrics import (classification_report, accuracy_score,
                             precision_score, recall_score, f1_score,
                             confusion_matrix, ConfusionMatrixDisplay,
                             roc_curve, auc)
from sklearn.preprocessing import label_binarize, MinMaxScaler

# STEP 1 : LOAD & DISPLAY RAW DATA

df_raw = pd.read_csv(r"C:\Users\ACER\Downloads\UpdatedResumeDataSet_raw.csv")

print("=" * 60)
print("INITIAL DATA INSPECTION")
print("=" * 60)
print(f"Dataset Name   : UpdatedResumeDataSet.csv")
print(f"Dataset Source : Kaggle — Resume Dataset")
print(f"Total Records  : {df_raw.shape[0]} rows")
print(f"Total Features : {df_raw.shape[1]} columns")
print(f"Feature Names  : {list(df_raw.columns)}\n")
print("--- Head (first 5 rows) ---")
print(df_raw.head().to_string())
print("\nMissing Values (Raw):")
print(df_raw.isnull().sum())
print("\nData Types (Raw):")
print(df_raw.dtypes)


# STEP 2 : DATA CLEANING

print("\n" + "=" * 60)
print("DATA CLEANING STEPS")
print("=" * 60)

df = df_raw.copy()

before = len(df)
df = df[df["Resume"].notna()].copy()
after = len(df)
print(f"[1] Removed rows with missing 'Resume'   : {before - after} rows removed  ({before} -> {after})")

before = len(df)
df = df[df["Category"].notna()].copy()
after = len(df)
print(f"[2] Removed rows with missing 'Category' : {before - after} rows removed  ({before} -> {after})")

df["Category"] = df["Category"].str.strip()
print("[3] Stripped whitespace from 'Category'")

df["Resume"] = (df["Resume"].astype(str)
                .str.lower()
                .str.replace(r"[^a-zA-Z0-9\s]", " ", regex=True)
                .str.replace(r"\s+", " ", regex=True)
                .str.strip())
print("[4] Normalized 'Resume' : lowercase + removed special characters + extra spaces")

before = len(df)
counts = df["Category"].value_counts()
df = df[df["Category"].isin(counts[counts >= 2].index)].copy()
after = len(df)
print(f"[5] Removed rare categories (< 2 occurrences) : {before - after} rows removed  ({before} -> {after})")

df = df.reset_index(drop=True)
print("[6] Reset index")

print(f"\nCleaned Dataset Shape : {df.shape[0]} rows x {df.shape[1]} columns")
print(f"Unique Categories     : {df['Category'].nunique()}")
print("\nCategory Distribution:")
print(df["Category"].value_counts().to_string())

# STEP 3 : DISPLAY CLEAN DATA

print("\n" + "=" * 60)
print("CLEAN DATA OVERVIEW")
print("=" * 60)
print(f"Records : {df.shape[0]}  |  Features : {df.shape[1]}\n")
print(df.head(10).to_string())
print("\nMissing Values (Clean):")
print(df.isnull().sum())
print("\nData Types (Clean):")
print(df.dtypes)


# VISUALIZATION 1 — Category Distribution Bar Chart

cat_counts = df["Category"].value_counts()
colors = cm.tab20(np.linspace(0, 1, len(cat_counts)))

plt.figure(figsize=(14, 6))
bars = plt.bar(cat_counts.index, cat_counts.values, color=colors, edgecolor="black")
for bar, val in zip(bars, cat_counts.values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
             str(val), ha="center", va="bottom", fontsize=8)
plt.xticks(rotation=45, ha="right", fontsize=8)
plt.xlabel("Job Category")
plt.ylabel("Number of Resumes")
plt.title("Job Category Distribution")
plt.tight_layout()
plt.show()


# VISUALIZATION 2 — Top 20 Most Common Skills

all_words = " ".join(df["Resume"]).split()
# Remove common stop words
stop_words = {"and", "the", "of", "in", "to", "a", "with", "for", "is",
              "on", "at", "by", "an", "as", "or", "be", "from", "this",
              "are", "it", "that", "have", "has", "was", "were", "will",
              "can", "knowledge", "experience", "skills", "skill", "working",
              "work", "ability", "good", "team", "management", "details",
              "education", "january", "february", "march", "april", "may",
              "june", "july", "august", "september", "october", "november",
              "december", "2016", "2017", "2018", "2019", "2020", "2015",
              "2014", "2013", "2012", "ms", "basic", "computer", "using"}
filtered = [w for w in all_words if w not in stop_words and len(w) > 2]
skill_counts = Counter(filtered).most_common(20)
skill_names  = [s[0] for s in skill_counts]
skill_values = [s[1] for s in skill_counts]

plt.figure(figsize=(12, 6))
bars2 = plt.bar(skill_names, skill_values, color="steelblue", edgecolor="black")
for bar, val in zip(bars2, skill_values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             str(val), ha="center", va="bottom", fontsize=8)
plt.xticks(rotation=45, ha="right")
plt.xlabel("Skill / Keyword")
plt.ylabel("Frequency")
plt.title("Top 20 Most Common Skills Across All Resumes")
plt.tight_layout()
plt.show()


# VISUALIZATION 3 — Top 10 Skills per Category (Top 5 Categories)

top5_cats = df["Category"].value_counts().head(5).index.tolist()

fig, axes = plt.subplots(1, 5, figsize=(22, 6))
for ax, cat in zip(axes, top5_cats):
    cat_words = " ".join(df[df["Category"] == cat]["Resume"]).split()
    cat_filtered = [w for w in cat_words if w not in stop_words and len(w) > 2]
    top_skills = Counter(cat_filtered).most_common(10)
    names  = [s[0] for s in top_skills][::-1]
    values = [s[1] for s in top_skills][::-1]
    ax.barh(names, values, color="coral", edgecolor="black")
    ax.set_title(cat, fontsize=8, fontweight="bold")
    ax.set_xlabel("Frequency")
plt.suptitle("Top 10 Skills for Top 5 Job Categories", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.show()


# VISUALIZATION 4 — Missing Values Heatmap

missing = df_raw.isnull().astype(int)
plt.figure(figsize=(8, 4))
plt.imshow(missing.T, aspect="auto", cmap="Reds", interpolation="none")
plt.colorbar(label="Missing (1=Yes, 0=No)")
plt.yticks(range(len(df_raw.columns)), df_raw.columns)
plt.xlabel("Row Index")
plt.title("Missing Values Heatmap (Raw Data Before Cleaning)")
plt.tight_layout()
plt.show()


# STEP 4 : FEATURE EXTRACTION — TF-IDF

print("\n" + "=" * 60)
print("FEATURE EXTRACTION — TF-IDF VECTORIZER")
print("=" * 60)

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True,
    max_features=3000
)

X = vectorizer.fit_transform(df["Resume"])
y = df["Category"]

print(f"Vocabulary Size      : {len(vectorizer.vocabulary_)} unique terms")
print(f"Feature Matrix Shape : {X.shape}")
print(f"Total Classes        : {y.nunique()}")

scaler = MinMaxScaler()
X = scaler.fit_transform(X.toarray())
classes = sorted(y.unique())


# STEP 5 : DATASET SPLIT
print("\n" + "=" * 60)
print("DATASET SPLIT SUMMARY")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print(f"  Total Samples         : {X.shape[0]}")
print(f"  Training Set          : {X_train.shape[0]} samples (80%)")
print(f"  Test Set              : {X_test.shape[0]} samples (20%)")
print(f"  Validation            : 5-Fold Stratified Cross-Validation")
print(f"  Each Fold             : ~{X_train.shape[0] // 5} samples per fold")

# STEP 6 : HYPERPARAMETER TUNING
print("\n" + "=" * 60)
print("HYPERPARAMETER TUNING — GridSearchCV")
print("=" * 60)

param_grid = {"alpha": [0.01, 0.1, 0.5, 1.0, 2.0, 5.0]}
grid_search = GridSearchCV(
    MultinomialNB(), param_grid,
    cv=5, scoring="f1_weighted", n_jobs=-1
)
grid_search.fit(X_train, y_train)

best_alpha = grid_search.best_params_["alpha"]
print(f"  Best Alpha Found  : {best_alpha}")
print(f"  Best CV F1-Score  : {grid_search.best_score_:.4f}")
print(f"\nAll Tuning Results:")
results_df = pd.DataFrame(grid_search.cv_results_)
for _, row in results_df[["param_alpha", "mean_test_score", "std_test_score"]].iterrows():
    print(f"  alpha={row['param_alpha']:.2f}  →  F1={row['mean_test_score']:.4f} ± {row['std_test_score']:.4f}")

print(f"\nFinal Model Hyperparameters:")
print(f"  Alpha (Laplace Smoothing) : {best_alpha}")
print(f"  Fit Prior                 : True")
print(f"  Number of CV Folds        : 5")
print(f"  Test Size                 : 20%")
print(f"  Random State              : 42")

# STEP 7 : CROSS-VALIDATION
print("\n" + "=" * 60)
print("CROSS-VALIDATION PERFORMANCE")
print("=" * 60)

model = MultinomialNB(alpha=best_alpha)
skf   = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
all_true_cv, all_pred_cv = [], []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
    X_tr, X_val = X_train[train_idx], X_train[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
    model.fit(X_tr, y_tr)
    y_val_pred = model.predict(X_val)
    fold_acc = accuracy_score(y_val, y_val_pred)
    print(f"  Fold {fold} Accuracy : {fold_acc:.4f}  ({fold_acc*100:.2f}%)")
    all_true_cv.extend(y_val)
    all_pred_cv.extend(y_val_pred)

print("\nCross-Validation Classification Report:\n")
print(classification_report(all_true_cv, all_pred_cv, zero_division=0))

cv_acc = accuracy_score(all_true_cv, all_pred_cv)
cv_pre = precision_score(all_true_cv, all_pred_cv, average="weighted", zero_division=0)
cv_rec = recall_score(all_true_cv, all_pred_cv, average="weighted", zero_division=0)
cv_f1  = f1_score(all_true_cv, all_pred_cv, average="weighted", zero_division=0)

print("Cross-Validation Metrics (Macro Average):")
print(f"  Accuracy  : {accuracy_score(all_true_cv, all_pred_cv):.4f}  ({accuracy_score(all_true_cv, all_pred_cv)*100:.2f}%)")
print(f"  Precision : {precision_score(all_true_cv, all_pred_cv, average='macro', zero_division=0):.4f}")
print(f"  Recall    : {recall_score(all_true_cv, all_pred_cv, average='macro', zero_division=0):.4f}")
print(f"  F1-Score  : {f1_score(all_true_cv, all_pred_cv, average='macro', zero_division=0):.4f}")

print("\nCross-Validation Metrics (Weighted Average):")
print(f"  Accuracy  : {cv_acc:.4f}  ({cv_acc*100:.2f}%)")
print(f"  Precision : {cv_pre:.4f}  ({cv_pre*100:.2f}%)")
print(f"  Recall    : {cv_rec:.4f}  ({cv_rec*100:.2f}%)")
print(f"  F1-Score  : {cv_f1:.4f}  ({cv_f1*100:.2f}%)")

# STEP 8 : FINAL TEST SET EVALUATION
print("\n" + "=" * 60)
print("FINAL TEST SET PERFORMANCE")
print("=" * 60)

model.fit(X_train, y_train)
y_test_pred = model.predict(X_test)

print(classification_report(y_test, y_test_pred, zero_division=0))

test_acc = accuracy_score(y_test, y_test_pred)
test_pre = precision_score(y_test, y_test_pred, average="weighted", zero_division=0)
test_rec = recall_score(y_test, y_test_pred, average="weighted", zero_division=0)
test_f1  = f1_score(y_test, y_test_pred, average="weighted", zero_division=0)

print("Final Test Metrics (Macro Average):")
print(f"  Accuracy  : {accuracy_score(y_test, y_test_pred):.4f}  ({accuracy_score(y_test, y_test_pred)*100:.2f}%)")
print(f"  Precision : {precision_score(y_test, y_test_pred, average='macro', zero_division=0):.4f}")
print(f"  Recall    : {recall_score(y_test, y_test_pred, average='macro', zero_division=0):.4f}")
print(f"  F1-Score  : {f1_score(y_test, y_test_pred, average='macro', zero_division=0):.4f}")

print("\nFinal Test Metrics (Weighted Average):")
print(f"  Accuracy  : {test_acc:.4f}  ({test_acc*100:.2f}%)")
print(f"  Precision : {test_pre:.4f}  ({test_pre*100:.2f}%)")
print(f"  Recall    : {test_rec:.4f}  ({test_rec*100:.2f}%)")
print(f"  F1-Score  : {test_f1:.4f}  ({test_f1*100:.2f}%)")

# VISUALIZATION 5 — Metrics Comparison (CV vs Test)
metric_names = ["Accuracy", "Precision", "Recall", "F1-Score"]
cv_scores    = [cv_acc, cv_pre, cv_rec, cv_f1]
test_scores  = [test_acc, test_pre, test_rec, test_f1]

x = np.arange(len(metric_names))
width = 0.35

plt.figure(figsize=(10, 6))
bars_cv   = plt.bar(x - width/2, cv_scores,   width, label="Cross-Validation", color="steelblue", edgecolor="black")
bars_test = plt.bar(x + width/2, test_scores, width, label="Test Set",          color="coral",     edgecolor="black")
plt.xticks(x, metric_names)
plt.ylabel("Score")
plt.ylim(0, 1.15)
plt.title("Model Performance : Cross-Validation vs Test Set")
plt.legend()
for i, (cv, ts) in enumerate(zip(cv_scores, test_scores)):
    plt.text(i - width/2, cv + 0.02, f"{cv*100:.1f}%", ha="center", fontsize=9, fontweight="bold")
    plt.text(i + width/2, ts + 0.02, f"{ts*100:.1f}%", ha="center", fontsize=9, fontweight="bold")
plt.tight_layout()
plt.show()

# VISUALIZATION 6 — Confusion Matrix
cm_matrix = confusion_matrix(y_test, y_test_pred, labels=model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm_matrix, display_labels=model.classes_)
fig, ax = plt.subplots(figsize=(16, 12))
disp.plot(cmap="Blues", ax=ax, xticks_rotation=45)
plt.title("Confusion Matrix on Test Set")
plt.tight_layout()
plt.grid(False)
plt.show()

# VISUALIZATION 7 — ROC Curve (One-vs-Rest)
y_test_bin = label_binarize(y_test, classes=classes)
y_score    = model.predict_proba(X_test)

fpr, tpr, roc_auc = {}, {}, {}
for i, cls in enumerate(classes):
    fpr[cls], tpr[cls], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc[cls] = auc(fpr[cls], tpr[cls])

all_fpr  = np.unique(np.concatenate([fpr[cls] for cls in classes]))
mean_tpr = np.zeros_like(all_fpr)
for cls in classes:
    mean_tpr += np.interp(all_fpr, fpr[cls], tpr[cls])
mean_tpr /= len(classes)
macro_auc = auc(all_fpr, mean_tpr)

plt.figure(figsize=(14, 8))
colors_roc = cm.tab20(np.linspace(0, 1, len(classes)))
for cls, col in zip(classes, colors_roc):
    plt.plot(fpr[cls], tpr[cls], lw=1.2, color=col, label=f"{cls} (AUC={roc_auc[cls]:.2f})")
plt.plot(all_fpr, mean_tpr, color="black", lw=2.5, linestyle="--",
         label=f"Macro-Average (AUC={macro_auc:.2f})")
plt.plot([0, 1], [0, 1], color="gray", linestyle=":", lw=1.5, label="Random Baseline")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (One-vs-Rest) — All Job Categories")
plt.legend(loc="lower right", fontsize=7, ncol=2)
plt.tight_layout()
plt.show()

# VISUALIZATION 8 — AUC Bar Chart per Category
auc_scores = [roc_auc[cls] for cls in classes]
bar_colors_auc = ["green" if s >= 0.7 else "orange" if s >= 0.5 else "red" for s in auc_scores]

plt.figure(figsize=(12, 10))
bars_auc = plt.barh(classes, auc_scores, color=bar_colors_auc, edgecolor="black")
for bar, score in zip(bars_auc, auc_scores):
    plt.text(score + 0.005, bar.get_y() + bar.get_height()/2,
             f"{score:.2f}", va="center", fontsize=8)
plt.axvline(x=macro_auc, color="red", linestyle="--", lw=2,
            label=f"Macro-Average AUC = {macro_auc:.2f}")
plt.axvline(x=0.5, color="gray", linestyle=":", lw=1.5, label="Random Baseline (0.5)")
plt.xlabel("AUC Score")
plt.title("AUC Score per Job Category (Green ≥ 0.7 | Orange ≥ 0.5 | Red < 0.5)")
plt.xlim(0, 1.2)
plt.legend()
plt.tight_layout()
plt.show()

# VISUALIZATION 9 — Feature Importance (Top Skills per Category)
feature_names = vectorizer.get_feature_names_out()
top_n = 10
top6_cats = classes[:6]

fig, axes = plt.subplots(2, 3, figsize=(20, 12))
axes = axes.flatten()
feat_colors = ["steelblue", "coral", "mediumseagreen", "orange", "mediumpurple", "gold"]

for ax, cls, col in zip(axes, top6_cats, feat_colors):
    cls_idx     = list(model.classes_).index(cls)
    log_probs   = model.feature_log_prob_[cls_idx]
    top_indices = log_probs.argsort()[-top_n:][::-1]
    top_feats   = [feature_names[i] for i in top_indices]
    top_vals    = [log_probs[i] for i in top_indices]
    ax.barh(top_feats[::-1], top_vals[::-1], color=col, edgecolor="black")
    ax.set_title(f"Top Skills — {cls}", fontsize=9, fontweight="bold")
    ax.set_xlabel("Log Probability")

plt.suptitle("Feature Importance : Top 10 Skills per Job Category", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.show()

# VISUALIZATION 10 — Prediction Probability (CLI)
print("\n" + "=" * 60)
print("JOB CATEGORY PREDICTION — CLI")
print("=" * 60)

user_input  = input("\nEnter your skills (e.g. Python Machine Learning SQL): ")
user_skills = user_input.lower()

user_vec = vectorizer.transform([user_skills])
user_vec = scaler.transform(user_vec.toarray())
probs    = model.predict_proba(user_vec)[0]
top3_idx = probs.argsort()[-3:][::-1]

print("\nTop 3 Job Category Predictions:")
print("-" * 50)
for rank, i in enumerate(top3_idx, 1):
    print(f"  #{rank}  {model.classes_[i]:<35} (probability = {probs[i]:.4f})")
print("-" * 50)
print(f"\nBest Match : {model.predict(user_vec)[0]}")

top3_labels = [model.classes_[i] for i in top3_idx]
top3_probs  = [probs[i] for i in top3_idx]

plt.figure(figsize=(9, 5))
bars_pred = plt.bar(top3_labels, top3_probs,
                    color=["gold", "silver", "peru"], edgecolor="black")
for bar, prob in zip(bars_pred, top3_probs):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
             f"{prob:.4f}", ha="center", fontsize=11, fontweight="bold")
plt.ylabel("Probability")
plt.xticks(rotation=15, ha="right")
plt.title(f"Top 3 Predicted Job Categories\nInput: {user_input}")
plt.ylim(0, max(top3_probs) + 0.1)
plt.tight_layout()
plt.show() 