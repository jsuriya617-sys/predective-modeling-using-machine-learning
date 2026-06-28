# ============================================================
# Predictive Modeling Using Machine Learning
# File: predictive_modeling.py
# Algorithms: Logistic Regression, Decision Tree, Random Forest
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc
)

# ─────────────────────────────────────────
# 1. LOAD & EXPLORE DATASET
# ─────────────────────────────────────────
print("=" * 50)
print("  PREDICTIVE MODELING USING MACHINE LEARNING")
print("=" * 50)

data   = load_breast_cancer()
X      = pd.DataFrame(data.data, columns=data.feature_names)
y      = pd.Series(data.target)

print(f"\n📊 Dataset: Breast Cancer Wisconsin")
print(f"   Samples  : {X.shape[0]}")
print(f"   Features : {X.shape[1]}")
print(f"   Classes  : {list(data.target_names)}")
print(f"   Class Distribution:\n{y.value_counts().to_string()}")

# ─────────────────────────────────────────
# 2. PREPROCESSING
# ─────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler       = StandardScaler()
X_train_sc   = scaler.fit_transform(X_train)
X_test_sc    = scaler.transform(X_test)

print(f"\n🔀 Train size : {X_train.shape[0]} samples")
print(f"   Test size  : {X_test.shape[0]}  samples")

# ─────────────────────────────────────────
# 3. DEFINE MODELS
# ─────────────────────────────────────────
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree"      : DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest"      : RandomForestClassifier(n_estimators=100, random_state=42),
}

# ─────────────────────────────────────────
# 4. TRAIN & EVALUATE
# ─────────────────────────────────────────
results = {}
print("\n" + "=" * 50)
print("  MODEL TRAINING & EVALUATION")
print("=" * 50)

for name, model in models.items():
    model.fit(X_train_sc, y_train)
    y_pred = model.predict(X_test_sc)
    y_prob = model.predict_proba(X_test_sc)[:, 1]
    cv     = cross_val_score(model, X_train_sc, y_train, cv=5)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)

    results[name] = {
        "model" : model,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "acc"   : round(acc  * 100, 2),
        "prec"  : round(prec * 100, 2),
        "rec"   : round(rec  * 100, 2),
        "f1"    : round(f1   * 100, 2),
        "cv_mean": round(cv.mean() * 100, 2),
        "cv_std" : round(cv.std()  * 100, 2),
    }

    print(f"\n🤖 {name}")
    print(f"   Accuracy  : {acc*100:.2f}%")
    print(f"   Precision : {prec*100:.2f}%")
    print(f"   Recall    : {rec*100:.2f}%")
    print(f"   F1 Score  : {f1*100:.2f}%")
    print(f"   CV Score  : {cv.mean()*100:.2f}% ± {cv.std()*100:.2f}%")
    print(f"\n   Classification Report:")
    print(classification_report(y_test, y_pred, target_names=data.target_names))

# ─────────────────────────────────────────
# 5. CONFUSION MATRICES
# ─────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Confusion Matrices", fontsize=16, fontweight='bold', y=1.02)

for ax, (name, res) in zip(axes, results.items()):
    cm = confusion_matrix(y_test, res["y_pred"])
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues', ax=ax,
        xticklabels=data.target_names,
        yticklabels=data.target_names,
        linewidths=0.5
    )
    ax.set_title(f"{name}\nAccuracy: {res['acc']}%", fontsize=11)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")

plt.tight_layout()
plt.savefig("confusion_matrices.png", dpi=150, bbox_inches='tight')
plt.show()
print("\n✅ Saved: confusion_matrices.png")

# ─────────────────────────────────────────
# 6. ROC CURVES
# ─────────────────────────────────────────
plt.figure(figsize=(8, 6))
colors = ['royalblue', 'seagreen', 'tomato']

for (name, res), color in zip(results.items(), colors):
    fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
    roc_auc     = auc(fpr, tpr)
    plt.plot(fpr, tpr, color=color, lw=2,
             label=f"{name} (AUC = {roc_auc:.3f})")

plt.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Guess (AUC = 0.500)')
plt.fill_between([0, 1], [0, 1], alpha=0.05, color='gray')
plt.xlabel("False Positive Rate", fontsize=12)
plt.ylabel("True Positive Rate", fontsize=12)
plt.title("ROC Curves — All Models", fontsize=14, fontweight='bold')
plt.legend(loc="lower right", fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("roc_curves.png", dpi=150, bbox_inches='tight')
plt.show()
print("✅ Saved: roc_curves.png")

# ─────────────────────────────────────────
# 7. METRICS COMPARISON BAR CHART
# ─────────────────────────────────────────
metrics   = ["acc", "prec", "rec", "f1"]
m_labels  = ["Accuracy", "Precision", "Recall", "F1 Score"]
names     = list(results.keys())
x         = np.arange(len(metrics))
width     = 0.25
colors2   = ['royalblue', 'seagreen', 'tomato']

fig, ax = plt.subplots(figsize=(11, 6))

for i, (name, color) in enumerate(zip(names, colors2)):
    vals = [results[name][m] for m in metrics]
    bars = ax.bar(x + i * width, vals, width, label=name, color=color, alpha=0.85, edgecolor='black')
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{val}%", ha='center', va='bottom', fontsize=8, fontweight='bold')

ax.set_xticks(x + width)
ax.set_xticklabels(m_labels, fontsize=12)
ax.set_ylim(80, 105)
ax.set_ylabel("Score (%)", fontsize=12)
ax.set_title("Model Performance Comparison", fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig("metrics_comparison.png", dpi=150, bbox_inches='tight')
plt.show()
print("✅ Saved: metrics_comparison.png")

# ─────────────────────────────────────────
# 8. FEATURE IMPORTANCE (Random Forest)
# ─────────────────────────────────────────
rf_model      = results["Random Forest"]["model"]
importances   = rf_model.feature_importances_
feat_series   = pd.Series(importances, index=data.feature_names).sort_values(ascending=False)
top_features  = feat_series.head(10)

plt.figure(figsize=(10, 5))
sns.barplot(x=top_features.values * 100, y=top_features.index, palette="viridis")
plt.xlabel("Feature Importance (%)", fontsize=12)
plt.title("Top 10 Feature Importances — Random Forest", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150, bbox_inches='tight')
plt.show()
print("✅ Saved: feature_importance.png")

# ─────────────────────────────────────────
# 9. SUMMARY TABLE
# ─────────────────────────────────────────
summary = pd.DataFrame({
    "Model"    : names,
    "Accuracy" : [results[n]["acc"]  for n in names],
    "Precision": [results[n]["prec"] for n in names],
    "Recall"   : [results[n]["rec"]  for n in names],
    "F1 Score" : [results[n]["f1"]   for n in names],
    "CV Mean"  : [results[n]["cv_mean"] for n in names],
}).set_index("Model")

print("\n" + "=" * 50)
print("  FINAL SUMMARY TABLE")
print("=" * 50)
print(summary.to_string())

best = summary["Accuracy"].idxmax()
print(f"\n🏆 Best Model: {best} with Accuracy = {summary.loc[best, 'Accuracy']}%")
print("\n✅ All done! Files saved:")
for f in ["confusion_matrices.png","roc_curves.png","metrics_comparison.png","feature_importance.png"]:
    print(f"   📁 {f}")
