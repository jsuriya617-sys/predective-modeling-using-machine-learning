import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report,
                             confusion_matrix, roc_curve, auc)

# 1. LOAD DATA
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# 2. SPLIT & SCALE
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# 3. TRAIN MODELS
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree"      : DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest"      : RandomForestClassifier(n_estimators=100, random_state=42),
}

results = {}
for name, model in models.items():
    model.fit(X_train_sc, y_train)
    y_pred = model.predict(X_test_sc)
    y_prob = model.predict_proba(X_test_sc)[:, 1]
    results[name] = {"model": model, "y_pred": y_pred, "y_prob": y_prob,
                     "acc": accuracy_score(y_test, y_pred)}
    print(f"\n{'='*45}\n{name}")
    print(f"Accuracy : {results[name]['acc']*100:.2f}%")
    print(classification_report(y_test, y_pred, target_names=data.target_names))

# 4. CONFUSION MATRICES
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Confusion Matrices", fontsize=16, fontweight='bold')
for ax, (name, res) in zip(axes, results.items()):
    cm = confusion_matrix(y_test, res["y_pred"])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=data.target_names, yticklabels=data.target_names)
    ax.set_title(f"{name}\nAcc: {res['acc']*100:.2f}%")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrices.png", dpi=150, bbox_inches='tight')
plt.show()

# 5. ROC CURVES
plt.figure(figsize=(8, 6))
colors = ['royalblue', 'seagreen', 'tomato']
for (name, res), color in zip(results.items(), colors):
    fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
    plt.plot(fpr, tpr, color=color, lw=2,
             label=f"{name} (AUC={auc(fpr,tpr):.3f})")
plt.plot([0,1],[0,1],'k--', lw=1.5, label='Random Guess')
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("ROC Curves", fontsize=14, fontweight='bold')
plt.legend(loc="lower right"); plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("roc_curves.png", dpi=150, bbox_inches='tight')
plt.show()

# 6. METRICS COMPARISON
metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
names   = list(results.keys())
x       = np.arange(len(metrics)); width = 0.25
colors2 = ['royalblue', 'seagreen', 'tomato']
fig, ax = plt.subplots(figsize=(11, 6))
for i, (name, color) in enumerate(zip(names, colors2)):
    y_pred = results[name]["y_pred"]
    vals = [accuracy_score(y_test, y_pred)*100,
            precision_score(y_test, y_pred)*100,
            recall_score(y_test, y_pred)*100,
            f1_score(y_test, y_pred)*100]
    bars = ax.bar(x + i*width, vals, width, label=name,
                  color=color, alpha=0.85, edgecolor='black')
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                f"{val:.1f}%", ha='center', va='bottom', fontsize=8, fontweight='bold')
ax.set_xticks(x+width); ax.set_xticklabels(metrics, fontsize=12)
ax.set_ylim(80, 105); ax.set_ylabel("Score (%)")
ax.set_title("Model Performance Comparison", fontsize=14, fontweight='bold')
ax.legend(); ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig("metrics_comparison.png", dpi=150, bbox_inches='tight')
plt.show()

# 7. FEATURE IMPORTANCE
rf   = results["Random Forest"]["model"]
feat = pd.Series(rf.feature_importances_,
                 index=data.feature_names).sort_values(ascending=False).head(10)
plt.figure(figsize=(10, 5))
sns.barplot(x=feat.values*100, y=feat.index, palette="viridis")
plt.xlabel("Importance (%)")
plt.title("Top 10 Feature Importances — Random Forest", fontweight='bold')
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150, bbox_inches='tight')
plt.show()

print("\n✅ Done! Saved:")
for f in ["confusion_matrices.png","roc_curves.png","metrics_comparison.png","feature_importance.png"]:
    print(f"   {f}")
