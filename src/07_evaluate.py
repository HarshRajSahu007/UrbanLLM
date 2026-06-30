import os
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving figures
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

_SRC       = Path(__file__).parent
_DATA      = _SRC.parent / "Data"

IN_FILE    = _DATA / "results" / "routed_complaints.csv"
OUT_REPORT = _DATA / "results" / "glm52_evaluation.txt"
OUT_CM     = _DATA / "results" / "glm52_confusion_matrix.png"

os.makedirs(str(OUT_REPORT.parent), exist_ok=True)

df = pd.read_csv(IN_FILE)

# Both ground-truth category and GLM-predicted label must be present
df = df.dropna(subset=["category", "glm52_pred"])

y_true = df["category"]
y_pred = df["glm52_pred"]

acc    = accuracy_score(y_true, y_pred)
f1     = f1_score(y_true, y_pred, average="weighted", zero_division=0)
report = classification_report(y_true, y_pred, zero_division=0)

with open(str(OUT_REPORT), "w", encoding="utf-8") as f:
    f.write(f"GLM-5.2 Accuracy:    {acc:.4f}\n")
    f.write(f"GLM-5.2 Weighted F1: {f1:.4f}\n\n")
    f.write(report)

# ── Confusion matrix ───────────────────────────────────────────────────────
labels = sorted(df["category"].unique())
cm     = confusion_matrix(y_true, y_pred, labels=labels)

fig, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=labels,
    yticklabels=labels,
    ax=ax,
)
ax.set_title("GLM-5.2 Confusion Matrix", fontsize=14, pad=12)
ax.set_xlabel("Predicted Label", fontsize=11)
ax.set_ylabel("True Label",      fontsize=11)
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
fig.savefig(str(OUT_CM), dpi=300)
plt.close(fig)

print(report)
print(f"Accuracy: {acc:.4f}   Weighted F1: {f1:.4f}")
print(f"Report saved:           {OUT_REPORT}")
print(f"Confusion matrix saved: {OUT_CM}")