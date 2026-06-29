import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

IN_FILE="data/results/routed_complaints.csv"
OUT_REPORT="data/results/glm52_evaluation.txt"
OUT_CM="paper/figures/glm52_confusion_matrix.png"

df=pd.read_csv(IN_FILE)

y_true=df["category"]
y_pred=df["glm52_pred"]

acc=accuracy_score(y_true,y_pred)
f1=f1_score(y_true,y_pred,average="weighted")

report=classification_report(y_true,y_pred)

with open(OUT_REPORT, "w", encoding="utf-8") as f:
    f.write(f"GLM-5.2 Accuracy:{acc:.4f}\n")
    f.write(f"GLM-5.2 Weighted F1:{f1:.4f}\n\n")
    f.write(report)

labels=sorted(df["category"].unique())
cm=confusion_matrix(y_true,y_pred,labels=labels)

plt.figure(figsize=(10,8))
plt.imshow(cm)
plt.title("GLM-5.2 Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.xticks(range(len(labels)),labels,rotation=90)
plt.yticks(range(len(labels)),labels)
plt.colorbar()
plt.tight_layout()
plt.savefig(OUT_CM,dpi=300)

print(report)
print(f"Saved confusion matrix: {OUT_CM}")