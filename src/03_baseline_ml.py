import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score
importjoblib

DATA_FILE="data/processed/complaints_clean.csv"
RESULT_FILE="data/results/baseline_results.txt"

df=pd.read_csv(DATA_FILE)

X=df["text"]
y=df["category"]

X_train,X_test,y_train,y_test=train_test_split(
X,y,test_size=0.2,random_state=42,stratify=y
)

model=Pipeline([
    ("tfidf",TfidfVectorizer(max_features=10000,ngram_range=(1,2))),
    ("clf",LogisticRegression(max_iter=1000))
])

model.fit(X_train,y_train)
pred=model.predict(X_test)

acc=accuracy_score(y_test,pred)
f1=f1_score(y_test,pred,average="weighted")

report=classification_report(y_test,pred)

with open(RESULT_FILE, "w", encoding="utf-8") as f:
    f.write(f"Accuracy:{acc:.4f}\n")
    f.write(f"Weighted F1:{f1:.4f}\n\n")
    f.write(report)

joblib.dump(model,"data/results/tfidf_logreg_model.pkl")

print(report)
print(f"Accuracy:{acc:.4f}, F1:{f1:.4f}")