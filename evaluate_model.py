
import pandas as pd
import joblib
import re

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt


# 1. Dataset load karna
df = pd.read_csv("SpamSmsDataset.xlsx - Sheet1.csv")

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)
print("Columns:", df.columns.tolist())


# 2. Missing values remove karna
df = df.dropna(subset=["message", "label"])

# Labels ko standard format mein lana
df["label"] = df["label"].astype(str).str.strip().str.lower()

print("\nLabel counts:")
print(df["label"].value_counts())


# 3. Text cleaning
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " URL ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


X = df["message"].astype(str).apply(clean_text)
y = df["label"]

# 4. Training/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 5. Saved model load karna
model = joblib.load("spam_ml_model.pkl")

# 6. Test data par predictions
y_pred = model.predict(X_test)

# 7. Results
accuracy = accuracy_score(y_test, y_pred)

print("\n========== MODEL EVALUATION ==========")
print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# 8. Confusion matrix
labels = sorted(set(y_test) | set(y_pred))
cm = confusion_matrix(y_test, y_pred, labels=labels)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=labels
)

display.plot(cmap="Blues", values_format="d")
plt.title("Spam Detection - Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()

print("\nConfusion matrix saved as confusion_matrix.png")
