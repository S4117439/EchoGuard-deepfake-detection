import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# Load features and labels
X = np.load("artifacts/X.npy")
y = np.load("artifacts/y.npy")

print("Total training samples:", len(X))
print("Real samples:", int(np.sum(y == 0)))
print("Fake samples:", int(np.sum(y == 1)))

real_count = int(np.sum(y == 0))
fake_count = int(np.sum(y == 1))

if len(X) < 10 or real_count < 2 or fake_count < 2:
    print("\nDataset is too small for a proper train/test split.")
    print("Training on all available data only.")
    print("Warning: predictions will be unreliable until you add more samples.\n")

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42
    )
    model.fit(X, y)

    joblib.dump(model, "artifacts/echoguard_model.pkl")
    print("✅ Model trained and saved to artifacts/echoguard_model.pkl")
else:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("\nTraining set size:", len(X_train))
    print("Test set size:", len(X_test))

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {accuracy:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Real", "Fake"]))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    joblib.dump(model, "artifacts/echoguard_model.pkl")
    print("\n✅ Model trained and saved to artifacts/echoguard_model.pkl")