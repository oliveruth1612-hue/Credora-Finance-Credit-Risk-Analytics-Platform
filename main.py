import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib

# Load datasets
customers = pd.read_csv("dataset/customers (1).csv")
credit = pd.read_csv("dataset/credit_history.csv")
loans = pd.read_csv("dataset/loan_applications.csv")
transactions = pd.read_csv("dataset/transactions.csv")

# Merge Customers + Credit History
merged = pd.merge(customers, credit, on="customer_id", how="left")

# Merge Loan Applications
merged = pd.merge(merged, loans, on="customer_id", how="left")

print("Merged Dataset Shape:")
print(merged.shape)

print("\nFirst 5 Rows:")
print(merged.head())
print("Merged Dataset Shape:")
print(merged.shape)

print("\nFirst 5 Rows:")
print(merged.head())

print("\nMissing Values:")
print(merged.isnull().sum())
print("\nData Types:")
print(merged.dtypes)
# Fill missing values

# Numeric columns
numeric_cols = merged.select_dtypes(include=["float64", "int64"]).columns
merged[numeric_cols] = merged[numeric_cols].fillna(0)

# Text columns
text_cols = merged.select_dtypes(include=["object"]).columns
merged[text_cols] = merged[text_cols].fillna("Unknown")

print("\nMissing Values After Cleaning:")
print(merged.isnull().sum())
# Convert text columns into numbers
encoder = LabelEncoder()

for col in merged.select_dtypes(include=["object"]).columns:
    merged[col] = encoder.fit_transform(merged[col].astype(str))

print("\nEncoded Data:")
print(merged.head())
# Keep only disbursed loans (where loan_default is available)
merged = merged[merged["loan_default"] != "Unknown"]

# Features (X)
X = merged.drop([
    "loan_default",
    "customer_id",
    "credit_id",
    "application_id",
    "first_name",
    "last_name",
    "email",
    "phone",
    "approval_status",
    "approved_amount",
    "decision_date",
    "disbursed_flag"
], axis=1)

# Target (y)
y = merged["loan_default"]

print("\nLoan Default Values:")
print(y.unique())




# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train model
model = RandomForestClassifier(
    random_state=42,
    class_weight="balanced"
)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]
risk_score = (y_prob * 100).round(2)
print("\nSample Risk Scores:")
print(risk_score[:10])
risk_category = []

for score in risk_score:
    if score < 30:
        risk_category.append("Low Risk")
    elif score < 70:
        risk_category.append("Medium Risk")
    else:
        risk_category.append("High Risk")

print("\nSample Risk Categories:")
print(risk_category[:10])
feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 10 Important Features:")
print(feature_importance.head(10))

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print(f"\nModel Accuracy: {accuracy:.2%}")
roc_auc = roc_auc_score(y_test, y_prob)
print(f"ROC-AUC Score: {roc_auc:.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
joblib.dump(model, "model/loan_approval_model.pkl")

print("Model saved successfully!")