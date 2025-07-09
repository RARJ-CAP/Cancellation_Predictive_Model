import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

def train_logistic_regression(X_train, y_train, X_test, y_test):
    """
    Train and evaluate a logistic regression model.
    """
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    print("=== Logistic Regression Results ===")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("ROC AUC Score:", roc_auc_score(y_test, y_prob))
    print("-" * 40)
    return model

def train_random_forest(X_train, y_train, X_test, y_test):
    """
    Train and evaluate a random forest classifier.
    """
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    print("=== Random Forest Results ===")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("ROC AUC Score:", roc_auc_score(y_test, y_prob))
    print("-" * 40)
    # Feature importances
    importances = pd.Series(model.feature_importances_, index=X_train.columns)
    print("Random Forest Feature Importances:\n", importances.sort_values(ascending=False))
    return model

def main():
    # Load the data
    df = pd.read_excel("Output.xlsx")
    print(f"Loaded {len(df)} rows from Output.xlsx")

    # Drop columns not useful for modeling (if present)
    drop_cols = ['reservation_date', 'reservation_time']
    df = df.drop(columns=[col for col in drop_cols if col in df.columns])

    # Check if there are enough samples
    if len(df) < 10:
        print("Not enough data to train/test split. Please provide more samples.")
        return

    # Define features and target
    X = df.drop(columns=['will_cancel'])
    y = df['will_cancel']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train and evaluate both models
    train_logistic_regression(X_train, y_train, X_test, y_test)
    train_random_forest(X_train, y_train, X_test, y_test)

if __name__ == "__main__":
    main()