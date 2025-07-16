import pandas as pd
import argparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


def main():
    parser = argparse.ArgumentParser(
        description="Train and evaluate Random Forest classifier on chess position data"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to the input CSV (must include a 'result' column)"
    )
    args = parser.parse_args()

    # Load data
    df = pd.read_csv(args.input)
    # Filter out invalid results if any (e.g., -2)
    df = df[df["result"] != -2]
    df = df.dropna(subset=["result"])

    # Split features and target
    X = df.drop(columns=["result"])
    y = df["result"]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Optionally scale features (uncomment if needed)
    # scaler = StandardScaler()
    # X_train = scaler.fit_transform(X_train)
    # X_test = scaler.transform(X_test)

    # Define models
    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            verbose=100,
            n_jobs=-1,
            random_state=42
        ),
    }

    # Train and evaluate
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        print(f"Accuracy in test: {acc:.4f}")
        print("Report:")
        print(classification_report(y_test, y_pred))
        print("-" * 40)

if __name__ == "__main__":
    main()