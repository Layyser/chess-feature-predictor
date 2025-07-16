import pandas as pd
import argparse
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif

# This short script is used to fastly check the importance of each feature
#
# Even though there are some features with "low mutual information" 
# those features may be critical to some algorithms such as Random Forest
#
# For example: turn may not seem important, 
# however the material diff may be more impactful than the elo if the game is ending

def main():
    parser = argparse.ArgumentParser(
        description="Quickly compute mutual information scores for features"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to the input CSV (must include a 'result' column)"
    )
    args = parser.parse_args()

    # Load and clean data
    df = pd.read_csv(args.input)
    df = df.dropna()

    # Separate features and target
    X = df.drop(columns=["result"])
    y = df["result"]

    # Encode target to numeric labels
    y_encoded = LabelEncoder().fit_transform(y)

    # Compute mutual information
    print("Computing MI...")
    mi = mutual_info_classif(X, y_encoded, discrete_features='auto')
    mi_scores = pd.Series(mi, index=X.columns).sort_values(ascending=False)

    # Print results
    print("Mutual information (MI) for each feature with the target:")
    for feature, score in mi_scores.items():
        print(f"{feature:20s}: {score:.6f}")

if __name__ == "__main__":
    main()