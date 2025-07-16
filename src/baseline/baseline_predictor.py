import pandas as pd


# This dummy predictor tries to predict who is going to win by:
# 1) The material predicts the winner
# 2) If not any material difference: the odds are the standard Elo formula odds
# This serves as a base comparison to my ML model


df = pd.read_csv("dataset90.csv")
df = df.dropna()

# Standard elo odds to win
def elo_probability(rating_a, rating_b):
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

# IMPORTANT: To denormalize the elo, it must be te opposite used during feature extraction
def elo_denorm(norm_elo):
    return norm_elo * (3000 - 1000) + 1000

def predict_winner(row):
    if row['material_diff'] > 0:
        return 2  # white wins
    elif row['material_diff'] < 0:
        return 0  # black wins
    else:
        p_white = elo_probability(elo_denorm(row['white_elo']), elo_denorm(row['black_elo']))
        p_black = 1 - p_white
        if p_white > p_black:
            return 2
        elif p_black > p_white:
            return 0
        else:
            return 1  # draw, very rare when probabilities are equal

df['prediction'] = df.apply(predict_winner, axis=1)

# Actual results and accuracy percentage
df['correct'] = (df['prediction'] == df['result']).astype(int)
accuracy = df['correct'].mean() * 100

print(df[['material_diff', 'white_elo', 'black_elo', 'result', 'prediction', 'correct']])
print(f"\nAccuracy: {accuracy:.4f}%")