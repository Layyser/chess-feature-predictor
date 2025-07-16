# Chess Position Outcome Predictor

Small Python project to practice scikit-learn. It extracts features from chess positions, trains a Random Forest model, and compares it against a simple baseline predictor.

## Description of the project

- **Goal:** Predict the outcome of a chess position (white win, draw, black win) in any game.
- **Workflow:**
  1. (Optional) Extract positions from PGN files (e.g. from Lichess.org)
  2. Compute features for each position
  3. Evaluate feature importance using mutual information
  4. Train a Random Forest model
  5. Compare performance against a baseline predictor

I personally selected Random Forest because it is a robust algorithm. I just wanted to practice the main concepts of the scikit-learn without digging too much into parameter hypertuning.


## Installation

1. Clone the repository
```bash
   git clone https://github.com/Layyser/chess-feature-predictor.git
   cd chess-feature-predictor
```

2. Create and activate a virtual environment
```bash
   python3 -m venv venv
   source venv/bin/activate
```

3. Install dependencies
```bash
   pip install -r requirements.txt
```

## Usage

### 1. (Optional) Extract features
To extract features, you need to download a PGN file for example in [Lichess.org](https://database.lichess.org/) and execute extract_features.py
```bash
    python src/feature_extraction/extract_features.py 
    --pgn path/to/your/pgn
    --games 20000 
    --workers 8 
    --output data/processed/features.csv
```
- **NOTE:** PGN files are around 300GB so consider using the dataset.csv provided in data/raw/dataset.csv instead of extracting the features from the PGN file

### 2. Compute feature importance
```bash
   python src/feature_selection/compute_importance.py 
   --input data/raw/dataset.csv
```

### 3. Train and evaluate model
Consider to modify the code and save the model if needed
```bash
   python src/modeling/train_model.py 
   --input data/raw/dataset.csv
```

### 4. Run baseline predictor to compare
```bash
   python src/baseline/baseline_predictor.py
```

## Results
```
Test set accuracy: 0.8949

              precision    recall  f1-score   support
         0.0       0.89      0.90      0.89     54075
         1.0       0.97      0.79      0.87      6847
         2.0       0.89      0.91      0.90     56968

    accuracy                           0.89    117890
   macro avg       0.92      0.86      0.89    117890
weighted avg       0.90      0.89      0.89    117890
```

## Dependencies
- pandas
- numpy
- scikit-learn
- python-chess

## Next steps & ideas to explore
### 1. Hyperparameter tuning & model exploration
- Test additional algorithms (e.g. XGBoost, LightGBM, neural networks)
- Perform grid search or Bayesian optimization to fine‑tune parameters (e.g. tree depth, learning rate, number of estimators)
- Use cross‑validation (k‑fold, stratified) to get more robust performance estimates

### 2. Build a more representative dataset
- Aggregate PGN data over several months or years instead of a single month
- Filter out anomalous or low‑quality games (e.g. abandonments, ultra‑short games)

### 3. Advanced feature engineering
- Design new board‑state features (e.g. king safety metrics, mobility scores...)
- Encode move‑history information (e.g. repetition counts, move timestamps)
- Incorporate opening classifications or engine evaluations as features

### 4. Handle class imbalance
- Experiment with resampling techniques (SMOTE, ADASYN, under‑sampling)
- Adjust class weights in models or use cost‑sensitive learning

### 5. Pipeline & reproducibility improvements
- Implement a full sklearn pipeline that handles preprocessing, feature selection, and modeling in one workflow
- Add unit tests for feature extraction and model evaluation

### 6. Deployment & user interface
- Implement the model into my [Light-Chess deployment](https://layyser.github.io/Light-Chess/)

## License

This project is licensed under the MIT License.