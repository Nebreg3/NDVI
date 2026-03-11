import warnings

import catboost as cb
import joblib
import pandas as pd
from sklearn.exceptions import InconsistentVersionWarning


def predict_with_catboost(model_path, scaler_path, input_data_path, output_data_path):
    """
    Load a pretrained CatBoost model and a scaler, apply them to new data,
    and save predictions into a .txt file.

    Parameters:
        model_path (str): Path to the pretrained CatBoost model (.cbm).
        scaler_path (str): Path to the saved scaler (.pkl).
        input_data_path (str): Path to the input dataset (.txt, TSV format).
        output_data_path (str): Path where the output file with predictions will be saved.
    """
    # Load model and scaler
    model = cb.CatBoostClassifier()
    model.load_model(model_path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", InconsistentVersionWarning)
        scaler = joblib.load(scaler_path)

    # Load new data
    new_data = pd.read_csv(input_data_path, sep='\t')

    # Select the same features used in training
    X_new = new_data[['NDVI', 'EVI', 'GNDVI', 'MSAVI']]

    # Scale the features
    X_new_scaled = scaler.transform(X_new)

    # Make predictions
    new_data['planted'] = model.predict(X_new_scaled)

    # Save the results
    new_data.to_csv(output_data_path, sep='\t', index=False)
    print(f"Predictions saved to {output_data_path}")

