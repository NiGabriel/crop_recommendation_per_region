###########################################################################################
#=========================================================================================#
# File: <control.py>                                                                      #
# Authors: <Gabriel Nishimwe>, <Manzi Kananura Justin>, <Gad Rukundo>, <Dalia Bwiza>      #
# Date Created: <04 Nov 2024>                                                             #
# Last Modified: <21 Nov 2024>                                                            #
#                                                                                         #
# This file contains functions to load dataset, retrain the model, and predict            #
# the top 5 crops. The adopted model was XGBoost.                                         #
#=========================================================================================#
###########################################################################################

import os
import pandas as pd
import numpy as np
import json
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import warnings
import re
from datetime import datetime

warnings.filterwarnings("ignore")


# Define the folder where dataset files are stored as a string path.
DATASET_FOLDER = 'dataset'

# Define the folder where model files are stored as a string path.
MODELS_FOLDER = 'models'

# Initialize the dataset file name as None, to be assigned later.
DATASET_NAME = None

# Initialize the path for the crop data file as None, to be assigned later.
CROP_DATA_FILE = None

# Initialize the path for the region data file as None, to be assigned later.
REGION_DATA_FILE = None


################################################################
#                   GETTING THE LATEST DATASET                 #
################################################################

def get_latest_dataset(folder):
    """Find the latest dataset files in the given folder."""
    global CROP_DATA_FILE, REGION_DATA_FILE
    
    crop_files = [f for f in os.listdir(folder) if f.endswith('.csv')]
    region_files = [f for f in os.listdir(folder) if f.endswith('.xlsx')]

    if crop_files:
        CROP_DATA_FILE = max(crop_files, key=lambda f: os.path.getmtime(os.path.join(folder, f)))

    if region_files:
        REGION_DATA_FILE = max(region_files, key=lambda f: os.path.getmtime(os.path.join(folder, f)))

    
    if not CROP_DATA_FILE and not REGION_DATA_FILE:
        raise FileNotFoundError("Required dataset files not found in the directory.")


# Create directories if they don't exist
os.makedirs(DATASET_FOLDER, exist_ok=True)
os.makedirs(MODELS_FOLDER, exist_ok=True)


################################################################
#                       LOADING THE DATASET                    #
################################################################

def load_data():
    """Load crop and region dataset"""
    try:
        # Ensure CROP_DATA_FILE and REGION_DATA_FILE are set
        if not CROP_DATA_FILE and not REGION_DATA_FILE:
            raise ValueError("CROP_DATA_FILE and REGION_DATA_FILE are not set. Ensure `get_latest_dataset` is called before loading data.")
    
        crop_data_path = os.path.join(DATASET_FOLDER, CROP_DATA_FILE)
        region_data_path = os.path.join(DATASET_FOLDER, REGION_DATA_FILE)

        print(CROP_DATA_FILE)
        print(REGION_DATA_FILE)

        # Print paths to help with debugging
        print(f"Loading crop data from: {crop_data_path}")
        print(f"Loading region data from: {region_data_path}")

        crop_data = pd.read_csv(crop_data_path)
        region_data = pd.read_excel(region_data_path, sheet_name = 2)

        if crop_data.empty and region_data.empty:
            raise RuntimeError("One of the datasets is empty. Please check the files.")

        print("Data loaded successfully")
        return crop_data, region_data
    
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        raise RuntimeError("Data loading failed. Please check your files.")
    except ValueError as e:
        print(f"Configuration Error: {e}")
        raise RuntimeError("Data loading failed. Please check your files.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise RuntimeError("Data loading failed. Please check your files.")

# Ensure the latest dataset is identified and load the data
try:
    get_latest_dataset(DATASET_FOLDER)  # This should update CROP_DATA_FILE
    crop_data = load_data()  # Now load the latest data
    region_data = load_data()

except FileNotFoundError as e:
    print(f"Error: {e}")
    exit(1)  # Exit the program if no datasets are found
except RuntimeError as e:
    print(f"Error: {e}")
    exit(1)  # Exit the program if data loading fails

# Load the data
crop_data, region_data = load_data()

if crop_data is None and region_data is None:
    raise RuntimeError("Data loading failed. Please check your files.")


# Step 1: Drop rows with specific unwanted text entries
unwanted_text = ['Protected Land', 'Water Body']
region_data = region_data[~region_data['pH'].isin(unwanted_text)]

 
# Step 2: Function to calculate average pH
def calculate_average_ph(ph_value):
    # Handle special case for values like '<5.0'
    if '<' in ph_value:
        # Convert '<5.0' to 5.0 (assuming '<5.0' means "up to 5.0")
        return float(ph_value.replace('<', ''))
    elif '>' in ph_value:
        return float(ph_value.replace('>',''))

    # Handle range values like '6.0-7.0'
    if '-' in ph_value:
        lower, upper = map(float, ph_value.split('-'))
        return (lower + upper) / 2

    # Convert single numeric value to float
    return float(ph_value)

# Step 3: Apply the function to the 'pH' column
region_data['pH_avg'] = region_data['pH'].apply(calculate_average_ph)


################################################################
#                       RETRAINING THE MODEL                   #
################################################################

# Initialize the models
models = {
    'XGBoost': XGBClassifier(
        n_estimators=200,                # Increase number of boosting rounds
        learning_rate=0.05,              # Lower learning rate
        max_depth=4,                     # Decrease max depth to avoid overfitting
        min_child_weight=3,              # Minimum sum of instance weight needed in a child
        gamma=0.1,                       # Minimum loss reduction required to split
        subsample=0.8,                   # Fraction of samples used for training each tree
        colsample_bytree=0.8,            # Fraction of features used for training each tree
        reg_lambda=1.5,                  # L2 regularization
        reg_alpha=0.5,                   # L1 regularization (adds sparsity)
        eval_metric='mlogloss',
        random_state=42
    )
}

label_encoder = LabelEncoder()


def train_and_evaluate(models, X_train, X_test, y_train, y_test):
    
    """
    Trains and evaluates a collection of machine learning models.

    This function performs the following steps for each model:
    - Trains the model using the provided training data.
    - Predicts outcomes on the testing data.
    - Calculates the accuracy of the model's predictions.
    - Handles any exceptions during training and records errors if they occur.
    - Returns a dictionary containing each model's trained instance, accuracy, 
      and any errors encountered during training.

    Parameters:
    - models (dict): A dictionary of model names as keys and model instances as values.
    - X_train (pd.DataFrame or np.array): Training feature data.
    - X_test (pd.DataFrame or np.array): Testing feature data.
    - y_train (pd.Series or np.array): Training target labels.
    - y_test (pd.Series or np.array): Testing target labels.

    Returns:
    - dict: A dictionary with model names as keys and their results (including the trained model 
      and accuracy) as values.
    """

    results = {}
    for name, model in models.items():
        print(f"Training {name}...")
        try:
            # Train the model
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Calculate accuracy
            accuracy = accuracy_score(y_test, y_pred)
            print(f"\n{name} Accuracy: {accuracy:.4f}")


            # Store results
            results[name] = {
                'model': model,
                'accuracy': accuracy
            }
        except Exception as e:
            print(f"Error training {name}: {str(e)}")
            results[name] = {
                'model': None,
                'accuracy': None,
                'error': str(e)
            }
    return results


def retrain_model(crop_file_path, region_file_path):

    """
    Handles the retraining of a machine learning model using a new dataset. 

    This function performs the following tasks:
    - Validates if the uploaded file is a CSV.
    - Renames the new dataset file to avoid conflicts with existing files.
    - Updates the global dataset file path for future reference.
    - Loads the dataset and prepares it for model training.
    - Splits the dataset into training and testing subsets.
    - Uses train_and_evaluate function to train and evaluate model using a predefined function.
    - Identifies and saves the best-performing model to a specified folder.
    - Returns the training results, including model performance metrics.
    """

    global CROP_DATA_FILE, REGION_DATA_FILE  # Add this to modify the global variable
    
    # Check if the file is a CSV
    if not crop_file_path.endswith('.csv'):
        raise ValueError("The uploaded file must be a CSV file.")
    
    # Check if the file is a CSV
    if not region_file_path.endswith('.xlsx'):
        raise ValueError("The uploaded file must be a XLSX file.")

    # Logic for naming the new file
    crop_base_name, ext = os.path.splitext(os.path.basename(crop_file_path))
    crop_existing_files = [f for f in os.listdir(DATASET_FOLDER) if f.startswith(crop_base_name) and f.endswith(ext)]

    region_base_name, ext = os.path.splitext(os.path.basename(region_file_path))
    region_existing_files = [f for f in os.listdir(DATASET_FOLDER) if f.startswith(region_base_name) and f.endswith(ext)]

    if crop_existing_files:
        crop_numbers = [
            int(re.search(r'_(\d+)', f).group(1))
            for f in crop_existing_files if re.search(r'_(\d+)', f)
        ]
        crop_new_number = max(crop_numbers, default=0) + 1
        crop_new_file_name = f"{crop_base_name}_{crop_new_number}{ext}"
    else:
        crop_new_file_name = f"{crop_base_name}_1{ext}"

    
    if region_existing_files:
        region_numbers = [
            int(re.search(r'_(\d+)', f).group(1))
            for f in region_existing_files if re.search(r'_(\d+)', f)
        ]
        region_new_number = max(region_numbers, default=0) + 1
        region_new_file_name = f"{region_base_name}_{region_new_number}{ext}"
    else:
        region_new_file_name = f"{region_base_name}_1{ext}"

    crop_renamed_file_path = os.path.join(DATASET_FOLDER, crop_new_file_name)

    region_renamed_file_path = os.path.join(DATASET_FOLDER, region_new_file_name)

    try:
        # Load the new dataset
        crop_new_data = pd.read_csv(crop_file_path)
        region_new_data = pd.read_excel(region_file_path, sheet_name = 2)

        # Save the new dataset with the generated name
        crop_new_data.to_csv(crop_renamed_file_path, index=False)
        region_new_data.to_excel(region_renamed_file_path, index=False)

        # Update the global dataset file
        CROP_DATA_FILE = crop_renamed_file_path  # Update the global dataset path

        REGION_DATA_FILE = region_renamed_file_path

        # Load the dataset for retraining
        crop_data = pd.read_csv(crop_renamed_file_path)

        region_data = pd.read_excel(region_renamed_file_path)

        sampled_dataset = crop_data

        # Prepare features and target
        train_features = sampled_dataset[['Altitude (masl)', 'temperature (C) ', 'pH', 'N', 'P', 'K',
                                           'Crop water need (mm/total growing period)', 'Humidity(%)']]
        target = sampled_dataset['Crop']

        # Encode the target variable
        label_encoder = LabelEncoder()
        target_encoded = label_encoder.fit_transform(target)

        # Split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            train_features, target_encoded, test_size=0.2, random_state=42
        )

        # Assuming you have a `train_and_evaluate` function defined elsewhere
        results = train_and_evaluate(models, X_train, X_test, y_train, y_test)

        # Save the best model
        best_model_name = max(results, key=lambda name: results[name]['accuracy'])
        best_model = results[best_model_name]['model']
        model_path = os.path.join(MODELS_FOLDER, f"{best_model_name}_model.pkl")
        joblib.dump(best_model, model_path)

        # Serialize results to avoid issues with JSON serialization
        serialized_results = json.loads(json.dumps(results, default=str))

        print(CROP_DATA_FILE)

        return {
            "message": f"Model retrained and saved at {model_path}",
            "dataset": crop_renamed_file_path,
            "results": serialized_results
        }

    except Exception as e:
        raise ValueError(f"Error during model retraining: {str(e)}")
    


################################################################
#                       PREDICT THE CROP                       #
################################################################

def preprocess_data(crop_data):
    """Preprocess dataset for training"""
    crop_dataset = crop_data

    train_features = crop_dataset[['Altitude (masl)', 'temperature (C) ', 'pH', 'N', 'P', 'K',
                              'Crop water need (mm/total growing period)', 'Humidity(%)']]
    target = crop_dataset['Crop']
    target_encoded = label_encoder.fit_transform(target)

    X_train, X_test, y_train, y_test = train_test_split(train_features, target_encoded, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


X_train, X_test, y_train, y_test = preprocess_data(crop_data)


def get_model_from_directory(model_name):
    """Retrieve a saved model from the 'models' folder without training or prediction"""
    model_path = f"models/{model_name}.pkl"
    
    if os.path.exists(model_path):
        try:
            model = joblib.load(model_path)
            print(f"Successfully loaded the {model_name} model.")
            return model
        except Exception as e:
            print(f"Error loading {model_name} model: {e}")
            return None
    else:
        print(f"Model {model_name} not found in the 'models' folder.")
        return None
    

def fetch_crop_data(crop_names, dataset):
    """
    Fetch additional crop details from the dataset, computing min and max for numerical fields
    and avoiding floats in the output.
    """
    crop_details_dict = {}
    for crop in crop_names:
        crop_data = dataset[dataset['Crop'] == crop]

        if not crop_data.empty:
            crop_details_dict[crop] = {
                "season_a_start": crop_data['Season A start(month)'].iloc[0] if 'Season A start(month)' in crop_data else None,
                "season_a_end": crop_data['Season A end'].iloc[0] if 'Season A end' in crop_data else None,
                "season_b_start": crop_data['Season B start(month)'].iloc[0] if 'Season B start(month)' in crop_data else None,
                "season_b_end": crop_data['Season B end(month)'].iloc[0] if 'Season B end(month)' in crop_data else None,
                "soil_type": crop_data['Soil type'].iloc[0] if 'Soil type' in crop_data else None,
                "crop_water_need": {
                    "min": int(crop_data['Crop water need (mm/total growing period)'].min()),
                    "max": int(crop_data['Crop water need (mm/total growing period)'].max())
                } if 'Crop water need (mm/total growing period)' in crop_data else None,
                "growing_period_days": {
                    "min": int(crop_data['Growing period (days)'].min()),
                    "max": int(crop_data['Growing period (days)'].max())
                } if 'Growing period (days)' in crop_data else None,
            }
    return crop_details_dict



def convert_to_serializable(obj):
    """
    Converts non-serializable objects into serializable formats (e.g., JSON-friendly).
    Handles specific types like numpy integers, dictionaries, lists, and objects with a __dict__ attribute.
    """
    if isinstance(obj, np.int64):
        return int(obj)
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif hasattr(obj, "__dict__"):
        return vars(obj)
    else:
        return str(obj)

def extract_monthly_avg(data, start_month, end_month, col_prefix):
    # Ensure that start_month and end_month are integers
    start_month = int(start_month)
    end_month = int(end_month)

    # Generate month indices based on the start and end months
    if start_month <= end_month:
        month_indices = list(range(start_month, end_month + 1))
    else:  # Handle case where the period spans the year-end (e.g., Nov to Feb)
        month_indices = list(range(start_month, 13)) + list(range(1, end_month + 1))

    # Extract relevant column names based on the month indices
    monthly_columns = [f"{col_prefix} - {datetime(1900, month, 1).strftime('%b')}" for month in month_indices]

    # Calculate the average of the selected months
    avg_value = data[monthly_columns].mean(axis=1).values[0]

    return avg_value


# Define the mapping of months to seasons
SEASON_MAP = {
    'A': [9, 10, 11, 12, 1],  # September to January
    'B': [2, 3, 4, 5, 6],     # February to June
    'C': [7, 8]               # July to August
}

def get_season_from_month(month):
    """Determine the season based on the given month."""
    for season, months in SEASON_MAP.items():
        if month in months:
            return season
    return None

def get_month_avg(region_data, start_date):
    """Get average temperature and total rainfall based on the season of the start date."""
    start_month = datetime.strptime(start_date, "%Y-%m-%d").month

    # Determine the season for the given month
    season = get_season_from_month(start_month)

    if not season:
        raise ValueError("Invalid month provided, unable to determine season.")

    # Get the list of months for the identified season
    season_months = SEASON_MAP[season]

    # Calculate the average temperature for the season
    avg_temperature = extract_monthly_avg(region_data, season_months[0], season_months[-1], "Average Temperature (°C)")

    # Calculate the total rainfall for the season manually
    total_rainfall = 0
    for month in season_months:
        monthly_rainfall = extract_monthly_avg(region_data, month, month, "Average Precipitation (mm)")
        if monthly_rainfall is not None:
            total_rainfall += monthly_rainfall

    return avg_temperature, total_rainfall
    

def predict_crops(user_input):

    """
    Predict the best crops based on user input features.
    
    Args:
        user_input (dict): A dictionary containing the input features.
        
    Returns:
        dict: A dictionary containing the top crop predictions with probabilities and additional details.
    """

    # Filter region data based on user input
    filtered_region = region_data[
        (region_data['District'] == user_input['district']) &
        (region_data['Sector'] == user_input['sector'])
    ]


    print(CROP_DATA_FILE)

    # Calculate temperature and rainfall based on the provided start date
    temperature, rainfall = get_month_avg(
        filtered_region,
        user_input['start_date_to_plant']
    )

    # Prepare the features for the model using the fixed values
    features = {
        'Altitude (masl)': filtered_region['Elevation'].values[0].item(),
        'temperature (C) ': temperature,
        'pH': filtered_region['pH_avg'].values[0].item(),
        'N': filtered_region['Nitrogen(%)'].values[0].item(),
        'P': filtered_region['Phosphorus(ppm)'].values[0].item(),
        'K': filtered_region['Potassium(ppm)'].values[0].item(),
        'Crop water need (mm/total growing period)': rainfall,
        'Humidity(%)': filtered_region['Humidity(%)'].values[0].item()
    }
    
    # Convert features into a DataFrame
    features_df = pd.DataFrame([features])
    
    try:
        # Convert columns to appropriate types
        for col in features_df.columns:
            features_df[col] = pd.to_numeric(features_df[col], errors='coerce')

        if features_df.isnull().any().any():
            return {"error": "Some input features contain invalid or missing values."}
        
        # Fetch the best model (replace with your model retrieval function)
        best_model = get_model_from_directory("XGBoost_model")
        if best_model is None:
            return {"error": "No trained model available."}
        
        # Ensure the feature order matches the model's training data
        feature_columns = [
            'Altitude (masl)', 'temperature (C) ', 'pH', 'N', 'P', 'K',
            'Crop water need (mm/total growing period)', 'Humidity(%)'
        ]
        features_df = features_df[feature_columns]
        
        # Predict probabilities
        try:
            predictions_proba = best_model.predict_proba(features_df)[0]
        except AttributeError as e:
            predicted_class = best_model.predict(features_df)[0]
            return [{"crop_name": label_encoder.inverse_transform([predicted_class])[0]}]
        
        # Get the indices of the top 5 predictions with the highest probability
        top_n = 5
        top_n_indices = np.argsort(predictions_proba)[-top_n:][::-1]
        
        # Decode the crop names and get their probabilities
        top_crops = label_encoder.inverse_transform(top_n_indices)
        top_probabilities = predictions_proba[top_n_indices]
        
        crop_data_path = os.path.join(DATASET_FOLDER, CROP_DATA_FILE)

        crop_data = pd.read_csv(CROP_DATA_FILE)

        print(crop_data_path)

        # Fetch additional data for the predicted crops
        crop_details = fetch_crop_data(top_crops, crop_data)
        
        # Format the output
        output = {
            "predicted_crops": [
                {
                    "crop_name": crop,
                    "probability": float(probability),
                    **crop_details.get(crop, {})  # Merge crop details if available
                }
                for crop, probability in zip(top_crops, top_probabilities)
            ]
        }
        
        # Convert to serializable format
        output_serializable = convert_to_serializable(output)
        print(json.dumps(output_serializable, indent=4))
        return output_serializable
    except Exception as e:
        return {"error": f"Prediction error: {str(e)}"}