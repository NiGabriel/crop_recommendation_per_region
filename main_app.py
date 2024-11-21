###########################################################################################
#=========================================================================================#
# File: <main.py>                                                                         #
# Authors: <Gabriel Nishimwe>, <Manzi Kananura Justin>, <Gad Rukundo>, <Dalia Bwiza>      #
# Date Created: <02 Oct 2024>                                                             #
# Last Modified: <21 Nov 2024>                                                            #
#                                                                                         #
# This file contains 2 endpoints for retraining the model and predicting 5 top crops      #
# according to the inputs from the user.                                                  #
#=========================================================================================#
###########################################################################################

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from control_file import retrain_model, predict_crops
from pathlib import Path
import traceback

app = FastAPI()

# Define the folder where the dataset files are stored as a Path object.
DATASET_FOLDER = Path("dataset")

# Define the name of the dataset file.
CROP_DATASET_NAME = "filtered_crop_suitability_dataset.csv"

REGION_DATASET_NAME = "crop_recommendations.xlsx"

# Combine the dataset folder and dataset file name to get the full path to the dataset.
CROP_DATASET_PATH = DATASET_FOLDER / CROP_DATASET_NAME

REGION_DATASET_PATH = DATASET_FOLDER / REGION_DATASET_NAME

# Ensure the dataset folder exists
DATASET_FOLDER.mkdir(exist_ok=True)


"""
The following class defines the schema for crop prediction input data using Pydantic's BaseModel.
Each attribute represents a specific feature required for crop prediction, and all fields are defined as strings.
This schema ensures input validation and provides a structured format for the prediction endpoint.
"""
class CropPrediction(BaseModel):
    district: str
    sector: str
    start_date_to_plant: str


    
################################################################
#                       PREDICT THE CROP                       #
################################################################

"""
This endpoint receives crop prediction data from the user, processes it into a structured format,
and sends it to the `predict_crops` function for prediction. The input data is validated using the 
`CropPrediction` Pydantic model, and any issues in processing the input data are caught and logged. 
The endpoint then returns the crop prediction response.
"""

@app.post("/predict_crop")
async def predict(input_data: CropPrediction):

    try:
        features = {
            'district': input_data.district,
            'sector': input_data.sector,
            'start_date_to_plant': input_data.start_date_to_plant
        }
    except Exception as e:
        print("\nDEBUG: Error in processing input data:", str(e))
        raise
 
    response = predict_crops(features)
    return response


################################################################
#                       RETRAIN THE MODEL                      #
################################################################

"""
This endpoint handles the retraining of the crop prediction model. It accepts a CSV file upload 
from the user, validates its format, and updates the dataset used for training. The file content 
is saved directly to the dataset path, and the model is retrained using the updated data. 
Any errors encountered during the upload or retraining process are caught and returned as HTTP exceptions.

"""
@app.post("/retrain")
async def retrain_the_model(
    crop_file: UploadFile = File(...),
    region_file: UploadFile = File(...)
):
    # Validate the uploaded crop file
    if not crop_file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="The crop dataset file must be a CSV.")
    if not region_file.filename.endswith(".xlsx"):
        raise HTTPException(status_code=400, detail="The region dataset file must be an Excel file.")

    try:
        # Save the crop dataset file
        crop_content = await crop_file.read()
        with open(str(CROP_DATASET_PATH), "wb") as f:  # Ensure path is a string
            f.write(crop_content)
        print(f"Crop dataset file saved at {CROP_DATASET_PATH}.")

        # Save the region dataset file
        region_content = await region_file.read()
        with open(str(REGION_DATASET_PATH), "wb") as f:  # Ensure path is a string
            f.write(region_content)
        print(f"Region dataset file saved at {REGION_DATASET_PATH}.")

        # Retrain the model with the updated datasets
        print("Starting model retraining...")
        results = retrain_model(str(CROP_DATASET_PATH), str(REGION_DATASET_PATH))  # Pass paths as strings

        return {"message": "Model retrained successfully.", "results": results}

    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error during model retraining: {str(e)}")