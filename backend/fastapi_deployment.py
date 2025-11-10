import json
import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, conint, confloat
from typing import Optional, Dict, Any
import google.genai as genai
import google.genai.types as types
from dotenv import load_dotenv
from uuid import uuid4

# Load environment variables from .env file (if it exists)
load_dotenv()

# --- Global Configuration and State Management ---
# Global client and model placeholder
gemini_client: Optional[genai.Client] = None
GEMINI_MODEL_NAME = "gemini-2.5-flash"

# State Management (In-Memory Task Status) for decoupled requests
# Stores results: {task_id: {"status": "pending" | "complete" | "error", "suggestions": {...}}}
GEMINI_TASK_RESULTS: Dict[str, Dict[str, Any]] = {}

# Correct model file name is used here
RFR_MODEL_FILENAME = "random_forest_regressor_model.pkl"

FEATURE_NAMES = [
    'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade', 'sqft_basement', 'zipcode', 'sqft_lot15', 'years_since_last_renovation', 'age'
]

# --- 1. Pydantic Models for Input Validation and Output Structure ---

class HouseFeatures(BaseModel):
    """Input validation model for the 14 house features."""
    sqft_living: conint(gt=0) = Field(..., description="Square footage of the house's living area.")
    grade: conint(ge=1, le=13) = Field(..., description="Overall grade given to the housing unit.")
    bathrooms: confloat(ge=0.0) = Field(..., description="Number of bathrooms.")
    bedrooms: conint(ge=0) = Field(..., description="Number of bedrooms.")
    age: conint(ge=0) = Field(..., description="Age of the house in years.")
    floors: confloat(ge=1.0) = Field(..., description="Number of floors in the house.")
    sqft_basement: conint(ge=0) = Field(..., description="Square footage of the basement.")
    view: conint(ge=0, le=4) = Field(..., description="Level of view quality (0=poor, 4=excellent).")
    zipcode: conint(gt=0) = Field(..., description="5-digit ZIP code.")
    sqft_lot15: conint(gt=0) = Field(..., description="Average square footage of the lot for the 15 nearest neighbors.")
    sqft_lot: conint(gt=0) = Field(..., description="Square footage of the lot.")
    condition: conint(ge=1, le=5) = Field(..., description="Overall condition of the house (1=poor, 5=excellent).")
    waterfront: conint(ge=0, le=1) = Field(..., description="Presence of a waterfront (1=yes, 0=no).")
    years_since_last_renovation: conint(ge=0) = Field(..., description="Years since the last renovation (0 if never renovated).")

class GeminiSuggestionsModel(BaseModel):
    """Pydantic model for validating the raw JSON structure received from the Gemini API."""
    suggestion_1: str = Field(..., description="The first suggestion for increasing house value.")
    suggestion_2: str = Field(..., description="The second suggestion for increasing house value.")
    suggestion_3: str = Field(..., description="The third suggestion for increasing house value.")
    suggestion_4: str = Field(..., description="The fourth suggestion for increasing house value.")
    suggestion_5: str = Field(..., description="The fifth suggestion for increasing house value.")

class FastPredictionResponse(BaseModel):
    """Pydantic model for the FAST response of the /predict_price endpoint."""
    predicted_price: float = Field(..., description="The house price predicted by the Random Forest Regressor.")
    task_id: str = Field(..., description="Unique ID to retrieve Gemini suggestions.")
    input_features: HouseFeatures = Field(..., description="The original input features used for prediction.")

class SuggestionsStatusResponse(BaseModel):
    """Pydantic model for the response of the /get_suggestions/{task_id} endpoint (Polling)."""
    status: str = Field(..., description="Status of the Gemini task: 'pending', 'complete', or 'error'.")
    suggestions: Optional[GeminiSuggestionsModel] = Field(None, description="The 5 suggestions, available if status is 'complete'.")
    error_detail: Optional[str] = Field(None, description="Error message, available if status is 'error'.")


# --- 2. Model Loading and Setup ---

app = FastAPI(
    title="AI-Powered House Price Predictor API",
    description="A decoupled FastAPI service for predicting house prices instantly and providing value-add suggestions via Google Gemini in the background.",
    version="1.0.0"
)

# CORS Middleware setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def load_models():
    """
    Loads the Random Forest Regressor model and initializes the Gemini client on startup.
    """
    global rfr_model
    global gemini_client

    # Load RFR Model
    try:
        model_path = os.path.join(os.path.dirname(__file__), RFR_MODEL_FILENAME)
        rfr_model = joblib.load(model_path)
        print("RFR Model loaded successfully.")
    except FileNotFoundError:
        print(f"ERROR: {RFR_MODEL_FILENAME} not found. Prediction will fail.")
    except Exception as e:
        print(f"ERROR: Failed to load model: {e}")

    # Initialize Gemini Client
    try:
        gemini_client = genai.Client()
        print("Gemini Client initialized successfully.")
    except Exception as e:
        print(f"ERROR: Failed to initialize Gemini Client: {e}. Check API key.")


# --- 3. Gemini Background Task Function ---

def run_gemini_in_background(price: float, features: HouseFeatures, task_id: str):
    """
    Calls the Gemini API and stores the result in the global task results dictionary.
    This function is executed in a background thread by FastAPI.
    """
    global GEMINI_TASK_RESULTS

    if gemini_client is None:
         GEMINI_TASK_RESULTS[task_id] = {"status": "error", "error_detail": "Gemini API Client not initialized."}
         return
         
    # Convert HouseFeatures to a clean dict for the prompt
    features_dict = features.model_dump()
    
    prompt = f"""
    ## A house price prediction model (Random Forest Regressor) estimated the current market value of a property to be ${price:,.2f}.
    The property features used for this prediction are:
    {json.dumps(features_dict, indent=2)}
        
    ## System Instructions
    Act as a professional real estate analyst. Generate a concise, single-paragraph analysis (Precisely 5 sentences) offering actionable advice.
    The suggestions should focus on how the seller can improve the house's listing description or make simple, impactful improvements (like 'condition' or 'years_since_last_renovation') to potentially increase the predicted value.
    Be encouraging and highlight the strongest features and areas for quick improvement. Your answer should be in given JSON format.
    
    ## Required JSON Structure:
    {{
    "suggestion_1": "Prioritize Curb Appeal and Entryway Updates: First impressions matter significantly."
    "suggestion_2": "Execute a Minor Kitchen Remodel: Kitchens and bathrooms are crucial selling points."
    "suggestion_3": "Perform Essential Maintenance and Energy Efficiency Upgrades: Buyers want a move-in ready home without immediate major repair costs."
    "suggestion_4": "Boost Your Bathroom's Look: Similar to the kitchen, a fresh look in the bathroom is very appealing."
    "suggestion_5": "Use Paint Strategically (Interior and Exterior): A fresh coat of paint is one of the most cost-effective ways to increase a home's appeal."
    }}        
    """
    
    try:
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=GeminiSuggestionsModel,
        )

        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL_NAME,
            contents=[prompt],
            config=config,
        )

        # The response text will be a valid JSON string matching the schema
        json_data = json.loads(response.text)
        
        # Validate and structure the data with pydantic
        solution = GeminiSuggestionsModel(**json_data)
        
        # Store the successful result
        GEMINI_TASK_RESULTS[task_id] = {
            "status": "complete",
            "suggestions": solution.model_dump(), # Store as a dictionary for later retrieval
            "error_detail": None
        }

    except Exception as e:
        print(f"Gemini SDK call failed for task {task_id}: {e}")
        # Store the error
        GEMINI_TASK_RESULTS[task_id] = {
            "status": "error",
            "suggestions": None,
            "error_detail": f"Error communicating with Gemini SDK: {e}"
        }

# --- 4. API Endpoints ---

@app.post("/predict_price", response_model=FastPredictionResponse, tags=["Inference"])
async def predict_price(features: HouseFeatures, background_tasks: BackgroundTasks):
    """
    Receives house features, predicts the price instantly, and triggers the 
    Gemini LLM suggestion generation as a background task.
    """
    if rfr_model is None:
        raise HTTPException(status_code=503, detail="RFR Model is not loaded. Check startup logs.")
    
    # 1. Prepare data for the RFR model
    feature_values = [
        features.bedrooms, features.bathrooms, features.sqft_living, features.sqft_lot, 
        features.floors, features.waterfront, features.view, features.condition, 
        features.grade, features.sqft_basement, features.zipcode, features.sqft_lot15, 
        features.years_since_last_renovation, features.age
    ]
    
    # 2. Create a Pandas DataFrame to maintain feature names
    input_df = pd.DataFrame([feature_values], columns=FEATURE_NAMES)
    
    # 3. Perform RFR Inference
    try:
        prediction = rfr_model.predict(input_df)[0]
        predicted_price = float(prediction)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model prediction failed: {e}")

    # 4. Set up Background Task for Gemini
    task_id = str(uuid4())
    # Initialize task status
    GEMINI_TASK_RESULTS[task_id] = {"status": "pending", "suggestions": None}
    
    # Add the slow Gemini call to run after the response is sent
    background_tasks.add_task(run_gemini_in_background, predicted_price, features, task_id)

    # 5. Return the fast response immediately
    return FastPredictionResponse(
        predicted_price=round(predicted_price, 2),
        task_id=task_id,
        input_features=features
    )

@app.get("/get_suggestions/{task_id}", response_model=SuggestionsStatusResponse, tags=["Inference"])
async def get_suggestions(task_id: str):
    """
    Retrieves the status and results of the Gemini background task using a task_id.
    """
    if task_id not in GEMINI_TASK_RESULTS:
        raise HTTPException(status_code=404, detail="Task ID not found.")
    
    result = GEMINI_TASK_RESULTS[task_id]
    
    if result["status"] == "complete":
        # Return complete results
        return SuggestionsStatusResponse(
            status="complete", 
            suggestions=GeminiSuggestionsModel(**result["suggestions"]),
            error_detail=None
        )
    elif result["status"] == "error":
        # Return error status
        return SuggestionsStatusResponse(
            status="error", 
            suggestions=None, 
            error_detail=result["error_detail"]
        )
    else:
        # Return pending status
        return SuggestionsStatusResponse(
            status="pending", 
            suggestions=None, 
            error_detail=None
        )

if __name__ == "__main__":
    import uvicorn
    # To run this file: 
    # 1. Ensure 'random_forest_regressor_model.pkl' is in the same directory.
    # 2. Install dependencies: pip install fastapi uvicorn pydantic joblib pandas python-dotenv google-genai
    # 3. Run: uvicorn fastapi_deployment:app --reload --port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)