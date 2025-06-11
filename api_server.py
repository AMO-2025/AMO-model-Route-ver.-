from fastapi import FastAPI, UploadFile, File
from typing import Optional
from Inference import EmotionPredictor

app = FastAPI()

# Initialize the emotion predictor
predictor = None

@app.on_event("startup")
async def load_model():
    """
    Load the model when the FastAPI application starts up.
    """
    global predictor
    try:
        predictor = EmotionPredictor()
    except Exception as e:
        print(f"Error loading model: {e}")
        raise RuntimeError(f"Failed to load model: {e}")

@app.post("/ml/analyze/emotion")
async def analyze_emotion(file: UploadFile = File(...), expectedEmotion: Optional[str] = None):
    """
    Receives an image file from the frontend and analyzes the emotion.
    Returns the predicted emotion and confidence, and checks against expectedEmotion if provided.
    """
    if predictor is None:
        return {"status": "error", "message": "Model not loaded. Server might be starting up or encountered an error during startup."}

    image_data = await file.read()
    predicted_label, confidence = predictor.predict_from_bytes(image_data)

    response_data = {
        "emotionTag": predicted_label,
        "confidence": round(confidence * 100, 2)
    }

    if expectedEmotion:
        if predicted_label == expectedEmotion:
            response_data["status"] = "match"
            response_data["message"] = "correct"
        else:
            response_data["status"] = "mismatch"
            response_data["message"] = "fail"
            response_data["expectedEmotion"] = expectedEmotion
    else:
        response_data["status"] = "success"
        response_data["message"] = "Emotion analysis completed."

    return response_data

# To run the API:
# 1. Save this code as a Python file (e.g., api_server.py)
# 2. Open your terminal in the directory where you saved the file.
# 3. Run: uvicorn api_server:app --reload 