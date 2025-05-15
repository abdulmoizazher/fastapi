from fastapi import FastAPI, HTTPException
import torch
from transformers import BertForSequenceClassification, BertTokenizer
from pydantic import BaseModel
from typing import List

# Initialize FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ✅ Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Load the trained model and tokenizer
model_path = "abdulmuzil/sentimental_model"   # Path to your trained model uploaded on hugging face 
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Set the model to evaluation mode
model.eval()

# List of emotions based on your training dataset
emotions = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization", "relief",
    "remorse", "sadness", "surprise", "neutral", "unknown", 'Depression', 'Suicidal',
    'Anxiety', 'Stress'  # Adjust according to your labels
]

# Define a Pydantic model for the input
class TextInput(BaseModel):
    text: str

# Define a Pydantic model for the output
class EmotionPrediction(BaseModel):
    emotion: str
    probability: float

class PredictionResponse(BaseModel):
    text: str
    predictions: List[EmotionPrediction]

# Define the prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
async def predict_emotion(input_data: TextInput):
    try:
        # Tokenize the input text
        inputs = tokenizer(input_data.text, padding=True, truncation=True, return_tensors="pt")

        # Get model predictions
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        # Apply sigmoid activation for multi-label classification
        probs = torch.sigmoid(logits).numpy()[0]

        # Prepare the response
        predictions = [
            {"emotion": emotion, "probability": float(probs[i])}
            for i, emotion in enumerate(emotions)
        ]

        return {
            "text": input_data.text,
            "predictions": predictions
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Health check route
@app.get("/health")
async def health_check():
    return {"status": "ok"}

# Run the server
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # Default to 8000 if PORT not set
    uvicorn.run("api:app", host="0.0.0.0", port=port, workers=4)