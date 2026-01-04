"""
FastAPI application for Music Genre Classification.
Provides REST API endpoint for predicting music genres from audio files.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import librosa
from pathlib import Path
import tempfile
from feature_extraction import extract_features_from_file

# Model configuration
MODEL_PATH = "./model/genre_model.pth"
CLASSES = ['blues', 'classical', 'country', 'disco', 'hiphop',
           'jazz', 'metal', 'pop', 'reggae', 'rock']
TARGET_SHAPE = (128, 128)

# Initialize FastAPI app
app = FastAPI(title="Music Genre Classifier API", version="1.0.0")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model variable
model = None


class MusicGenreClassifier(nn.Module):
    """CNN model for music genre classification."""
    
    def __init__(self, num_classes=10):
        super(MusicGenreClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding='same')
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding='same')
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding='same')
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)

        # Calculate the input size for the fully connected layer
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, 128, 128)
            x = F.relu(self.conv1(dummy_input))
            x = F.relu(self.conv2(x))
            x = self.pool(F.relu(self.conv3(x)))
            self.fc1_input_size = x.view(1, -1).shape[1]

        self.fc1 = nn.Linear(self.fc1_input_size, 1200)
        self.fc2 = nn.Linear(1200, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def load_model():
    """Load the trained model from disk."""
    global model, CLASSES
    
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Please train the model first using train.py"
        )
    
    # Load checkpoint
    checkpoint = torch.load(MODEL_PATH, map_location='cpu')
    
    # Initialize model
    num_classes = checkpoint.get('num_classes', len(CLASSES))
    model = MusicGenreClassifier(num_classes=num_classes)
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # Set to evaluation mode
    
    # Update classes if available in checkpoint
    if 'classes' in checkpoint:
        CLASSES = checkpoint['classes']
    
    print(f"Model loaded successfully from {MODEL_PATH}")
    print(f"Number of classes: {len(CLASSES)}")
    print(f"Classes: {CLASSES}")


@app.on_event("startup")
async def startup_event():
    """Load model when the API starts."""
    try:
        load_model()
    except Exception as e:
        print(f"Error loading model: {e}")
        print("API will start but predictions will fail until model is available.")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Music Genre Classifier API",
        "version": "1.0.0",
        "endpoints": {
            "predict": "POST /predict - Upload an audio file to predict its genre"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }


@app.post("/predict")
async def predict_genre(file: UploadFile = File(...)):
    """
    Predict the genre of an uploaded audio file.
    
    Accepts: WAV, MP3, FLAC, and other formats supported by librosa
    Returns: Predicted genre, confidence score, and all class probabilities
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please ensure the model file exists."
        )
    
    # Validate file type
    allowed_extensions = {'.wav', '.mp3', '.flac', '.m4a', '.ogg', '.wma'}
    file_ext = Path(file.filename).suffix.lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file_ext}. Supported formats: {', '.join(allowed_extensions)}"
        )
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
        try:
            # Write uploaded file to temporary file
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
            
            # Validate file size (basic check)
            if len(content) < 1000:  # Less than 1KB is likely invalid
                raise HTTPException(
                    status_code=400,
                    detail="File is too small. Please upload a valid audio file."
                )
            
            # Extract features
            try:
                features = extract_features_from_file(tmp_file_path, target_shape=TARGET_SHAPE)
            except Exception as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Error processing audio file: {str(e)}. Please ensure the file is a valid audio file."
                )
            
            # Convert to tensor and add batch dimension
            features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
            
            # Make prediction
            with torch.no_grad():
                outputs = model(features_tensor)
                probabilities = torch.exp(outputs)  # Convert log probabilities to probabilities
                probabilities_np = probabilities[0].cpu().numpy()
            
            # Get predicted class
            predicted_idx = np.argmax(probabilities_np)
            predicted_genre = CLASSES[predicted_idx]
            confidence = float(probabilities_np[predicted_idx])
            
            # Create dictionary of all probabilities
            all_probabilities = {
                CLASSES[i]: float(probabilities_np[i])
                for i in range(len(CLASSES))
            }
            
            return JSONResponse(content={
                "predicted_genre": predicted_genre,
                "confidence": confidence,
                "all_probabilities": all_probabilities
            })
            
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error during prediction: {str(e)}"
            )
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

