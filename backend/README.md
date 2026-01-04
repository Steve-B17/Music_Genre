# Music Genre Classifier - Backend

FastAPI backend for music genre classification using PyTorch CNN model.

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Train the model:**
   - Place your audio data in a directory structure like:
     ```
     genres_original/
     ├── blues/
     │   ├── file1.wav
     │   └── file2.wav
     ├── classical/
     ├── country/
     └── ...
     ```
   - Update the `data_dir` path in `train.py` if needed
   - Run training:
     ```bash
     python train.py
     ```
   - This will save the model to `./model/genre_model.pth`

## Running the API

```bash
python app.py
```

Or using uvicorn directly:
```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

## API Endpoints

### GET `/`
Root endpoint with API information.

### GET `/health`
Health check endpoint. Returns model loading status.

### POST `/predict`
Predict genre from uploaded audio file.

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body: audio file (WAV, MP3, FLAC, etc.)

**Response:**
```json
{
  "predicted_genre": "rock",
  "confidence": 0.85,
  "all_probabilities": {
    "blues": 0.01,
    "classical": 0.02,
    "country": 0.03,
    "disco": 0.02,
    "hiphop": 0.05,
    "jazz": 0.01,
    "metal": 0.10,
    "pop": 0.15,
    "reggae": 0.03,
    "rock": 0.85
  }
}
```

## Model Architecture

- **Input:** Mel-spectrogram (128x128)
- **Architecture:** CNN with 3 convolutional layers + 2 fully connected layers
- **Classes:** 10 genres (blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock)

## File Structure

```
backend/
├── app.py                 # FastAPI application
├── train.py              # Training script
├── feature_extraction.py # Feature extraction utilities
├── model/
│   └── genre_model.pth   # Trained model (generated after training)
├── requirements.txt
└── README.md
```

