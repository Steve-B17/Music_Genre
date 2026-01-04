# Music Genre Classifier - Complete Application

End-to-end music genre classification application with FastAPI backend and React frontend.

## Project Structure

```
.
├── backend/
│   ├── app.py                # FastAPI application
│   ├── train.py              # Training script
│   ├── feature_extraction.py # Feature extraction utilities
│   ├── model/
│   │   └── genre_model.pth   # Trained model (generated after training)
│   ├── requirements.txt
│   └── README.md
│
├── frontend/
│   ├── index.html
│   ├── package.json
│   ├── vite.config.js
│   ├── src/
│   │   ├── App.jsx           # Main React component
│   │   ├── main.jsx          # Entry point
│   │   ├── api.js            # API client
│   │   └── App.css           # Styles
│   └── README.md
│
├── Music_Genre_Classifier.ipynb  # Original Jupyter notebook
└── README.md                     # This file
```

## Quick Start

### Prerequisites

- Python 3.8+ with pip
- Node.js 16+ with npm
- Audio dataset for training (optional - model can be pre-trained)

### Step 1: Setup Backend

1. Navigate to backend directory:
   ```bash
   cd backend
   ```

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. (Optional) Train the model:
   - Place your audio data in a directory structure:
     ```
     genres_original/
     ├── blues/
     ├── classical/
     ├── country/
     ├── disco/
     ├── hiphop/
     ├── jazz/
     ├── metal/
     ├── pop/
     ├── reggae/
     └── rock/
     ```
   - Update `data_dir` in `train.py` if needed
   - Run training:
     ```bash
     python train.py
     ```
   - This saves the model to `backend/model/genre_model.pth`

4. Start the FastAPI server:
   ```bash
   python app.py
   ```
   
   Or using uvicorn:
   ```bash
   uvicorn app:app --reload --host 0.0.0.0 --port 8000
   ```

   The API will be available at `http://localhost:8000`

### Step 2: Setup Frontend

1. Open a new terminal and navigate to frontend directory:
   ```bash
   cd frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm run dev
   ```

   The frontend will be available at `http://localhost:3000`

### Step 3: Use the Application

1. Open `http://localhost:3000` in your browser
2. Upload an audio file (WAV, MP3, FLAC, M4A, or OGG)
3. Click "Predict Genre" to get the classification result
4. View the predicted genre, confidence score, and all genre probabilities

## Features

### Backend
- FastAPI REST API
- PyTorch CNN model for genre classification
- Mel-spectrogram feature extraction using librosa
- Support for multiple audio formats
- Error handling and validation
- CORS enabled for frontend communication

### Frontend
- React + Vite application
- Drag & drop file upload
- Real-time genre prediction
- Beautiful, modern UI
- Loading states and error handling
- Probability visualization

## API Endpoints

### `GET /`
Root endpoint with API information.

### `GET /health`
Health check endpoint.

### `POST /predict`
Predict genre from uploaded audio file.

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body: audio file

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

## Model Details

- **Architecture:** Convolutional Neural Network (CNN)
- **Input:** Mel-spectrogram (128x128)
- **Classes:** 10 genres (blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock)
- **Training:** 15 epochs with Adam optimizer

## Trained Model
Download the trained model (.pth):
[Model Link](https://huggingface.co/Ste17/Music_genre_classifier)

## Development

### Backend Development

- The backend uses FastAPI with automatic API documentation
- Visit `http://localhost:8000/docs` for interactive API documentation
- Model is loaded once at startup for efficient predictions

### Frontend Development

- Uses Vite for fast development and hot module replacement
- API client in `src/api.js` handles all backend communication
- Styled with CSS (no external frameworks)

## Troubleshooting

### Backend Issues

- **Model not found:** Ensure you've trained the model using `train.py` or have a pre-trained model at `backend/model/genre_model.pth`
- **Port already in use:** Change the port in `app.py` or use `uvicorn app:app --port 8001`
- **Import errors:** Make sure all dependencies are installed with `pip install -r requirements.txt`

### Frontend Issues

- **Cannot connect to backend:** Ensure the backend is running on port 8000, or update `VITE_API_URL` in `.env`
- **Build errors:** Clear `node_modules` and reinstall: `rm -rf node_modules package-lock.json && npm install`

## License

This project is for educational purposes.

