# Music Genre Classifier - Frontend

React frontend for the Music Genre Classifier application.

## Setup

1. **Install dependencies:**
   ```bash
   npm install
   ```

## Running

**Development server:**
```bash
npm run dev
```

The frontend will be available at `http://localhost:3000`

**Build for production:**
```bash
npm run build
```

**Preview production build:**
```bash
npm run preview
```

## Configuration

The frontend is configured to connect to the backend at `http://localhost:8000` by default.

To change the backend URL, create a `.env` file in the frontend directory:
```
VITE_API_URL=http://localhost:8000
```

## Features

- Upload audio files via drag & drop or file picker
- Real-time genre prediction
- Confidence scores and probability distributions
- Loading states and error handling
- Clean, modern UI

## Supported Audio Formats

- WAV
- MP3
- FLAC
- M4A
- OGG
- WMA

## File Structure

```
frontend/
├── index.html
├── package.json
├── vite.config.js
├── src/
│   ├── App.jsx        # Main application component
│   ├── App.css        # Styles
│   ├── main.jsx       # Entry point
│   └── api.js         # API client
└── README.md
```

