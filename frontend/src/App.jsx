import { useState, useRef } from 'react';
import { predictGenre } from './api';
import './App.css';

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [dragover, setDragover] = useState(false);
  const fileInputRef = useRef(null);

  // Handle file selection
  const handleFileSelect = (file) => {
    // Validate file type
    const allowedTypes = ['audio/wav', 'audio/mpeg', 'audio/mp3', 'audio/flac', 'audio/m4a', 'audio/ogg'];
    const fileExt = file.name.split('.').pop().toLowerCase();
    const allowedExtensions = ['wav', 'mp3', 'flac', 'm4a', 'ogg', 'wma'];
    
    if (!allowedExtensions.includes(fileExt)) {
      setError(`Unsupported file type: ${fileExt}. Please upload WAV, MP3, FLAC, M4A, or OGG files.`);
      setSelectedFile(null);
      return;
    }

    setSelectedFile(file);
    setError(null);
    setResult(null);
  };

  // Handle file input change
  const handleFileInputChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      handleFileSelect(file);
    }
  };

  // Handle drag and drop
  const handleDragOver = (e) => {
    e.preventDefault();
    setDragover(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setDragover(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setDragover(false);
    const file = e.dataTransfer.files[0];
    if (file) {
      handleFileSelect(file);
    }
  };

  // Handle form submission
  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!selectedFile) {
      setError('Please select an audio file first.');
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const prediction = await predictGenre(selectedFile);
      setResult(prediction);
    } catch (err) {
      setError(err.message || 'Failed to predict genre. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  // Format confidence as percentage
  const formatConfidence = (value) => {
    return (value * 100).toFixed(1);
  };

  // Sort probabilities by value
  const sortedProbabilities = result?.all_probabilities
    ? Object.entries(result.all_probabilities)
        .sort(([, a], [, b]) => b - a)
        .map(([genre, prob]) => ({ genre, probability: prob }))
    : [];

  return (
    <div className="app">
      <div className="header">
        <h1>🎵 Music Genre Classifier</h1>
        <p>Upload an audio file to predict its genre</p>
      </div>

      <form onSubmit={handleSubmit}>
        <div className="upload-section">
          <label className="upload-label">Select Audio File</label>
          <div className="file-input-wrapper">
            <input
              ref={fileInputRef}
              type="file"
              className="file-input"
              id="file-input"
              accept="audio/*"
              onChange={handleFileInputChange}
            />
            <label
              htmlFor="file-input"
              className={`file-input-label ${dragover ? 'dragover' : ''}`}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
            >
              <span>📁 Click to upload or drag and drop</span>
              <div className="file-info">
                Supported formats: WAV, MP3, FLAC, M4A, OGG
              </div>
            </label>
          </div>

          {selectedFile && (
            <div className="selected-file">
              <strong>Selected file:</strong> {selectedFile.name} ({(selectedFile.size / 1024 / 1024).toFixed(2)} MB)
            </div>
          )}
        </div>

        <button
          type="submit"
          className="submit-button"
          disabled={!selectedFile || loading}
        >
          {loading ? 'Predicting...' : 'Predict Genre'}
        </button>
      </form>

      {loading && (
        <div className="loading">
          <div className="spinner"></div>
          <div>Analyzing audio file...</div>
        </div>
      )}

      {error && (
        <div className="error">
          <div className="error-title">Error</div>
          <div>{error}</div>
        </div>
      )}

      {result && (
        <div className="result-section">
          <div className="result-title">Prediction Result</div>
          
          <div className="prediction-result">
            <div className="genre-name">{result.predicted_genre}</div>
            <div className="confidence">
              Confidence: <span className="confidence-value">{formatConfidence(result.confidence)}%</span>
            </div>
          </div>

          <div className="probabilities">
            <div className="probabilities-title">All Genre Probabilities</div>
            {sortedProbabilities.map(({ genre, probability }) => (
              <div
                key={genre}
                className={`probability-item ${
                  probability === result.confidence ? 'high-confidence' : ''
                }`}
              >
                <span className="probability-label">{genre}</span>
                <div className="probability-bar">
                  <div
                    className="probability-bar-fill"
                    style={{ width: `${probability * 100}%` }}
                  ></div>
                </div>
                <span className="probability-value">{formatConfidence(probability)}%</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

export default App;

