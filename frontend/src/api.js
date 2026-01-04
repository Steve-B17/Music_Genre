/**
 * API client for communicating with the FastAPI backend
 */

import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'multipart/form-data',
  },
});

/**
 * Predict genre from audio file
 * @param {File} file - Audio file to classify
 * @returns {Promise} Prediction response with genre, confidence, and probabilities
 */
export const predictGenre = async (file) => {
  const formData = new FormData();
  formData.append('file', file);

  try {
    const response = await api.post('/predict', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  } catch (error) {
    if (error.response) {
      // Server responded with error status
      throw new Error(error.response.data.detail || 'Prediction failed');
    } else if (error.request) {
      // Request was made but no response received
      throw new Error('Unable to connect to server. Please ensure the backend is running.');
    } else {
      // Something else happened
      throw new Error(error.message || 'An unexpected error occurred');
    }
  }
};

/**
 * Check API health
 * @returns {Promise} Health status
 */
export const checkHealth = async () => {
  try {
    const response = await api.get('/health');
    return response.data;
  } catch (error) {
    return { status: 'unhealthy', model_loaded: false };
  }
};

