"""
Feature extraction module for music genre classification.
Extracts mel-spectrogram features from audio files using librosa.
"""

import numpy as np
import librosa


def extract_mel_spectrogram(audio_data, sample_rate, target_shape=(128, 128)):
    """
    Extract mel-spectrogram from audio data.
    
    Args:
        audio_data: Audio signal as numpy array
        sample_rate: Sample rate of the audio
        target_shape: Target shape for the mel-spectrogram (n_mels, time_steps)
        
    Returns:
        Mel-spectrogram as numpy array with shape (1, n_mels, time_steps)
    """
    # Generate mel spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(
        y=audio_data, 
        sr=sample_rate, 
        n_mels=target_shape[0]
    )
    
    # Fix length to target shape
    mel_spectrogram = librosa.util.fix_length(mel_spectrogram, size=target_shape[1])
    
    # Add channel dimension: (1, n_mels, time_steps)
    mel_spectrogram = np.expand_dims(mel_spectrogram, axis=0)
    
    return mel_spectrogram


def load_and_preprocess_audio(file_path, chunk_duration=4, overlap_duration=2, target_shape=(128, 128), return_all_chunks=False):
    """
    Load audio file and extract mel-spectrogram features.
    
    Args:
        file_path: Path to audio file
        chunk_duration: Duration of each chunk in seconds
        overlap_duration: Overlap between chunks in seconds
        target_shape: Target shape for mel-spectrogram
        return_all_chunks: If True, return all chunks (for training). If False, return first chunk only (for prediction)
        
    Returns:
        Array of mel-spectrograms and sample_rate
    """
    # Load audio file
    audio_data, sample_rate = librosa.load(file_path, sr=None)
    
    # Calculate chunk parameters
    chunk_samples = chunk_duration * sample_rate
    overlap_samples = overlap_duration * sample_rate
    
    # Ensure minimum length
    if len(audio_data) < chunk_samples:
        # Pad if too short
        audio_data = librosa.util.fix_length(audio_data, size=chunk_samples)
        chunks = [audio_data]
    elif return_all_chunks:
        # For training: split into all chunks with overlap
        num_chunks = int(np.ceil((len(audio_data) - chunk_samples) / (chunk_samples - overlap_samples))) + 1
        chunks = []
        for i in range(num_chunks):
            start = i * (chunk_samples - overlap_samples)
            end = min(start + chunk_samples, len(audio_data))
            chunk = audio_data[start:end]
            # Ensure chunk has correct length
            if len(chunk) < chunk_samples:
                chunk = librosa.util.fix_length(chunk, size=chunk_samples)
            chunks.append(chunk)
    else:
        # For prediction: use the first chunk only
        chunk = audio_data[:chunk_samples]
        chunks = [chunk]
    
    # Extract features for each chunk
    features = []
    for chunk in chunks:
        mel_spec = extract_mel_spectrogram(chunk, sample_rate, target_shape)
        features.append(mel_spec)
    
    return np.array(features), sample_rate


def extract_features_from_file(file_path, target_shape=(128, 128)):
    """
    Simplified function to extract features from a file for prediction.
    Returns the first chunk's features.
    
    Args:
        file_path: Path to audio file
        target_shape: Target shape for mel-spectrogram
        
    Returns:
        Mel-spectrogram features ready for model input
    """
    features, _ = load_and_preprocess_audio(file_path, target_shape=target_shape)
    return features[0]  # Return first chunk

