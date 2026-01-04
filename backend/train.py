"""
Training script for Music Genre Classification model.
Trains a PyTorch CNN model and saves it for use in the API.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
from sklearn.model_selection import train_test_split
from feature_extraction import load_and_preprocess_audio


# Genre classes
CLASSES = ['blues', 'classical', 'country', 'disco', 'hiphop',
           'jazz', 'metal', 'pop', 'reggae', 'rock']


class AudioDataset(Dataset):
    """Dataset class for audio data."""
    
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


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


def load_data(data_dir, classes, target_shape=(128, 128)):
    """
    Load and preprocess all audio files from the data directory.
    
    Args:
        data_dir: Root directory containing genre subdirectories
        classes: List of genre class names
        target_shape: Target shape for mel-spectrograms
        
    Returns:
        data: Array of mel-spectrograms
        labels: Array of class labels
    """
    data = []
    labels = []
    
    for i_class, class_name in enumerate(classes):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.exists(class_dir):
            print(f"Warning: Directory {class_dir} does not exist. Skipping.")
            continue
            
        print(f"Processing-- {class_name}")
        for filename in os.listdir(class_dir):
            if filename.endswith('.wav'):
                file_path = os.path.join(class_dir, filename)
                try:
                    # Load and preprocess audio - return_all_chunks=True for training
                    audio_data, sample_rate = load_and_preprocess_audio(
                        file_path, 
                        chunk_duration=4,
                        overlap_duration=2,
                        target_shape=target_shape,
                        return_all_chunks=True
                    )
                    
                    # Add all chunks to dataset
                    for chunk_features in audio_data:
                        data.append(chunk_features)
                        labels.append(i_class)
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
                    continue

    return np.array(data), np.array(labels)


def evaluate_model(model, loader, criterion, device):
    """Evaluate model on validation data."""
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    average_loss = running_loss / len(loader)
    return average_loss, accuracy


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    """Train the model."""
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        val_loss, val_accuracy = evaluate_model(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

    return train_losses, train_accuracies, val_losses, val_accuracies


def main():
    """Main training function."""
    # Configuration
    data_dir = "./genres_original"  # Update this path to your data directory
    model_save_path = "./model/genre_model.pth"
    num_epochs = 15
    batch_size = 16
    learning_rate = 0.0001
    target_shape = (128, 128)
    
    # Check if data directory exists
    if not os.path.exists(data_dir):
        print(f"Error: Data directory '{data_dir}' not found.")
        print("Please update the 'data_dir' variable in train.py to point to your data directory.")
        return
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("Loading and preprocessing data...")
    data, labels = load_data(data_dir, CLASSES, target_shape)
    
    if len(data) == 0:
        print("Error: No data loaded. Please check your data directory.")
        return
    
    print(f"Loaded {len(data)} samples")
    
    # Split data
    X_train, X_val, Y_train, Y_val = train_test_split(
        data, labels, test_size=0.2, random_state=42
    )
    
    print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
    
    # Create datasets and dataloaders
    train_dataset = AudioDataset(X_train, Y_train)
    val_dataset = AudioDataset(X_val, Y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Create model
    model = MusicGenreClassifier(num_classes=len(CLASSES)).to(device)
    
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Train model
    print("\nStarting training...")
    train_losses, train_accuracies, val_losses, val_accuracies = train_model(
        model, train_loader, val_loader, criterion, optimizer, num_epochs, device
    )
    
    # Save model
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_class': MusicGenreClassifier,
        'num_classes': len(CLASSES),
        'classes': CLASSES,
        'target_shape': target_shape
    }, model_save_path)
    
    print(f"\nModel saved to {model_save_path}")
    print("Training complete!")


if __name__ == "__main__":
    main()

