import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import librosa
import numpy as np
import torchaudio
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold

class DatasetLoad(Dataset):
    def __init__(self, audio_files, labels):
        self.audio_files = audio_files
        self.labels = labels

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]
        label = self.labels[idx]
        waveform, sample_rate = torchaudio.load(audio_path)
        mfcc = librosa.feature.mfcc(y=waveform.numpy()[0], sr=sample_rate, n_mfcc=13)
        mfcc = np.mean(mfcc.T, axis=0)
        return torch.tensor(mfcc, dtype=torch.float32), label

class FNN_Model(nn.Module):
    def __init__(self, num_classes=10):
        super(FNN_Model, self).__init__()
        self.fc1 = nn.Linear(13, 128)  # Input size = 13 (MFCC feature size)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# CNN model
class CNN_Model(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN_Model, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 6, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x


# Training Function
def Train_one_epoch(model, train_loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for data in train_loader:
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / len(train_loader)
    accuracy = 100 * correct / total
    return epoch_loss, accuracy

# Evaluation Function
def Evaluate(model, val_loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data in val_loader:
            inputs, labels = data
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    val_loss = running_loss / len(val_loader)
    accuracy = 100 * correct / total
    return val_loss, accuracy

# Get Mean and Std of MFCCs for Normalization
def Get_mean_and_std(dataset):
    mfccs = []
    for data in dataset:
        mfcc, _ = data
        mfccs.append(mfcc)
    mfccs = torch.stack(mfccs)
    mean = mfccs.mean(dim=0)
    std = mfccs.std(dim=0)
    return mean, std

# Save Training Details
def Recod_and_Save_Train_Detial(train_details, file_path):
    np.save(file_path, train_details)

# FNN Model for Alternative
class FNN_Model(nn.Module):
    def __init__(self, Num_classes=10):
        super(FNN_Model, self).__init__()
        self.fc1 = nn.Linear(13, 128)
        self.fc2 = nn.Linear(128, Num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
