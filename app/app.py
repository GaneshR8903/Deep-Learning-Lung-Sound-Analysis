import streamlit as st
import torch
import torchaudio
import librosa
import numpy as np
import matplotlib.pyplot as plt
import requests
from io import BytesIO
from scipy.signal import butter, lfilter
from sklearn.preprocessing import LabelEncoder
from Deep_learning import DatasetLoad, CNN_Model, Train_one_epoch, Evaluate, \
    Get_mean_and_std, Recod_and_Save_Train_Detial, FNN_Model

# Define a function to load the model
def load_model(model_path, num_classes=10):
    model = CNN_Model(num_classes=num_classes)
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model

def preprocess_audio(audio_path):
    waveform, sample_rate = torchaudio.load(audio_path, backend='soundfile')

    # Apply filtering (lowpass example)
    def butter_lowpass(cutoff, fs, order=5):
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return b, a

    def butter_lowpass_filter(data, cutoff, fs, order=5):
        b, a = butter_lowpass(cutoff, fs, order)
        y = lfilter(b, a, data)
        return y

    filtered_waveform = butter_lowpass_filter(waveform.numpy(), cutoff=1000, fs=sample_rate)
    mfcc = librosa.feature.mfcc(y=filtered_waveform.flatten(), sr=sample_rate, n_mfcc=13)
    mfcc = np.mean(mfcc.T, axis=0)
    mfcc_tensor = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return waveform, filtered_waveform, mfcc_tensor


def analyze_waveform(waveform, sample_rate, threshold=0.05):
    """
    Analyze waveform to detect more deviations based on statistical measures.
    """
    waveform = waveform.flatten()
    time = np.linspace(0, len(waveform) / sample_rate, len(waveform))
    deviations = []

    # Using a moving average or z-score method for better deviation detection
    window_size = 50  # Define a window for calculating moving average
    moving_avg = np.convolve(waveform, np.ones(window_size) / window_size, mode='valid')
    deviations = np.abs(waveform[window_size-1:] - moving_avg) > threshold
    
    # Collect deviation points
    deviation_points = [(time[i + window_size - 1], waveform[i + window_size - 1]) 
                        for i in range(len(deviations)) if deviations[i]]
    
    return deviation_points, time

def plot_waveform_with_deviations(waveform, sr, deviations, title):
    """
    Plot the waveform and mark deviations.
    """
    fig, ax = plt.subplots(figsize=(10, 3))
    time = np.linspace(0., len(waveform) / sr, len(waveform))
    ax.plot(time, waveform, label="Waveform", color="blue")

    if deviations:
        deviation_times, deviation_values = zip(*deviations)
        ax.scatter(deviation_times, deviation_values, color="red", label="Deviations", s=10)
    
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.legend()
    st.pyplot(fig)

def predict_disease(mfcc, models, label_encoder):
    """
    Predict disease using the ensemble of models.
    """
    model_predictions = []
    mfcc = mfcc.to(dtype=torch.float32)
    
    for model in models:
        model.eval()
        with torch.no_grad():
            output = model(mfcc)
            _, predicted = torch.max(output, 1)
            model_predictions.append(predicted.item())
    
    final_prediction = max(set(model_predictions), key=model_predictions.count)
    
    # Decode the prediction using the label encoder
    disease = label_encoder.inverse_transform([final_prediction])[0]
    return disease

# Load models
models = []
model_paths = [
    r"app\Fold_0_model_weights_Epoch_Final.pth",
    r"app\Fold_1_model_weights_Epoch_Final.pth",
    r"app\Fold_2_model_weights_Epoch_Final.pth",
    r"app\Fold_3_model_weights_Epoch_Final.pth",
    r"app\Fold_4_model_weights_Epoch_Final.pth"
]

for path in model_paths:
    models.append(load_model(path, num_classes=10))

# Load label encoder
label_encoder = LabelEncoder()
label_encoder.classes_ = np.load('label_encoder_classes.npy', allow_pickle=True)

# Streamlit UI
st.title("AI-Powered Lung Disease Diagnosis")
uploaded_file = st.file_uploader("Upload Lung Sound File (WAV Format)", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')
    waveform, filtered_waveform, mfcc = preprocess_audio(uploaded_file)

    # Correcting the function call for plot_waveform_with_deviations
    deviations, time = analyze_waveform(waveform, 16000, threshold=0.05)
    plot_waveform_with_deviations(waveform.flatten(), 16000, deviations, title="Original Lung Sound")

    deviations, time = analyze_waveform(filtered_waveform, 16000, threshold=0.09)
    plot_waveform_with_deviations(filtered_waveform.flatten(), 16000, deviations, title="Filtered Lung Sound")

    # Compute the difference between the original and filtered waveforms
    difference_waveform = waveform - filtered_waveform

    st.write("### Deviations Detected")
    deviations_count = len(deviations)
    st.write(f"Total Deviations Detected: {deviations_count}")
    st.write("These deviations may indicate potential abnormalities in the lung sound.")

    prediction = predict_disease(mfcc, models, label_encoder)
    if prediction:
        st.write(f"Predicted Lung Disease: **{prediction}**")
    else:
        st.write("The system detected an unrecognized pattern. Please check the model configuration.")
