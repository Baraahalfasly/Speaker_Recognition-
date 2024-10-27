import sounddevice as sd
import numpy as np
import librosa
import os
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import joblib

# Path to audio files
data_dir = r"C:\Users\m288756\Desktop\Baraah3\broject2"  # Make sure to use the correct path

# Names of speakers
speaker_names = ["abhar", "afkar"]  # Speaker names

# Load audio samples and labels
X = []  # Features
y = []  # Labels

for speaker in speaker_names:
    speaker_dir = os.path.join(data_dir, speaker)
    if not os.path.exists(speaker_dir):
        print(f"Directory does not exist: {speaker_dir}")
        continue

    print(f"Loading files from: {speaker_dir}")
    
    for file in os.listdir(speaker_dir):
        if file.endswith(".mp3") or file.endswith(".wav"):
            file_path = os.path.join(speaker_dir, file)
            # Load the audio file
            audio, sr = librosa.load(file_path, sr=None)
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            mfccs_processed = np.mean(mfccs.T, axis=0)  # Average across time
            X.append(mfccs_processed)
            y.append(speaker)
            print(f"Loaded: {file_path}")  # Print the loaded file path

# Check if we have any data before proceeding
if not X or not y:
    print("No audio data found for training. Please check your audio files.")
else:
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Train the model
    model = SVC(kernel='linear', probability=True)
    model.fit(X, y_encoded)

    # Save the model
    joblib.dump(model, 'speaker_model.pkl')

    print("Model trained and saved successfully.")

# Function to recognize the speaker
def recognize_speaker():
    print("Listening... Press Ctrl + C to stop.")
    while True:
        try:
            duration = 5  # Seconds
            audio = sd.rec(int(duration * 44100), samplerate=44100, channels=1)
            sd.wait()  # Wait until the recording is finished
            audio = audio.flatten()
            mfccs = librosa.feature.mfcc(y=audio, sr=44100, n_mfcc=13)
            mfccs_processed = np.mean(mfccs.T, axis=0).reshape(1, -1)

            # Predict the speaker
            prediction = model.predict(mfccs_processed)
            speaker_name = label_encoder.inverse_transform(prediction)
            print(f"Recognized speaker: {speaker_name[0]}")
        
        except KeyboardInterrupt:
            print("Microphone closed. Exiting the program.")
            break
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    recognize_speaker()
