"""
TO RUN: python testing.py my_heartbeat.wav
NOTE: use python3
To ignore the warnings:
 python -W ignore testing.py my_heartbeat.wav
"""
import os
import sys
import librosa
import numpy as np
from tensorflow.keras.models import load_model
import warnings
warnings.filterwarnings('ignore')

def extract_features(audio_path, offset=0.5, duration=3):
    """Extract MFCC features with proper shape for CNN"""
    y, sr = librosa.load(audio_path, offset=offset, duration=duration, sr=None)
    
    # Extract MFCCs directly from audio (better than melspectrogram conversion)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40, n_fft=2048, hop_length=512)
    
    # Pad or truncate to consistent shape
    if mfccs.shape[1] < 130:  # Adjust 130 to match your training shape
        mfccs = np.pad(mfccs, ((0,0), (0,130-mfccs.shape[1])), mode='constant')
    else:
        mfccs = mfccs[:, :130]
    
    return mfccs.T  # Transpose to get (timesteps, features)

if __name__ == "__main__":
    # Load model
    try:
        model = load_model("heartbeat_classifier_normalised.h5")
        print("Model loaded successfully")
        print(f"Model input shape: {model.input_shape}")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        sys.exit(1)
    
    # Get audio file to classify
    if len(sys.argv) < 2:
        print("Usage: python testing.py audio_file.wav")
        sys.exit(1)
    
    classify_file = sys.argv[1]
    
    # Extract and prepare features
    try:
        features = extract_features(classify_file)
        print(f"Extracted features shape: {features.shape}")
        
        # Handle different model architectures
        if len(model.input_shape) == 5:  # 5D input expected (batch, sequence, height, width, channels)
            # Reshape to (1, 1, height, width, channels) - adding sequence dimension
            x_test = np.expand_dims(np.expand_dims(features, -1), 0)  # Add channel and batch dims
            x_test = np.expand_dims(x_test, 1)  # Add sequence dimension
            print(f"Reshaped for 5D model: {x_test.shape}")
        elif len(model.input_shape) == 4:  # 4D input expected (batch, height, width, channels)
            # Reshape to (1, height, width, channels)
            x_test = np.expand_dims(np.expand_dims(features, -1), 0)
            print(f"Reshaped for 4D model: {x_test.shape}")
        else:
            # For other architectures, try flattening
            x_test = features.flatten().reshape(1, -1)
            print(f"Flattened input: {x_test.shape}")
        
        # Predict
        pred = model.predict(x_test, verbose=0)
        print(f"Raw prediction: {pred}")
        
        # Get prediction (updated for modern Keras)
        if len(pred[0]) == 1:  # Binary classification with sigmoid
            pred_class = int(pred[0][0] > 0.5)
            confidence = pred[0][0] if pred_class == 1 else 1 - pred[0][0]
        else:  # Categorical classification
            pred_class = np.argmax(pred[0])
            confidence = np.max(pred[0])
        
        # Print results
        if pred_class:
            print("\nNormal heartbeat")
        else:
            print("\nAbnormal heartbeat")
        print(f"Confidence: {confidence:.2%}")
        
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        import traceback
        traceback.print_exc()