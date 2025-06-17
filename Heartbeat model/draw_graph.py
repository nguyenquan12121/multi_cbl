import os
import sys
import librosa
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scipy.io import wavfile

from matplotlib.widgets import TextBox

def extract_features(audio_path, offset=0.5, duration=3):
    """Extract MFCC features with proper shape for CNN"""
    y, sr = librosa.load(audio_path, offset=offset, duration=duration, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40, n_fft=2048, hop_length=512)
    
    if mfccs.shape[1] < 130:
        mfccs = np.pad(mfccs, ((0,0), (0,130-mfccs.shape[1])), mode='constant')
    else:
        mfccs = mfccs[:, :130]
    
    return mfccs.T

def create_visualization(audio_path, prediction, confidence):
    """Create the visualization plot with results"""
    # Load audio file
    sample_rate, data = wavfile.read(audio_path)
    
    # If stereo, use left channel only
    if len(data.shape) > 1:
        data = data[:, 0]
    
    time = np.arange(0, len(data)) / sample_rate
    n = len(data)
    freq = np.fft.rfftfreq(n, d=1/sample_rate)
    fft_data = np.abs(np.fft.rfft(data))
    
    # Create figure with customized subplot heights
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), 
                             gridspec_kw={'height_ratios': [3, 3, 1.25]})
    ax1, ax2 = axes

    # Time domain plot
    ax1.plot(time, data, color='b')
    ax1.set_title('Audio Waveform')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude')
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Frequency domain plot
    ax2.plot(freq, fft_data, color='r')
    ax2.set_title('Frequency Spectrum')
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Magnitude')
    ax2.set_xlim(0, 5000)
    ax2.grid(True, linestyle='--', alpha=0.6)

    # Result display, closer to middle graph
    ax2.axis('off')
    ax2.text(0.1, 0.9, f"Prediction: {prediction}",
          fontsize=12,
          family='monospace',
          bbox=dict(facecolor='red', alpha=0.8))
    ax2.text(0.1, 0.7, f"Confidence: {confidence:.2%}",
             fontsize=12,
             family='monospace',
             bbox=dict(facecolor='green', alpha=0.8))

    # Figure styling
    fig.patch.set_facecolor('#f5f5f5')
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    # Load model
    try:
        model = load_model("heartbeat_classifier_normalised.h5")
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        sys.exit(1)
    
    # Get audio file
    if len(sys.argv) < 2:
        print("Usage: python testing.py audio_file.wav")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    
    try:
        # Extract and prepare features
        features = extract_features(audio_file)
        x_test = np.expand_dims(np.expand_dims(features, -1), 0)
        
        # Predict
        pred = model.predict(x_test, verbose=0)
        
        # Get prediction
        if len(pred[0]) == 1:  # Binary classification
            pred_class = int(pred[0][0] > 0.5)
            confidence = pred[0][0] if pred_class == 1 else 1 - pred[0][0]
        else:  # Categorical classification
            pred_class = np.argmax(pred[0])
            confidence = np.max(pred[0])
        
        # Determine prediction label
        prediction = "Normal heartbeat" if pred_class else "Abnormal heartbeat"
        
        # Create and save visualization
        fig = create_visualization(audio_file, prediction, confidence)
        output_file = os.path.splitext(audio_file)[0] + "_results.png"
        fig.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Results saved to {output_file}")
        
        # Show plot (optional)
        plt.show()
        
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        import traceback
        traceback.print_exc()
        