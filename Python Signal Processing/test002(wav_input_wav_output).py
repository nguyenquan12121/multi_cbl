import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, filtfilt, sosfilt

def butter_bandpass(lowcut, highcut, fs, order):
    """Butterworth bandpass filter design."""
    return butter(order, [lowcut, highcut], btype='band', fs=fs, output='sos')

def process_audio(input_path, output_path):
    # Load the audio file
    fs, data = wavfile.read(input_path)
    
    # Convert stereo to mono if needed
    if len(data.shape) > 1:
        data = data[:, 0]
    
    # Normalize and center signal
    data = data.astype(np.float32)
    data -= np.mean(data)
    
    # Apply bandpass filter (50-180 Hz)
    sos = butter_bandpass(50, 180, fs, order=4)
    filtered = sosfilt(sos, data)
    
    # Apply amplification (20x gain)
    filtered *= 20
    
    # Normalize to 16-bit integer range and convert
    max_val = np.max(np.abs(filtered))
    if max_val > 0:
        filtered = filtered / max_val  # Scale to [-1, 1]
    filtered_int = np.int16(filtered * 32767)
    
    # Save processed audio
    wavfile.write(output_path, fs, filtered_int)
    print(f"Filtered audio saved to {output_path}")

if __name__ == "__main__":
    input_file = r'C:\Users\oliwi\OneDrive\Desktop\Q4\CBL Digital Twin\github_repo\multi_cbl\Python Signal Processing\output8.wav'
    output_file = r'C:\Users\oliwi\OneDrive\Desktop\Q4\CBL Digital Twin\github_repo\multi_cbl\Python Signal Processing\filtered_output8.wav'
    process_audio(input_file, output_file)