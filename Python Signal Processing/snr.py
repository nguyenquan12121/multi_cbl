import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import butter, filtfilt, find_peaks, sosfilt

# Butterworth bandpass filter
def butter_bandpass(lowcut, highcut, fs, order):
    return butter(order, [lowcut, highcut], btype='band', fs=fs, output='sos')

# Envelope smoothing filter
def butter_lowpass(cutoff, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    return butter(order, normal_cutoff, btype='low')

# Load the audio file
filename = r'C:\Users\oliwi\OneDrive\Desktop\Q4\CBL Digital Twin\github_repo\multi_cbl\Python Signal Processing\output8.wav'
fs, data = wavfile.read(filename)

# ---------- Preprocess ----------
# Convert stereo to mono if needed
if len(data.shape) > 1:
    data = data[:, 0]

# Normalize and center
data = data.astype(np.float32)
data -= np.mean(data)

# ----------- Noise Filtering Analysis -----------
# Save original power before filtering
original_power = np.mean(data ** 2)

# Bandpass filter (50–180 Hz)
sos = butter_bandpass(50, 180, fs, order=4)
filtered = sosfilt(sos, data)
filtered *= 20  # software gain

# Compute residual noise
residual = data - filtered
residual_power = np.mean(residual ** 2)
filtered_power = np.mean(filtered ** 2)

# Noise reduction ratio and dB
noise_reduction_ratio = residual_power / original_power
noise_reduction_db = 10 * np.log10(original_power / residual_power)

print(f"Original Power: {original_power:.2f}")
print(f"Filtered Power: {filtered_power:.2f}")
print(f"Residual (Noise) Power: {residual_power:.2f}")
print(f"Noise Reduction Ratio: {noise_reduction_ratio:.2f}")
print(f"Noise Reduction: {noise_reduction_db:.2f} dB")

# ----------- Envelope and Peak Detection -----------
envelope = np.abs(filtered)
b_env, a_env = butter_lowpass(20, fs, order=2)
env_smoothed = filtfilt(b_env, a_env, envelope)

# Peak detection
min_spacing_samples = int(0.1 * fs)
peak_threshold = np.max(env_smoothed) * 0.4
peaks, _ = find_peaks(env_smoothed, distance=min_spacing_samples, height=peak_threshold)

# Plot
t = np.arange(len(filtered)) / fs

plt.figure(figsize=(12, 6))
plt.plot(t, filtered, label="Filtered Signal (50–180 Hz)", alpha=0.7)
plt.plot(t, env_smoothed, label="Envelope", color='orange')
plt.plot(t[peaks], env_smoothed[peaks], 'rx', label="Detected Peaks")
plt.title("Heart Sound Detection")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# Heartbeat BPM Estimation
duration_seconds = len(data) / fs
is_heartbeat = False

if len(peaks) > 1:
    intervals = np.diff(peaks) / fs
    bpm = 60 / np.mean(intervals)
    print(f"Estimated Heart Rate: {bpm:.1f} BPM")

    if 40 < bpm < 180 and len(peaks) >= 3 and duration_seconds >= 1.5:
        is_heartbeat = True
else:
    print("Not enough peaks to estimate BPM.")

# Output classification
if is_heartbeat:
    print("This sound likely contains a heartbeat.")
else:
    print("This sound does NOT resemble a heartbeat.")
