import serial
import numpy as np
from scipy.signal import butter, lfilter, find_peaks
import matplotlib.pyplot as plt
import time

#
# test001: order 2 butterworth bandpass filter. Used with lfilter.
# input: 2 bytes per sample, 1 byte for each channel (arduino)
# 

PORT = 'COM11'  
BAUD = 115200
SAMPLE_RATE = 1000  # Hz
DURATION = 10       # seconds
CHUNK_SIZE = 1000   # bytes to read per loop (500 samples since 2 bytes/sample)

# bandapass filter
def butter_bandpass(lowcut, highcut, fs, order=2):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    return butter(order, [low, high], btype='band')

# lowpass filter
def butter_lowpass(cutoff, fs, order=2):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    return butter(order, normal_cutoff, btype='low')


ser = serial.Serial(PORT, BAUD, timeout=1)
time.sleep(2)  # Wait for Arduino reset

print("Recording...")

raw_data = []
start_time = time.time()

while time.time() - start_time < DURATION:
    data = ser.read(CHUNK_SIZE)
    if len(data) % 2 != 0:
        continue  # Ensure full samples
    raw = np.frombuffer(data, dtype=np.uint8)
    samples = raw[0::2] + (raw[1::2] << 8)
    raw_data.extend(samples)

ser.close()
print("Done.")

# center and normalize
raw_data = np.array(raw_data, dtype=np.int16)
raw_data -= 512  # center around 0, values from 0 to 1023

# bandpass filter
b_band, a_band = butter_bandpass(30, 80, SAMPLE_RATE, order=2)
filtered = lfilter(b_band, a_band, raw_data)

# envelope detection
envelope = np.abs(filtered)
b_env, a_env = butter_lowpass(5, SAMPLE_RATE, order=2)  # Smooth envelope
env_smoothed = lfilter(b_env, a_env, envelope)

# detect peaks
min_spacing_samples = int(0.5 * SAMPLE_RATE)  # 0.5 seconds between peaks
peak_threshold = np.max(env_smoothed) * 0.4    # Only strong peaks


peaks, _ = find_peaks(env_smoothed, distance=min_spacing_samples, height=peak_threshold)

# plot
t = np.arange(len(filtered)) / SAMPLE_RATE

plt.figure(figsize=(12, 6))
plt.plot(t, filtered, label="Filtered Signal (20–60 Hz)", alpha=0.7)
plt.plot(t, env_smoothed, label="Envelope", color='orange')
plt.plot(t[peaks], env_smoothed[peaks], 'rx', label="Detected S1 Peaks")
plt.title("Heart Sound Detection – S1 Beats Only")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# bpm
if len(peaks) > 1:
    intervals = np.diff(peaks) / SAMPLE_RATE
    bpm = 60 / np.mean(intervals)
    print(f"Estimated Heart Rate (S1-based): {bpm:.1f} BPM")
else:
    print("Not enough S1 peaks detected to estimate BPM.")
