# INPUT: COM port, real-time audio input
# OUTPUT: real-time graph output
# Real-time Heart Sound Detection with Single Filtering
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import serial
from scipy.signal import butter, sosfilt, sosfilt_zi, lfilter, lfilter_zi, find_peaks
import sys

# Configuration
COM_PORT = 'COM12'       # Update to your COM port
BAUD_RATE = 115200      # Match device baud rate
FS = 1000               # Sampling rate (Hz)
BUFFER_SECONDS = 2      # Display window (seconds)
FILTER_LOWCUT = 50      # Bandpass lower cutoff (Hz)
FILTER_HIGHCUT = 180    # Bandpass upper cutoff (Hz)
ENV_CUTOFF = 20         # Envelope lowpass cutoff (Hz)
GAIN = 20               # Amplification factor

# Derived parameters
BUFFER_SIZE = int(BUFFER_SECONDS * FS)
MIN_PEAK_SPACING = int(0.1 * FS)  # 0.1 seconds

# Initialize filters
sos_band = butter(4, [FILTER_LOWCUT, FILTER_HIGHCUT], btype='band', fs=FS, output='sos')
b_env, a_env = butter(2, ENV_CUTOFF, btype='low', fs=FS)

# Initialize filter states
zi_band = sosfilt_zi(sos_band) * 0
zi_env = lfilter_zi(b_env, a_env) * 0

# Initialize data buffers
time_buffer = np.linspace(0, BUFFER_SECONDS, BUFFER_SIZE)
raw_data = np.zeros(BUFFER_SIZE)
filtered_data = np.zeros(BUFFER_SIZE)
envelope_data = np.zeros(BUFFER_SIZE)

# Initialize serial connection
try:
    ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=0.1)
    print(f"Connected to {COM_PORT} at {BAUD_RATE} baud")
except serial.SerialException as e:
    print(f"Error opening serial port: {e}")
    sys.exit(1)

# Set up plot
fig, ax = plt.subplots(figsize=(12, 6))
(line_filtered,) = ax.plot(time_buffer, filtered_data, label="Filtered Signal", alpha=0.7)
(line_envelope,) = ax.plot(time_buffer, envelope_data, color='orange', label="Envelope")
scatter_peaks = ax.scatter([], [], color='red', marker='x', label="Detected Peaks")
ax.set_title("Real-time Heart Sound Detection")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Amplitude")
ax.legend()
ax.grid()
ax.set_xlim(0, BUFFER_SECONDS)
ax.set_ylim(-0.5, 0.5)  # Adjust based on your signal range

# Peak detection variables
last_peaks = np.array([])
bpm_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, 
                   bbox=dict(facecolor='white', alpha=0.7))

def update(frame):
    global raw_data, filtered_data, envelope_data, zi_band, zi_env, last_peaks
    
    # Read available serial data
    samples = []
    while ser.in_waiting > 0:
        try:
            line = ser.readline().decode('ascii', errors='ignore').strip()
            if line:
                samples.append(float(line))
        except (ValueError, UnicodeDecodeError):
            continue
    
    if not samples:
        return line_filtered, line_envelope, scatter_peaks, bpm_text
    
    # Process new samples
    new_samples = np.array(samples, dtype=np.float32)
    new_samples -= np.mean(new_samples)  # Remove DC offset
    
    # Update raw data buffer (shift left and append new data)
    n = len(new_samples)
    raw_data[:-n] = raw_data[n:]
    raw_data[-n:] = new_samples
    
    # Apply bandpass filter with state preservation
    filtered_new, zi_band = sosfilt(sos_band, new_samples, zi=zi_band)
    filtered_new *= GAIN  # Apply gain
    
    # Update filtered data buffer
    filtered_data[:-n] = filtered_data[n:]
    filtered_data[-n:] = filtered_new
    
    # Compute envelope (absolute value + lowpass filter)
    env_new = np.abs(filtered_new)
    env_smoothed, zi_env = lfilter(b_env, a_env, env_new, zi=zi_env)
    
    # Update envelope buffer
    envelope_data[:-n] = envelope_data[n:]
    envelope_data[-n:] = env_smoothed
    
    # Update plot data
    line_filtered.set_ydata(filtered_data)
    line_envelope.set_ydata(envelope_data)
    
    # Auto-scale y-axis with margin
    y_margin = 0.1
    y_min = min(np.min(filtered_data), np.min(envelope_data)) - y_margin
    y_max = max(np.max(filtered_data), np.max(envelope_data)) + y_margin
    ax.set_ylim(y_min, y_max)
    
    # Peak detection on the full envelope buffer
    threshold = np.percentile(envelope_data, 95) * 0.4
    peaks, _ = find_peaks(envelope_data, 
                          distance=MIN_PEAK_SPACING, 
                          height=threshold)
    last_peaks = peaks
    
    # Update peak markers
    scatter_peaks.set_offsets(np.column_stack((time_buffer[peaks], envelope_data[peaks])))
    
    # Calculate and display BPM if valid peaks
    if len(peaks) > 1:
        intervals = np.diff(time_buffer[peaks])
        bpm = 60 / np.mean(intervals)
        bpm_text.set_text(f"Estimated BPM: {bpm:.1f}")
        
    return line_filtered, line_envelope, scatter_peaks, bpm_text

# Start animation
ani = FuncAnimation(fig, update, blit=True, interval=50, cache_frame_data=False)
plt.tight_layout()
plt.show()

# Cleanup
ser.close()
print("Serial connection closed")