import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import serial
from scipy.signal import butter, sosfilt, sosfilt_zi, lfilter, lfilter_zi, find_peaks, hilbert
import sys

# Configuration
COM_PORT = 'COM12'       # Update to your COM port
BAUD_RATE = 115200      # Match device baud rate
FS = 1000               # Sampling rate (Hz)
BUFFER_SECONDS = 3      # Increased display window
FILTER_LOWCUT = 20      # Lower cutoff to capture more low-frequency components
FILTER_HIGHCUT = 150    # Lower high cutoff for cleaner signal
ENV_CUTOFF = 5          # Lower envelope cutoff for smoother output
GAIN = 30               # Increased amplification

# Derived parameters
BUFFER_SIZE = int(BUFFER_SECONDS * FS)
MIN_PEAK_SPACING = int(0.1 * FS)  # 0.1 seconds

# Initialize filters
sos_band = butter(4, [FILTER_LOWCUT, FILTER_HIGHCUT], btype='band', fs=FS, output='sos')
b_env, a_env = butter(4, ENV_CUTOFF, btype='low', fs=FS)  # Higher order for smoother envelope

# Initialize filter states
zi_band = sosfilt_zi(sos_band) * 0
zi_env = lfilter_zi(b_env, a_env) * 0

# Initialize data buffers
time_buffer = np.linspace(0, BUFFER_SECONDS, BUFFER_SIZE)
raw_data = np.zeros(BUFFER_SIZE)
filtered_data = np.zeros(BUFFER_SIZE)
envelope_data = np.zeros(BUFFER_SIZE)
analytic_signal = np.zeros(BUFFER_SIZE)  # For Hilbert transform

# Initialize serial connection
try:
    ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=0.1)
    print(f"Connected to {COM_PORT} at {BAUD_RATE} baud")
except serial.SerialException as e:
    print(f"Error opening serial port: {e}")
    sys.exit(1)

# Set up plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
plt.subplots_adjust(hspace=0.4)

# Top plot: Raw filtered signal
(line_filtered,) = ax1.plot(time_buffer, filtered_data, label="Filtered Signal", alpha=0.7, color='blue')
ax1.set_title("Raw Heart Sound Signal")
ax1.set_ylabel("Amplitude")
ax1.legend()
ax1.grid()

# Bottom plot: Processed signal for visualization
(line_envelope,) = ax2.plot(time_buffer, envelope_data, color='green', label="Processed Signal")
scatter_peaks = ax2.scatter([], [], color='red', marker='o', s=40, label="Detected Peaks")
ax2.set_title("Heartbeat Visualization")
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Intensity")
ax2.legend()
ax2.grid()

# Text displays
bpm_text = ax2.text(0.02, 0.95, '', transform=ax2.transAxes, 
                   bbox=dict(facecolor='white', alpha=0.8))
hrv_text = ax2.text(0.02, 0.85, '', transform=ax2.transAxes, 
                   bbox=dict(facecolor='white', alpha=0.8))

# Peak detection variables
last_peaks = np.array([])
peak_history = []
bpm_history = []
SMOOTHING_FACTOR = 0.1
current_yrange = 0.5

def update(frame):
    global raw_data, filtered_data, envelope_data, zi_band, zi_env, last_peaks
    global peak_history, bpm_history, current_yrange
    
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
        return line_filtered, line_envelope, scatter_peaks, bpm_text, hrv_text
    
    # Process new samples
    new_samples = np.array(samples, dtype=np.float32)
    new_samples -= np.mean(new_samples)  # Remove DC offset
    
    # Update raw data buffer
    n = len(new_samples)
    raw_data[:-n] = raw_data[n:]
    raw_data[-n:] = new_samples
    
    # Apply bandpass filter
    filtered_new, zi_band = sosfilt(sos_band, new_samples, zi=zi_band)
    filtered_new *= GAIN
    
    # Update filtered data buffer
    filtered_data[:-n] = filtered_data[n:]
    filtered_data[-n:] = filtered_new
    
    # Compute envelope using Hilbert transform for better shape
    analytic_signal_new = hilbert(filtered_new)
    env_new = np.abs(analytic_signal_new)
    
    # Apply lowpass filter to envelope
    env_smoothed, zi_env = lfilter(b_env, a_env, env_new, zi=zi_env)
    
    # Update envelope buffer
    envelope_data[:-n] = envelope_data[n:]
    envelope_data[-n:] = env_smoothed
    
    # Update plot data
    line_filtered.set_ydata(filtered_data)
    line_envelope.set_ydata(envelope_data)
    
    # Auto-scale y-axis with stabilization
    current_max = max(np.max(np.abs(filtered_data)), np.max(envelope_data))
    target_yrange = current_max * 1.5
    current_yrange = (1 - SMOOTHING_FACTOR) * current_yrange + SMOOTHING_FACTOR * target_yrange
    ax1.set_ylim(-current_yrange, current_yrange)
    ax2.set_ylim(0, current_yrange)
    
    # Peak detection - more robust algorithm
    threshold = np.percentile(envelope_data, 90) * 0.5  # Higher threshold
    min_height = np.max(envelope_data) * 0.3
    peaks, properties = find_peaks(envelope_data, 
                                  distance=MIN_PEAK_SPACING, 
                                  height=min_height,
                                  prominence=min_height*0.5,
                                  width=MIN_PEAK_SPACING/10)
    last_peaks = peaks
    
    # Update peak markers
    scatter_peaks.set_offsets(np.column_stack((time_buffer[peaks], envelope_data[peaks])))
    
    # Calculate and display BPM with smoothing
    current_bpm = None
    if len(peaks) > 1:
        intervals = np.diff(time_buffer[peaks])
        current_bpm = 60 / np.mean(intervals)
        
        # Store peaks for HRV calculation
        peak_history.extend(time_buffer[peaks].tolist())
        if len(peak_history) > 20:  # Keep last 20 peaks
            peak_history = peak_history[-20:]
            
        # Smooth BPM calculation
        bpm_history.append(current_bpm)
        if len(bpm_history) > 5:  # Keep last 5 readings
            bpm_history = bpm_history[-5:]
        smoothed_bpm = np.mean(bpm_history)
        
        bpm_text.set_text(f"BPM: {smoothed_bpm:.1f}")
        
        # Calculate HRV (Heart Rate Variability)
        if len(peak_history) > 2:
            rr_intervals = np.diff(peak_history)
            rmssd = np.sqrt(np.mean(np.square(np.diff(rr_intervals))))
            hrv_text.set_text(f"HRV: {rmssd*1000:.1f} ms")
    
    return line_filtered, line_envelope, scatter_peaks, bpm_text, hrv_text

# Start animation
ani = FuncAnimation(fig, update, blit=True, interval=50, cache_frame_data=False)
plt.tight_layout()
plt.show()

# Cleanup
ser.close()
print("Serial connection closed")