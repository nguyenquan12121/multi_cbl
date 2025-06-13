import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import butter, filtfilt, lfilter, find_peaks, sosfilt

#
# test002: order 4 butterworth bandpass filter. Used with sosfilt instead of lfilter, since 
# lfilter is not stable for high order filters (proved to be ok for <4).
#

# changed lfilt to filtfilt


# Butterworth bandpass filter
# Butterworht used for smooth filtering, no peaks 
# order means how sharply the fiklter cuts off the frequencies, low = less sharp
def butter_bandpass(lowcut, highcut, fs, order):
    return butter(order, [lowcut, highcut], btype='band', fs=fs, output='sos')



# envelope smoothing filter
# Butterworth used for smooth filtering, no peaks
# envelope is used to detect the peaks (S1 and S2), taking the absolute value of the signal
def butter_lowpass(cutoff, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    return butter(order, normal_cutoff, btype='low')



# load the audio file
filename = r'C:\Users\oliwi\OneDrive\Desktop\Q4\CBL Digital Twin\github_repo\multi_cbl\Python Signal Processing\output7.wav'
fs, data = wavfile.read(filename) # fs read from a header, data is the signal

# ---------- Preprocess ----------
# convert stereo to mono if needed
if len(data.shape) > 1:
    data = data[:, 0]

# Normalize and center
data = data.astype(np.float32)
data -= np.mean(data)

# bandpass filter 
sos = butter_bandpass(50, 180, fs, order=4)
filtered = sosfilt(sos, data)
# software gain (simulating analog amplification)
filtered *= 20 # 20x gain 

# envelope detection
envelope = np.abs(filtered)
b_env, a_env = butter_lowpass(20, fs, order=2)
env_smoothed = filtfilt(b_env, a_env, envelope)

# peak detection 
min_spacing_samples = int(0.1 * fs) # 0.1 seconds between peaks	
peak_threshold = np.max(env_smoothed) * 0.4
peaks, _ = find_peaks(env_smoothed, distance=min_spacing_samples, height=peak_threshold)




# plot
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

# bpm
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

# output classification
if is_heartbeat:
    print("This sound likely contains a heartbeat.")
else:
    print("This sound does NOT resemble a heartbeat.")
