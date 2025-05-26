import serial
import numpy as np
from scipy.io.wavfile import write

SERIAL_PORT = 'COM11'  # Adjust for your OS
BAUD_RATE = 115200
DURATION = 5  # seconds
FS = 1000  # Sampling frequency
NUM_SAMPLES = FS * DURATION

ser = serial.Serial(SERIAL_PORT, BAUD_RATE)
data = []

print("Recording...")
while len(data) < NUM_SAMPLES:
    try:
        line = ser.readline().decode().strip()
        value = int(line)
        data.append(value)
    except:
        continue

ser.close()

# Normalize & convert to 16-bit PCM
data = np.array(data)
data = data - np.mean(data)
data = np.int16(data / np.max(np.abs(data)) * 32767)

write("recorded_heartbeat2.wav", FS, data)
print("Saved to recorded_heartbeat.wav")
