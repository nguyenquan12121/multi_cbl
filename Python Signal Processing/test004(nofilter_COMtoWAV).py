import serial
import time
import numpy as np
from scipy.io.wavfile import write

#
## 8kHZ sampling rate, sort of chipmunk voice because this code is parsing
## raw bytes instead of ASCII values
#

# SETTINGS
COM_PORT = 'COM11'           # Replace with your Arduino port
BAUD_RATE = 115200
DURATION = 10                # Recording time in seconds
SAMPLE_RATE = 8000           # Must match Arduino (125 µs interval)
WAV_FILE = 'voice_output.wav'

def record_audio():
    print(f"Connecting to {COM_PORT} at {BAUD_RATE} baud...")
    try:
        with serial.Serial(COM_PORT, BAUD_RATE, timeout=1) as ser:
            samples = []
            print(f"Recording for {DURATION} seconds at {SAMPLE_RATE} Hz...")

            start_time = time.time()
            while time.time() - start_time < DURATION:
                if ser.in_waiting >= 2:
                    high_byte = ser.read(1)
                    low_byte = ser.read(1)

                    if high_byte and low_byte:
                        sample = (high_byte[0] << 8) | low_byte[0]
                        samples.append(sample)

            print(f"Recording finished. Collected {len(samples)} samples.")

    except serial.SerialException as e:
        print(f"Serial error: {e}")
        return

    if not samples:
        print("No samples received.")
        return

    # Normalize to [-1, 1] and convert to 16-bit PCM
    samples_np = np.array(samples, dtype=np.float32)
    samples_np = (samples_np - 512) / 512.0
    samples_np = np.clip(samples_np, -1.0, 1.0)
    int16_samples = np.int16(samples_np * 32767)

    write(WAV_FILE, SAMPLE_RATE, int16_samples)
    print(f"WAV file saved as '{WAV_FILE}'")

if __name__ == "__main__":
    record_audio()
