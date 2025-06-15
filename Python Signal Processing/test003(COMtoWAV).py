import serial
import time
import numpy as np
from scipy.io.wavfile import write

#
# Thomas's Arduino COM -> TXT -> WAV conversion script; 8kHz
#


# CONFIGURATION
COM_PORT = 'COM11'           # Replace with your Arduino COM port
BAUD_RATE = 115200
DURATION = 10                # Seconds
TXT_FILE = 'arduino_output9.txt' # Doesn't matter if you change or not - overwrites
WAV_FILE = 'output9.wav'     #CHANGE 
SAMPLE_RATE = 800           # Based on Arduino timing

def record_serial_to_txt():
    print(f"Connecting to {COM_PORT} at {BAUD_RATE} baud...")
    try:
        with serial.Serial(COM_PORT, BAUD_RATE, timeout=1) as ser, open(TXT_FILE, 'w') as file:
            print(f"Recording serial data for {DURATION} seconds...")
            start_time = time.time()
            while time.time() - start_time < DURATION:
                line = ser.readline().decode(errors='ignore').strip()
                if line:
                    file.write(line + '\n')
        print(f"Done. Data saved to '{TXT_FILE}'")
    except serial.SerialException as e:
        print(f"Serial error: {e}")
        return False
    return True

def convert_txt_to_wav():
    print("Converting text data to WAV format...")
    valid_samples = []
    with open(TXT_FILE, 'r') as file:
        for line in file:
            try:
                val = float(line.strip())
                valid_samples.append(val)
            except ValueError:
                continue

    if not valid_samples:
        print("No valid data to convert.")
        return

    samples = np.array(valid_samples)

    # Normalize and scale
    samples = (samples - 512) / 512.0       # Normalize to range [-1, 1]
    samples = np.clip(samples, -1.0, 1.0)   # Prevent out-of-range errors
    scaled = np.int16(samples * 32767)

    write(WAV_FILE, SAMPLE_RATE, scaled)
    print(f"WAV file successfully saved as '{WAV_FILE}'")

if __name__ == "__main__":
    if record_serial_to_txt():
        convert_txt_to_wav()
