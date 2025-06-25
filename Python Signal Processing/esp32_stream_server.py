import os
import time
import threading
import numpy as np
import serial
from flask import Flask, render_template_string, jsonify
import scipy.io.wavfile as wavfile
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.signal import butter, sosfilt, sosfilt_zi, lfilter, lfilter_zi, hilbert, find_peaks

app = Flask(__name__)

# === Config ===
FS = 1000
BUFFER_SECONDS = 5
BUFFER_SIZE = FS * BUFFER_SECONDS
COM_PORT = 'COM12'  # ← Change to your port
BAUD_RATE = 115200
GAIN = 30
FILTER_LOWCUT = 20
FILTER_HIGHCUT = 150
ENV_CUTOFF = 5
MIN_PEAK_SPACING = int(0.6 * FS)

# === Globals ===
serial_data_buffer = np.zeros(BUFFER_SIZE, dtype=np.float32)
recorded_data = []
is_recording = False
record_lock = threading.Lock()

# === Filters ===
sos_band = butter(4, [FILTER_LOWCUT, FILTER_HIGHCUT], btype='band', fs=FS, output='sos')
b_env, a_env = butter(4, ENV_CUTOFF, btype='low', fs=FS)

# === UI Template ===
html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Heartbeat Monitor</title>
    <style>
        body { font-family: Arial; margin: 40px; background: #f0f0f0; }
        h1 { color: #333; }
        button { padding: 10px 20px; margin: 5px; font-size: 16px; }
        #status { font-weight: bold; margin-top: 20px; color: #005500; }
    </style>
</head>
<body>
    <h1>Heartbeat Monitor</h1>
    <button onclick="fetch('/start_graph')">Open Live Graph</button><br>
    <button onclick="startRecording()">Start Recording</button>
    <button onclick="stopRecording()">Stop Recording</button>
    <p id="status">Status: Idle</p>
    <p id="link"></p>
    <script>
    function startRecording() {
        fetch('/start_recording');
        document.getElementById('status').innerText = 'Status: Recording...';
        document.getElementById('status').style.color = 'red';
        document.getElementById('link').innerText = '';
    }

    function stopRecording() {
        fetch('/stop_recording')
        .then(res => res.json())
        .then(data => {
            if (data.file) {
                document.getElementById('status').innerText = 'Status: ' + data.message;
                document.getElementById('status').style.color = 'green';
                document.getElementById('link').innerHTML = '<a href="' + data.file + '" download>Download heartbeat.wav</a>';
            } else {
                document.getElementById('status').innerText = 'Status: No data recorded.';
                document.getElementById('status').style.color = 'orange';
                document.getElementById('link').innerText = '';
            }
        });
    }
</script>

</body>
</html>
"""
html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Electronic Stethoscope</title>
    <style>
        body {
            background-color: #f2f8ff;
            font-family: 'Segoe UI', sans-serif;
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 40px;
            color: #333;
        }
        h1 {
            font-size: 2.5em;
            color: #0055aa;
            margin-bottom: 10px;
        }
        .intro {
            max-width: 600px;
            text-align: center;
            font-size: 1.1em;
            margin-bottom: 40px;
            line-height: 1.6;
        }
        .button-container {
            display: flex;
            flex-direction: column;
            gap: 15px;
            align-items: center;
        }
        button {
            background-color: #007bff;
            color: white;
            border: none;
            padding: 15px 30px;
            font-size: 16px;
            border-radius: 8px;
            cursor: pointer;
            transition: background 0.3s ease;
        }
        button:hover {
            background-color: #005fc2;
        }
        #status {
            margin-top: 30px;
            font-weight: bold;
            font-size: 1.1em;
        }
        #link {
            margin-top: 10px;
            font-size: 1em;
        }
        a {
            color: #007bff;
            text-decoration: none;
            font-weight: bold;
        }
        a:hover {
            text-decoration: underline;
        }
    </style>
</head>
<body>
    <h1>Electronic Stethoscope</h1>
    <div class="intro">
        <p>
            This tool allows you to visualize and record heart sounds using a digital stethoscope.
            <br><br>
            <strong>Instructions:</strong><br>
            - Click <b>“Open Live Graph”</b> to visualize the heartbeat signal in real-time.<br>
            - Click <b>“Start Recording”</b> to begin saving the signal.<br>
            - Click <b>“Stop Recording”</b> to save it as a WAV file and download it.<br>
        </p>
    </div>

    <div class="button-container">
        <button onclick="fetch('/start_graph')">Open Live Graph</button>
        <button onclick="startRecording()">Start Recording</button>
        <button onclick="stopRecording()">Stop Recording</button>
    </div>

    <p id="status">Status: Idle</p>
    <p id="link"></p>

    <script>
        function startRecording() {
            fetch('/start_recording');
            document.getElementById('status').innerText = 'Status: Recording...';
            document.getElementById('status').style.color = 'red';
            document.getElementById('link').innerText = '';
        }

        function stopRecording() {
            fetch('/stop_recording')
            .then(res => res.json())
            .then(data => {
                if (data.file) {
                    document.getElementById('status').innerText = 'Status: ' + data.message;
                    document.getElementById('status').style.color = 'green';
                    document.getElementById('link').innerHTML = '<a href="' + data.file + '" download>Download heartbeat.wav</a>';
                } else {
                    document.getElementById('status').innerText = 'Status: No data recorded.';
                    document.getElementById('status').style.color = 'orange';
                    document.getElementById('link').innerText = '';
                }
            });
        }
    </script>
</body>
</html>
"""

# === Flask Routes ===
@app.route('/')
def index():
    return render_template_string(html_template)

@app.route('/start_graph')
def start_graph():
    threading.Thread(target=plot_heartbeat_graph, daemon=True).start()
    return '', 204

@app.route('/start_recording')
def start_recording():
    global is_recording, recorded_data
    with record_lock:
        recorded_data = []
        is_recording = True
    return '', 204

@app.route('/stop_recording')
def stop_recording():
    global is_recording

    # Step 1: Turn off recording
    with record_lock:
        is_recording = False

    # Step 2: Wait just a moment
    time.sleep(0.2)

    # Step 3: Copy & clear data safely
    with record_lock:
        data_to_save = recorded_data.copy()
        recorded_data.clear()

    if not data_to_save:
        return jsonify({"message": "No data recorded.", "file": ""})

    # Step 4: Save to heartbeat.wav
    filename = "heartbeat.wav"
    if not os.path.exists("static"):
        os.makedirs("static")
    filepath = os.path.join("static", filename)
    wavfile.write(filepath, FS, np.array(data_to_save, dtype=np.int16))

    print(f"[INFO] Saved {len(data_to_save)} samples to {filename}")
    return jsonify({
        "message": "Stopped recording and saved heartbeat.wav",
        "file": "/" + filepath
    })




# === Serial Reader Thread ===
def serial_reader():
    global serial_data_buffer, recorded_data, is_recording
    try:
        ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=0.1)
        print(f"[OK] Connected to {COM_PORT} at {BAUD_RATE}")
    except serial.SerialException as e:
        print(f"[ERROR] Serial error: {e}")
        return

    while True:
        try:
            line = ser.readline().decode('ascii', errors='ignore').strip()
            if line:
                sample = float(line)
                # Maintain rolling buffer for display
                serial_data_buffer[:-1] = serial_data_buffer[1:]
                serial_data_buffer[-1] = sample

                # Always record if flag is on
                with record_lock:
                    if is_recording:
                        # Scale float [-1, 1] to 16-bit integer
                        pcm_sample = int(np.clip(sample * 32767, -32768, 32767))
                        recorded_data.append(pcm_sample)
        except Exception:
            continue


# === Graph Function ===
def plot_heartbeat_graph():
    raw = np.zeros(BUFFER_SIZE)
    filtered = np.zeros(BUFFER_SIZE)
    envelope = np.zeros(BUFFER_SIZE)
    time_axis = np.linspace(0, BUFFER_SECONDS, BUFFER_SIZE)
    zi_band = sosfilt_zi(sos_band)
    zi_env = lfilter_zi(b_env, a_env)
    bpm_history = []

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    plt.subplots_adjust(hspace=0.4)

    line_filt, = ax1.plot(time_axis, filtered, label="Filtered", color='blue')
    ax1.set_title("Filtered Heart Sound")
    ax1.set_ylabel("Amplitude")
    ax1.grid()
    ax1.legend()

    line_env, = ax2.plot(time_axis, envelope, label="Envelope", color='green')
    scatter = ax2.scatter([], [], color='red', s=30, label='Peaks')
    bpm_text = ax2.text(0.02, 0.95, '', transform=ax2.transAxes, bbox=dict(facecolor='white'))
    ax2.set_title("Heartbeat Envelope & BPM")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Intensity")
    ax2.grid()
    ax2.legend()

    def update(frame):
        nonlocal filtered, envelope, zi_band, zi_env, bpm_history
        samples = serial_data_buffer.copy()
        samples -= np.mean(samples)

        filt, zi_band = sosfilt(sos_band, samples, zi=zi_band)
        filt *= GAIN
        filtered = filt

        env = np.abs(hilbert(filt))
        env_smooth, zi_env = lfilter(b_env, a_env, env, zi=zi_env)
        envelope = env_smooth

        line_filt.set_ydata(filtered)
        line_env.set_ydata(envelope)

        max_val = max(np.max(np.abs(filtered)), np.max(envelope)) * 1.5
        ax1.set_ylim(-max_val, max_val)
        ax2.set_ylim(0, max_val)

        min_height = np.max(envelope) * 0.3
        peaks, _ = find_peaks(envelope, distance=MIN_PEAK_SPACING, height=min_height)
        scatter.set_offsets(np.c_[time_axis[peaks], envelope[peaks]])

        if len(peaks) > 1:
            intervals = np.diff(time_axis[peaks])
            bpm = 60 / np.mean(intervals)
            if 40 < bpm < 180:
                bpm_history.append(bpm)
                bpm_history = bpm_history[-5:]
                bpm_avg = np.mean(bpm_history)
                bpm_text.set_text(f"BPM: {bpm_avg:.1f}")

        return line_filt, line_env, scatter, bpm_text

    ani = FuncAnimation(fig, update, blit=True, interval=50)
    try:
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print("[Graph Closed]")

# === App Start ===
if __name__ == '__main__':
    threading.Thread(target=serial_reader, daemon=True).start()
    app.run(debug=False, port=5000)
