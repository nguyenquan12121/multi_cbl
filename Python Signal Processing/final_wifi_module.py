import os
import time
import threading
import numpy as np
import socket
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
UDP_PORT = 4210  # Listening port
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

# === HTML Template ===
html_template = """<!DOCTYPE html>
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
    with record_lock:
        is_recording = False
    time.sleep(0.2)
    with record_lock:
        data_to_save = recorded_data.copy()
        recorded_data.clear()

    if not data_to_save:
        return jsonify({"message": "No data recorded.", "file": ""})

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

# === UDP Receiver ===
def udp_receiver():
    global serial_data_buffer, recorded_data, is_recording

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("", UDP_PORT))

    print(f"[OK] Listening for UDP packets on port {UDP_PORT}")

    while True:
        try:
            data, _ = sock.recvfrom(1024)
            line = data.decode('utf-8').strip()
            if line:
                sample = float(line)
                serial_data_buffer[:-1] = serial_data_buffer[1:]
                serial_data_buffer[-1] = sample

                with record_lock:
                    if is_recording:
                        # Scale ESP32 12-bit ADC (0–4095) to 16-bit WAV
                        pcm_sample = int(np.clip(sample * 32767 / 4095, -32768, 32767))
                        recorded_data.append(pcm_sample)
        except Exception as e:
            print(f"[ERROR] UDP receive: {e}")
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
    threading.Thread(target=udp_receiver, daemon=True).start()
    app.run(debug=False, port=5000)
