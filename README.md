# Digital Wireless Stethoscope - Group 7

## Project Overview

This project focuses on creating a **digital twin** of a medical stethoscope capable of **wireless data transmission**, **real-time visualization**, and **heartbeat analysis**. It enhances traditional auscultation by capturing high-quality heart sounds using a MAX9814 microphone and processing them with Python-based signal tools.

The final system enables:
- Wireless transmission of heartbeat data via ESP32 Wi-Fi.
- Real-time heartbeat visualization with peak detection.
- WAV file recording of stethoscope signals.
- Heartbeat classification (normal/abnormal) from saved audio.

## Project Goals
- Capture heartbeat audio using MAX9814 (50–180 Hz range).
- Amplify, filter, and transmit data wirelessly.
- Allow real-time plotting + WAV file recording.
- Enable heartbeat classification.
- Improve stethoscope design via k-Wave simulations.

## Python Signal Processing Folder

This folder contains all the signal processing and visualization code used to test, debug, and run the digital stethoscope prototype. The code is organized into two main modules — one for **wireless UDP-based data** and another for **serial input**.

### Wireless (Wi-Fi) Module

- **`final_wifi_module.py`**  
  This is the main wireless UDP receiver that communicates with the ESP32 over Wi-Fi.

  **Use with**: `wireless_wifi_same_sample_rate.ino` in the Arduino Code folder.  

### Serial (USB) Module – For Faster Transmission

- **`esp32_stream_server.py`**  
  A serial-based version of the same visualization pipeline, using USB for input. Offers higher sampling reliability (especially for >1kHz rates). 

  **Use with**: `simple_analog_output_higher_sample_rate.ino` in the Arduino Code folder.  

## Heartbeat Model Folder

This folder contains files used to train and test a CNN-based heartbeat classifier that labels sounds as **normal** or **abnormal**.

- Based on: https://github.com/MananAgarwal/Heartbeat-Classifier  
- Dataset: https://www.kaggle.com/datasets/kinguistics/heartbeat-sounds  
  (Place in folders `set_a/` and `set_b/`)  
- Train using the Jupyter notebook `Heartbeat Classifier.ipynb`.  
- Test a file using `python3 test_model.py your_file.wav`.


## Springer Segmentation Python Folder

This folder includes an ML model that segments heartbeats into phases like **S1** and **S2** based on clinical research.

Although explored, this method was not used in the final prototype.  
Instead, a custom signal processing pipeline was developed for better control, real-time integration, and robustness with noisy hardware data.

## Hardware & System Overview

- **Hardware**: ESP32 DevKit, MAX9814 mic, 3D-printed hexagonal chestpiece, balloon membrane, glass wool insulation.
- **Sampling**: 1kHz, sent wirelessly via UDP or over USB serial.
- **Filtering**: 4th-order Butterworth bandpass (50–180 Hz), low-pass envelope filter (cutoff 20 Hz).
- **Peak Detection**: Detects S1/S2 heart sounds, calculates BPM in real time.

## Future Improvements
- Enhance heartbeat classification accuracy.
- Increase noise suppression beyond 50%.
- Expand 3D k-Wave simulation integration.

## Authors
Group 7: V. Dees, O. Hejduk, Q. Nguyen, M. Oh, T. Schoppert, P. Strońska
