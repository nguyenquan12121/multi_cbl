// Sample interval adjustable

const int micPin = A0;
const unsigned long sampleIntervalMicros = 1000; // Target 1kHz sample rate

void setup() {
  Serial.begin(115200);  // MUCH faster communication
}

void loop() {
  static unsigned long lastSampleTime = 0;
  if (micros() - lastSampleTime >= sampleIntervalMicros) {
    int sample = analogRead(micPin);
    Serial.println(sample);  // Just output the number for easier parsing
    lastSampleTime = micros();
  }
}
