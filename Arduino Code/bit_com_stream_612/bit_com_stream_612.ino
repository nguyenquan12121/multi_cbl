// Higher sample rate using transmission with bytes instead of ASCII values

const int micPin = A0;
const unsigned long intervalMicros = 125; // 8 kHz sampling
unsigned long lastMicros = 0;

void setup() {
  Serial.begin(115200);
}

void loop() {
  if (micros() - lastMicros >= intervalMicros) {
    lastMicros = micros();

    int sample = analogRead(micPin); // 0–1023 (10-bit)
    byte highByte = (sample >> 8) & 0xFF;
    byte lowByte = sample & 0xFF;

    Serial.write(highByte);
    Serial.write(lowByte);
  }
}
