#include <WiFi.h>
#include <WiFiUdp.h>

const char* ssid = "ssid";
const char* password = "passwd";

const char* udpAddress = "<ip_address>";  // IP of the receiver laptop
const int udpPort = 4210;

const int micPin = A0;
const unsigned long sampleIntervalMicros = 1000; 

WiFiUDP udp;

void setup() {
  Serial.begin(115200);
  WiFi.begin(ssid, password);
  
  while (WiFi.status() != WL_CONNECTED) {
    delay(1000);
    Serial.println("Connecting to WiFi...");
  }

  Serial.println("WiFi connected!");
  udp.begin(udpPort);  
}

void loop() {
  static unsigned long lastSampleTime = 0;
  if (micros() - lastSampleTime >= sampleIntervalMicros) {
    int sample = analogRead(micPin);

    char buffer[8];
    itoa(sample, buffer, 10);
    udp.beginPacket(udpAddress, udpPort);
    udp.write((uint8_t*)buffer, strlen(buffer));
    udp.endPacket();

    lastSampleTime = micros();
  }
}
