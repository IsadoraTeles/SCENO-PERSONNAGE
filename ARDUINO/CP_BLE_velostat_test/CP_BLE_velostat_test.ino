#include <Adafruit_CircuitPlayground.h>
#include <Arduino.h>
#include <SPI.h>
#include <Wire.h>
#include "Adafruit_BLE.h"
#include "Adafruit_BluefruitLE_SPI.h"
#include "Adafruit_BluefruitLE_UART.h"
#include "BluefruitConfig.h"

#define FACTORYRESET_ENABLE         1
#define MINIMUM_FIRMWARE_VERSION    "0.6.6"
#define MODE_LED_BEHAVIOUR          "MODE"

#define VELOSTAT_PIN A10 // Analog input for velostat
#define BLE_BUFFER_SIZE 32 // Adjust the size as needed

float _ewmaAlpha = 0.1;  // the EWMA alpha value (α)
double _ewma = 0;        // the EWMA result (Si), initialized to zero

Adafruit_BluefruitLE_UART ble(BLUEFRUIT_HWSERIAL_NAME, BLUEFRUIT_UART_MODE_PIN);

unsigned long previousMillis = 0;
const long interval = 100; // Interval of 500ms for sending data twice per second

void error(const __FlashStringHelper *err) {
  if (Serial) Serial.println(err);
  while (1) {
    // Add a way to escape this condition if necessary, such as a hardware button press
  }
}

void setup(void) {
  Serial.begin(115200);
  CircuitPlayground.begin();

  if (!ble.begin(VERBOSE_MODE)) {
    error(F("Couldn't find Bluefruit, make sure it's in Command mode & check wiring?"));
  }

  if (FACTORYRESET_ENABLE) {
    if (!ble.factoryReset()) {
      error(F("Couldn't factory reset"));
    }
  }

  ble.echo(false);
  ble.verbose(false);

  while (!ble.isConnected()) {
    delay(500);
  }

  if (ble.isVersionAtLeast(MINIMUM_FIRMWARE_VERSION)) {
    ble.sendCommandCheckOK("AT+HWModeLED=" MODE_LED_BEHAVIOUR);
  }

  ble.setMode(BLUEFRUIT_MODE_DATA);

  pinMode(VELOSTAT_PIN, INPUT_PULLUP);
  for (int i = 0; i < 10; ++i) { // Initialize readings
    int reading = analogRead(VELOSTAT_PIN);
    _ewma = (_ewmaAlpha * reading) + (1 - _ewmaAlpha) * _ewma;
  }

  // Set accelerometer range to ±2g for higher precision
  CircuitPlayground.setAccelRange(LIS3DH_RANGE_2_G);
}

void loop(void) {
  int VALUE = analogRead(VELOSTAT_PIN);
  _ewma = (_ewmaAlpha * VALUE) + (1 - _ewmaAlpha) * _ewma;

  // Check accelerometer data
  float y = CircuitPlayground.motionY();
  float z = CircuitPlayground.motionZ();

  // Map Y and Z values to range 0-100
  int yMapped = constrain(map(y, -2, 2, 0, 100), 0, 100);
  int zMapped = constrain(map(z, -2, 2, 0, 100), 0, 100);

  // Map the EWMA filtered value to 0-100 range
  int velostatValue = map(static_cast<int>(_ewma), 0, 1023, 0, 100);
  velostatValue = constrain(velostatValue, 0, 100);

  // Combine the data into a single integer
  // Assuming yMapped (7 bits), zMapped (7 bits), and velostat value (8 bits)
  int encodedData = (yMapped << 14) | (zMapped << 7) | (velostatValue & 0x7F);

  // Print the encoded data to Serial for debugging
  Serial.print("Encoded Data: ");
  Serial.println(encodedData, HEX); // Print the encoded data as hex

  // Check if it's time to send the value
  unsigned long currentMillis = millis();
  if (currentMillis - previousMillis >= interval) {
    previousMillis = currentMillis;

    // Convert the encoded data to a hexadecimal string
    char buffer[5];
    sprintf(buffer, "%04X", encodedData); // Convert integer to 4-char hex string
    ble.print(buffer);
    Serial.print("Sent Encoded Data: ");
    Serial.println(buffer); // Print the value sent over BLE
  }
}
