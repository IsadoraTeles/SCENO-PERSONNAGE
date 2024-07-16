#include "Adafruit_Crickit.h"
#include "seesaw_servo.h"
#include <ESP8266WiFi.h>
#include <PubSubClient.h>

// WiFi credentials
#define WLAN_SSID       "OLI"
#define WLAN_PASS       "bisbis2224"

// MQTT broker address and port
const char broker[] = "192.168.1.138";
int port = 1883;
const char* topics[] = { "servo1", "servo2", "servo3" };

// Crickit and servo definitions
#define NUM_SERVOS 3
Adafruit_Crickit crickit;
seesaw_Servo servos[] = { seesaw_Servo(&crickit), 
                          seesaw_Servo(&crickit),
                          seesaw_Servo(&crickit)};
int servoPins[] = { CRICKIT_SERVO1, CRICKIT_SERVO2, CRICKIT_SERVO3 };

// MQTT client
WiFiClient espClient;
PubSubClient mqttClient(espClient);

void setup() {
  Serial.begin(9600);

  // Connect to WiFi
  Serial.println(); Serial.println();
  Serial.print("Connecting to ");
  Serial.println(WLAN_SSID);
  WiFi.begin(WLAN_SSID, WLAN_PASS);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println();
  Serial.println("WiFi connected");
  Serial.println("IP address: "); Serial.println(WiFi.localIP());

  // Connect to MQTT broker
  mqttClient.setServer(broker, port);
  mqttClient.setCallback(messageReceived);
  MQTT_connect();

  // Initialize Crickit and servos
  if (!crickit.begin()) {
    Serial.println("Error starting Crickit!");
    while (1);
  } else {
    Serial.println("Crickit started");
  }
  for (int i = 0; i < NUM_SERVOS; i++) {
    servos[i].attach(servoPins[i]);
  }
  servos[0].write(90); // Ensure servo1 starts in a stopped state
  servos[1].write(90); // Ensure servo2 starts in a stopped state
  servos[2].write(85); // Set servo3 to 85 degrees initially
  Serial.println("Setup complete");

  // Subscribe to MQTT topics
  for (int i = 0; i < NUM_SERVOS; i++) {
    mqttClient.subscribe(topics[i]);
    Serial.print("Subscribed to topic: ");
    Serial.println(topics[i]);
  }
}

void loop() {
  if (!mqttClient.connected()) {
    MQTT_connect();
  }
  mqttClient.loop();

  // Check for capacitive touch to stop all motors
  if (crickit.touchRead(1) > 500) {
    stopAllMotors();
  }
}

void MQTT_connect() {
  while (!mqttClient.connected()) {
    Serial.print("Attempting MQTT connection...");
    if (mqttClient.connect("ESP8266Client")) {
      Serial.println("connected");
      for (int i = 0; i < NUM_SERVOS; i++) {
        mqttClient.subscribe(topics[i]);
        Serial.print("Subscribed to topic: ");
        Serial.println(topics[i]);
      }
    } else {
      Serial.print("failed, rc=");
      Serial.print(mqttClient.state());
      Serial.println(" try again in 5 seconds");
      delay(5000);
    }
  }
}

void messageReceived(char* topic, byte* payload, unsigned int length) {
  payload[length] = '\0'; // Null-terminate the message
  String message = String((char*)payload);

  Serial.print("Message arrived [");
  Serial.print(topic);
  Serial.print("] ");
  Serial.println(message);

  int servoIndex = -1;
  for (int i = 0; i < NUM_SERVOS; i++) {
    if (String(topic) == String(topics[i])) {
      servoIndex = i;
      break;
    }
  }
  if (servoIndex == -1) return;

  int angle = message.toInt();

  Serial.print("Received angle for ");
  Serial.print(topic);
  Serial.print(": ");
  Serial.println(angle);

  controlMotor(servoIndex, angle);
}

void controlMotor(int servoIndex, int angle) {
  if (angle < 0 || angle > 180) {
    Serial.print("Invalid angle for motor ");
    Serial.println(servoIndex);
    return;
  }

  servos[servoIndex].write(angle);
  Serial.print("Motor ");
  Serial.print(servoIndex);
  Serial.print(" set to angle ");
  Serial.println(angle);

  delay(1000); // Rotate for 1 second
  stopMotor(servoIndex);
}

void stopMotor(int servoIndex) {
  servos[servoIndex].write(90); // Stop the motor
  Serial.print("Stopped motor ");
  Serial.println(servoIndex);
}

void stopAllMotors() {
  for (int i = 0; i < NUM_SERVOS; i++) {
    stopMotor(i);
  }
}
