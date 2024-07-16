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
const char* topics[] = { "servo1", "servo2", "servo3", "servo4" };

// Crickit and servo definitions
#define NUM_SERVOS 4
Adafruit_Crickit crickit;
seesaw_Servo servos[] = { seesaw_Servo(&crickit), 
                          seesaw_Servo(&crickit),
                          seesaw_Servo(&crickit),
                          seesaw_Servo(&crickit)};
int servoPins[] = { CRICKIT_SERVO1, CRICKIT_SERVO2, CRICKIT_SERVO3, CRICKIT_SERVO4 };

// Movement control
const unsigned long maxRotationTime = 10000; // 10 seconds rotation time
unsigned long remainingTimeUp[NUM_SERVOS] = {maxRotationTime, maxRotationTime, maxRotationTime, maxRotationTime};
unsigned long remainingTimeDown[NUM_SERVOS] = {0, 0, 0, 0}; // Start with 0 seconds for downward movement
bool isMoving[NUM_SERVOS] = {false, false, false, false};
unsigned long movementStartTime[NUM_SERVOS] = {0, 0, 0, 0};
char currentDirection[NUM_SERVOS] = {'S', 'S', 'S', 'S'}; // 'S' for Stop

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
    servos[i].write(85); // Ensure all servos start in a stopped state
  }
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
  for (int i = 0; i < 8; i++) {
    if (crickit.touchRead(i) > 500) {
      stopAllMotors();
      break;
    }
  }

  // Check the rotation time limit for each motor
  unsigned long currentMillis = millis();
  for (int i = 0; i < NUM_SERVOS; i++) {
    if (isMoving[i]) {
      unsigned long elapsedTime = currentMillis - movementStartTime[i];
      if (currentDirection[i] == 'U') {
        if (elapsedTime >= remainingTimeUp[i]) {
          stopMotor(i);
          remainingTimeDown[i] += remainingTimeUp[i]; // Regain time on the other direction
          remainingTimeDown[i] = min(remainingTimeDown[i], maxRotationTime); // Cap to maxRotationTime
          remainingTimeUp[i] = 0; // No time left for up rotation
          Serial.print("Stopped motor ");
          Serial.print(i);
          Serial.println(" after max up rotation duration.");
        } else {
          remainingTimeUp[i] -= elapsedTime;
          remainingTimeDown[i] += elapsedTime; // Add elapsed time to down direction
          remainingTimeDown[i] = min(remainingTimeDown[i], maxRotationTime); // Cap to maxRotationTime
          movementStartTime[i] = currentMillis;
        }
      } else if (currentDirection[i] == 'D') {
        if (elapsedTime >= remainingTimeDown[i]) {
          stopMotor(i);
          remainingTimeUp[i] += remainingTimeDown[i]; // Regain time on the other direction
          remainingTimeUp[i] = min(remainingTimeUp[i], maxRotationTime); // Cap to maxRotationTime
          remainingTimeDown[i] = 0; // No time left for down rotation
          Serial.print("Stopped motor ");
          Serial.print(i);
          Serial.println(" after max down rotation duration.");
        } else {
          remainingTimeDown[i] -= elapsedTime;
          remainingTimeUp[i] += elapsedTime; // Add elapsed time to up direction
          remainingTimeUp[i] = min(remainingTimeUp[i], maxRotationTime); // Cap to maxRotationTime
          movementStartTime[i] = currentMillis;
        }
      }
    }
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

  char direction = message.charAt(0);

  Serial.print("Received message for ");
  Serial.print(topic);
  Serial.print(": ");
  Serial.println(message);

  controlMotor(servoIndex, direction);
}

void controlMotor(int servoIndex, char direction) {
  unsigned long currentMillis = millis();

  // If currently moving, stop the motor and update remaining time
  if (isMoving[servoIndex]) {
    unsigned long elapsedTime = currentMillis - movementStartTime[servoIndex];
    if (currentDirection[servoIndex] == 'U') {
      remainingTimeUp[servoIndex] -= elapsedTime;
      remainingTimeDown[servoIndex] += elapsedTime;
    } else if (currentDirection[servoIndex] == 'D') {
      remainingTimeDown[servoIndex] -= elapsedTime;
      remainingTimeUp[servoIndex] += elapsedTime;
    }
    stopMotor(servoIndex);
  }

  // Ensure the direction is valid and there is remaining time for the requested direction
  if (direction == 'U' && remainingTimeUp[servoIndex] == 0) {
    Serial.print("Cannot rotate motor ");
    Serial.print(servoIndex);
    Serial.println(" up, max rotation time reached.");
    return;
  }
  if (direction == 'D' && remainingTimeDown[servoIndex] == 0) {
    Serial.print("Cannot rotate motor ");
    Serial.print(servoIndex);
    Serial.println(" down, max rotation time reached.");
    return;
  }

  int angle = 85; // Initialize to stop position
  if (direction == 'U') {
    angle = 60; // Adjust for upward rotation
  } else if (direction == 'D') {
    angle = 120; // Adjust for downward rotation
  }

  // Update motor speed and direction
  servos[servoIndex].write(angle);
  isMoving[servoIndex] = true;
  movementStartTime[servoIndex] = currentMillis;
  currentDirection[servoIndex] = direction;

  Serial.print("Motor ");
  Serial.print(servoIndex);
  Serial.print(" rotating ");
  Serial.print(direction == 'U' ? "up" : "down");
  Serial.print(" with angle ");
  Serial.println(angle);
}

void stopMotor(int servoIndex) {
  servos[servoIndex].write(85);
  isMoving[servoIndex] = false;
  currentDirection[servoIndex] = 'S';
  Serial.print("Stopped motor ");
  Serial.println(servoIndex);
}

void stopAllMotors() {
  for (int i = 0; i < NUM_SERVOS; i++) {
    stopMotor(i);
  }
}
