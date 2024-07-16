#include <ESP8266WiFi.h>
#include <PubSubClient.h>

// WiFi credentials.
#define WLAN_SSID       "Isa"
#define WLAN_PASS       "242424ii"

// MQTT broker address and port.
const char broker[] = "192.168.223.219";
int        port     = 1883;

// MQTT topics.
const char topicSonar[] = "sonar";
const char topicFSR[] = "fsr";

// Timing intervals.
const long intervalFSR = 100; // FSR should check ten times per second
const long intervalSonar = 100; // Sonar should check once per second

unsigned long previousMillisFSR = 0;
unsigned long previousMillisSonar = 0;

// Sonar pins and variables.
const int trigPin = 13;
const int echoPin = 16;
const float cal_factor = 0.0343; // Adjusted calibration factor

int lastSonar = -1;

// Flags to indicate if a zero value has been sent.
bool zeroSentFSR = false;
bool zeroSentSonar = false;

// FSR pin and variables.
#define FSR_PIN A0
int lastFSR = -1;

WiFiClient espClient;
PubSubClient mqttClient(espClient);

void setup() 
{
  Serial.begin(9600);

  pinMode(trigPin, OUTPUT);
  pinMode(echoPin, INPUT);

  // Connect to WiFi network.
  WiFi.begin(WLAN_SSID, WLAN_PASS);
  while (WiFi.status() != WL_CONNECTED) 
  {
    delay(500);
    Serial.print(".");
  }

  // Once connected to WiFi, connect to the MQTT broker.
  mqttClient.setServer(broker, port);
  MQTT_connect();
}

void loop() 
{
  if (!mqttClient.connected()) 
  {
    MQTT_connect();
  }
  mqttClient.loop();

  unsigned long currentMillis = millis();

  // FSR reading and publishing.
  if (currentMillis - previousMillisFSR >= intervalFSR) 
  {
    previousMillisFSR = currentMillis;
    int fsrReading = analogRead(FSR_PIN);
    
    if (abs(fsrReading - lastFSR) >= 5) 
    {
      publishFSR(fsrReading);
      lastFSR = fsrReading;
    }
  }

  // Sonar reading and publishing.
  if (currentMillis - previousMillisSonar >= intervalSonar) 
  {
    previousMillisSonar = currentMillis;

    int distance = getSonarDistance();
    
    if (abs(distance - lastSonar) >= 3 && distance < 200) 
    {
      publishSonar(distance);
      lastSonar = distance; // Update the last sonar value sent.
    }
  }
}

void MQTT_connect() 
{
  while (!mqttClient.connected()) 
  {
    Serial.print("Attempting MQTT connection...");
    if (mqttClient.connect("ESP8266Client")) 
    {
      Serial.println("connected");
    } 
    else 
    {
      Serial.print("failed, rc=");
      Serial.print(mqttClient.state());
      Serial.println(" try again in 5 seconds");
      delay(5000);
    }
  }
}

void publishSonar(int sonardata) 
{
  Serial.print("Sonar data: ");
  Serial.println(sonardata);
  mqttClient.publish(topicSonar, String(sonardata).c_str());
}

void publishFSR(int fsrdata) 
{
  Serial.print("FSR data: ");
  Serial.println(fsrdata);
  mqttClient.publish(topicFSR, String(fsrdata).c_str());
}

int getSonarDistance() 
{
  long durations[5];
  for (int i = 0; i < 5; i++) 
  {
    digitalWrite(trigPin, LOW);
    delayMicroseconds(2);
    digitalWrite(trigPin, HIGH);
    delayMicroseconds(10);
    digitalWrite(trigPin, LOW);
    
    durations[i] = pulseIn(echoPin, HIGH);
  }

  // Sort the durations array and return the median value
  sortArray(durations, 5);
  long duration = durations[2]; // median value

  return int(duration / 2 * cal_factor);
}

void sortArray(long arr[], int size) 
{
  for (int i = 0; i < size - 1; i++) 
  {
    for (int j = 0; j < size - i - 1; j++) 
    {
      if (arr[j] > arr[j + 1]) 
      {
        long temp = arr[j];
        arr[j] = arr[j + 1];
        arr[j + 1] = temp;
      }
    }
  }
}
