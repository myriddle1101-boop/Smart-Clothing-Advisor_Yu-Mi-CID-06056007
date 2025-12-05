#include <WiFi.h>
#include <WebServer.h>
#include <ESP32Servo.h> 


const int servoPin = 14;      
const int centerAngle = 90;   
const int thumbsUpAngle = 180; 
const int thumbsDownAngle = 0; 

Servo servoMotor;


const char* ssid = "riddle_iPhone";
const char* password = "my12345678";


WebServer server(80);

void setup() {
  Serial.begin(115200);
  
  servoMotor.attach(servoPin);
  servoMotor.write(centerAngle); 

  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("");
  Serial.print("WiFi connected. IP address: ");
  Serial.println(WiFi.localIP());

  server.on("/servo", HTTP_GET, handleServoCommand);
  server.begin();
}

void loop() {
  server.handleClient();
}

void handleServoCommand() {
  if (server.hasArg("status")) {
    int status = server.arg("status").toInt();
    
    int targetAngle = centerAngle;
    
    switch (status) {
      case 1: 
        targetAngle = thumbsUpAngle;
        Serial.println("Received: THUMBS UP (Warm Enough)");
        break;
      case 0: 
        targetAngle = thumbsDownAngle;
        Serial.println("Received: THUMBS DOWN (Too Cold)");
        break;
      case 2: 
      default:
        targetAngle = centerAngle;
        Serial.println("Received: CENTER (Unknown)");
        break;
    }
    
    servoMotor.write(targetAngle);
    server.send(200, "text/plain", "OK");
  } else {
    server.send(400, "text/plain", "Error: Missing status parameter");
  }
}