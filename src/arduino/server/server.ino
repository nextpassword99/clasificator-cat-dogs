#include <ESP8266WiFi.h>
#include <ESP8266WebServer.h>

const char *ssid = "System32";
const char *pass = "edisonp21";

ESP8266WebServer server(80);

const int ledDog = 2;
const int ledCat = 4;

void setup()
{
    Serial.begin(115200);
    WiFi.begin(ssid, pass);

    pinMode(ledDog, OUTPUT);
    pinMode(ledCat, OUTPUT);

    digitalWrite(ledDog, LOW);
    digitalWrite(ledCat, LOW);

    while (WiFi.status() != WL_CONNECTED)
    {
        delay(1000);
        Serial.println("Conectando...");
    }

    Serial.println(WiFi.localIP());

    server.on("/led", HTTP_GET, handleRequest);

    server.begin();
}

void loop()
{
    server.handleClient();
}

void handleRequest()
{
    if (!server.hasArg("predict"))
    {
        server.send(400, "text/plain", "Falta el parámetro 'predict'");
        return;
    }

    String predict = server.arg("predict");
    float value = predict.toFloat();

    if (value > 0.5)
    {
        isDog();
    }
    else
    {
        isCat();
    }

    server.send(200, "text/plain", "LED actualizado: " + String(value > 0.5 ? "Perro" : "Gato"));

    delay(10000);

    none();
}

void isDog()
{
    digitalWrite(ledCat, LOW);
    digitalWrite(ledDog, HIGH);
}

void isCat()
{
    digitalWrite(ledDog, LOW);
    digitalWrite(ledCat, HIGH);
}

void none()
{
    Serial.println("Apagando LEDs");
    digitalWrite(ledDog, LOW);
    digitalWrite(ledCat, LOW);
}
