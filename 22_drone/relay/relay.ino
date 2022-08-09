
void setup() {
  Serial.begin(9600);
  // put your setup code here, to run once:
  pinMode(3, OUTPUT); // Define the Relaypin as output pin
  pinMode(4, OUTPUT);
  digitalWrite(3, HIGH);
  digitalWrite(4, HIGH);
} 
  
void loop() {
  // put your main code here, to run repeatedly:
  char a;
  a = Serial.read();
  if(a=='a'){
    Serial.println("승압");
    digitalWrite(3, LOW); // Sends high signal
    delay(2000); // Waits for 1 second                               
    digitalWrite(3, HIGH); // Makes the signal low
    }
  else if(a=='b'){
    Serial.println("Shoot");
    digitalWrite(4, LOW); // Sends high signal
    delay(500); // Waits for 1 second
    digitalWrite(4, HIGH); // Makes the signal low
  }
}
