#include <Wire.h>
#include <Servo.h>

// 조종기 PWM
#define Throttle_max 2000
#define Throttle_min 1000
#define NEUTRAL_THROTTLE 1500

// volatile 변수 -> 인터럽트 -> 동시실행을 위해 선언되어야 함
volatile int nThrottleIn3 = NEUTRAL_THROTTLE; //  
volatile int nThrottleIn4 = NEUTRAL_THROTTLE; //
volatile int nThrottleIn5 = NEUTRAL_THROTTLE; // 
volatile int nThrottleIn6 = NEUTRAL_THROTTLE; // 

volatile unsigned long ulStartPeriod3 = 0;
volatile unsigned long ulStartPeriod4 = 0;
volatile unsigned long ulStartPeriod5 = 0;
volatile unsigned long ulStartPeriod6 = 0;

volatile boolean bNewThrottleSignal3 = false; 
volatile boolean bNewThrottleSignal4 = false; 
volatile boolean bNewThrottleSignal5 = false; 
volatile boolean bNewThrottleSignal6 = false; 

//인터럽트 핀 할당
const byte interruptPin3 = 18; // CH3 상하 서보
const byte interruptPin4 = 19; //CH4 좌우 서보2
const byte interruptPin5 = 20; //CH5 코일건 스위치 ON/OFF
const byte interruptPin6 = 21; //CH1 서보 초기화 

// 미션부 초기값 설정
int rlangle = 90;
Servo servo1;

// 미션부 핀 할당
const int s_pin = 10;

char flag = 'Y';

//3번 채널 PWM 신호측정
void calcInput3()
{
  if(digitalRead(interruptPin3) == HIGH)
  { 
    ulStartPeriod3 = micros(); 
  }
  else
  {
    if(ulStartPeriod3 && (bNewThrottleSignal3 == false))
    {
      nThrottleIn3 = (int)(micros() - ulStartPeriod3);
      ulStartPeriod3 = 0;

      bNewThrottleSignal3 = true;
    }
  }
}

//2번 채널 PWM 신호측정
void calcInput4()
{
  if(digitalRead(interruptPin4) == HIGH)
  { 
    ulStartPeriod4 = micros(); 
  }
  else
  {
    if(ulStartPeriod4 && (bNewThrottleSignal4 == false))
    {
      nThrottleIn4 = (int)(micros() - ulStartPeriod4);
      ulStartPeriod4 = 0;

      bNewThrottleSignal4 = true;
    }
  }
}

//5번 채널 PWM 신호측정
void calcInput5()
{
  if(digitalRead(interruptPin5) == HIGH)
  { 
    ulStartPeriod5 = micros(); 
  }
  else
  {
    if(ulStartPeriod5 && (bNewThrottleSignal5 == false))
    {
      nThrottleIn5 = (int)(micros() - ulStartPeriod5);
      ulStartPeriod5 = 0;

      bNewThrottleSignal5 = true;
    }
  }
}

//5번 채널 PWM 신호측정
void calcInput6()
{
  if(digitalRead(interruptPin6) == HIGH)
  { 
    ulStartPeriod6 = micros(); 
  }
  else
  {
    if(ulStartPeriod6 && (bNewThrottleSignal6 == false))
    {
      nThrottleIn6 = (int)(micros() - ulStartPeriod6);
      ulStartPeriod6 = 0;

      bNewThrottleSignal6 = true;
    }
  }
}
void setup() {
  Serial.begin(9600);
  // 채널 할당 -> 작년 것과 동일하게 작성 / 신호가 바뀔때 (change[HIGH <-> LOW]) 인터럽트 발생 하여 calcInput을 실행시킨다
  attachInterrupt(digitalPinToInterrupt(interruptPin3),calcInput3,CHANGE); 
  attachInterrupt(digitalPinToInterrupt(interruptPin4),calcInput4,CHANGE); 
  attachInterrupt(digitalPinToInterrupt(interruptPin5),calcInput5,CHANGE); 
  attachInterrupt(digitalPinToInterrupt(interruptPin6),calcInput6,CHANGE); 
  servo1.attach(s_pin);
  servo1.write(rlangle);
  delay(10);
}

void loop() {
  // None
  if(bNewThrottleSignal3)
  {
    
    bNewThrottleSignal3 = false;
  }
  // 좌, 우 각도조절 서보모터 움직임
  if(bNewThrottleSignal4)
  {
    if(nThrottleIn4>1700){
      if(rlangle<135){
        rlangle+=1;
        servo1.attach(s_pin);
        servo1.write(rlangle);
        delay(20);
      }
     }
    else if(nThrottleIn4<1400){
      if(rlangle>45){
        rlangle -= 1;
        servo1.attach(s_pin);
        servo1.write(rlangle);
        delay(20);
      }
    }
    bNewThrottleSignal4 = false;
  }
  // 코일건 발사
  if(bNewThrottleSignal5)
  {
    if (nThrottleIn5 < 1350 && flag == 'Y'){
      // 충전 -> 장전 -> 발사
      flag = 'N';
      }
     else if(nThrottleIn5 > 1800){
      flag = 'Y';
      }
    bNewThrottleSignal5 = false;
  }
  // 각도조절 서보 동작 초기위치로
  if(bNewThrottleSignal6)
  {
    if (nThrottleIn6>1100){
      rlangle = 90;
      servo1.attach(s_pin);
      servo1.write(rlangle);
      delay(100);
    }
    bNewThrottleSignal6 = false;
  }
  // put your main code here, to run repeatedly:
  Serial.print("pwm  ");Serial.print(nThrottleIn3);Serial.print("|");Serial.print(nThrottleIn4);Serial.print("|");Serial.print(nThrottleIn5);Serial.print("|");Serial.print(nThrottleIn6);Serial.print("|\n");
}
