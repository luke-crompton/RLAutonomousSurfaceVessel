#include <ESP32Servo.h>
#include <math.h>

// ============================================================
// Combined ESP32 sketch:
// - Reads 5 ultrasonic sensors one-at-a-time continuously
// - Accepts control commands over USB serial:
//     action_speed,action_yaw,u_meas,r_meas
// - Runs motor controller at 20 Hz
// - Sends latest sensor sweep back as CSV:
//     US1,US2,US3,US4,US5
//
// Notes:
// - ESC pins moved off GPIO12 / GPIO13 so US5 can keep using them.
// - Sensor polling is independent of the fixed 20 Hz control loop.
// ============================================================

Servo escLeft;
Servo escRight;

// ---------------- Motor pins (changed) ----------------
const int escLeftPin  = 21;
const int escRightPin = 22;

// ---------------- ESC ranges ----------------
const int ESC_MIN  = 1100;
const int ESC_MAX  = 1800;
const int ESC_STOP = 1500;

// ---------------- Ultrasonic pins ----------------
const int NUM_SENSORS = 5;
const int echoPins[NUM_SENSORS] = {34, 35, 25, 27, 12};
const int trigPins[NUM_SENSORS] = {32, 33, 26, 14, 13};

float distancesCm[NUM_SENSORS] = {-1, -1, -1, -1, -1};
int currentSensor = 0;
unsigned long nextSensorTriggerUs = 0;
const unsigned long sensorGapUs = 12000UL;      // 12 ms gap between sensors
const unsigned long echoTimeoutUs = 22000UL;    // ~3.7 m max, better than 30 ms

// ---------------- Controller state ----------------
float action_speed_limited = 0.0f;
float action_yaw_limited   = 0.0f;
float prev_action_speed    = 0.0f;
float prev_action_yaw      = 0.0f;

const float max_speed_cmd_step   = 0.12f;
const float max_yaw_cmd_step     = 0.10f;
const float max_yaw_reverse_step = 0.05f;

struct ESCCommands {
  int left_us;
  int right_us;
};

float int_u = 0.0f;
float int_r = 0.0f;

volatile float action_speed = 0.0f;
volatile float action_yaw   = 0.0f;
volatile float u_meas       = 0.0f;
volatile float r_meas       = 0.0f;

unsigned long lastControlMs = 0;
const float Ts = 0.05f;
const unsigned long controlPeriodMs = 50;

char serialBuf[128];
int serialPos = 0;

// ============================================================
// Utility helpers
// ============================================================
float clampFloat(float x, float xmin, float xmax) {
  if (x > xmax) return xmax;
  if (x < xmin) return xmin;
  return x;
}

int commandToMicroseconds(float cmd) {
  cmd = clampFloat(cmd, -1.0f, 1.0f);
  const int neutral_us = 1500;
  const int span_us = 400;
  return neutral_us + (int)(cmd * span_us);
}

float rateLimit(float target, float previous, float maxStep) {
  float delta = target - previous;
  if (delta > maxStep)  delta = maxStep;
  if (delta < -maxStep) delta = -maxStep;
  return previous + delta;
}

// ============================================================
// Motor controller
// ============================================================
ESCCommands computeThrusterESC(float action_speed,
                               float action_yaw,
                               float u_meas,
                               float r_meas,
                               float Ts) {
  const float v_max_fwd     = 1.0f;
  const float v_max_rev     = 0.2f;
  const float yaw_rate_max  = 3.14159265359f / 2.0f;
  const float turn_deadzone = 0.02f;

  const float Kp_speed = 0.35f;
  const float Ki_speed = 0.0f;

  const float Kp_yaw = 0.25f;
  const float Ki_yaw = 0.0f;

  action_speed = clampFloat(action_speed, -1.0f, 1.0f);
  action_yaw   = clampFloat(action_yaw,   -1.0f, 1.0f);

  float u_des = (action_speed >= 0.0f) ? (action_speed * v_max_fwd)
                                        : (action_speed * v_max_rev);

  float r_des = action_yaw * yaw_rate_max;
  if (fabs(r_des) < turn_deadzone * yaw_rate_max) {
    r_des = 0.0f;
  }

  float error_u = u_des - u_meas;
  float error_r = r_des - r_meas;

  int_u += error_u * Ts;
  int_r += error_r * Ts;

  int_u = clampFloat(int_u, -2.0f, 2.0f);
  int_r = clampFloat(int_r, -2.0f, 2.0f);

  float drive_cmd = Kp_speed * error_u + Ki_speed * int_u;
  float turn_cmd  = Kp_yaw   * error_r + Ki_yaw   * int_r;

  drive_cmd = clampFloat(drive_cmd, -1.0f, 1.0f);
  turn_cmd  = clampFloat(turn_cmd,  -1.0f, 1.0f);

  float left_cmd  = clampFloat(drive_cmd - turn_cmd, -1.0f, 1.0f);
  float right_cmd = clampFloat(drive_cmd + turn_cmd, -1.0f, 1.0f);

  ESCCommands out;
  out.left_us  = commandToMicroseconds(left_cmd);
  out.right_us = commandToMicroseconds(right_cmd);
  return out;
}

// ============================================================
// Serial command parsing
// ============================================================
bool parseCommandLine(char *line, float &a_speed, float &a_yaw, float &u, float &r) {
  int parsed = sscanf(line, "%f,%f,%f,%f", &a_speed, &a_yaw, &u, &r);
  return (parsed == 4);
}

void readSerialLines() {
  while (Serial.available() > 0) {
    char c = Serial.read();

    if (c == '\n' || c == '\r') {
      if (serialPos > 0) {
        serialBuf[serialPos] = '\0';

        float as, ay, um, rm;
        if (parseCommandLine(serialBuf, as, ay, um, rm)) {
          action_speed = as;
          action_yaw   = ay;
          u_meas       = um;
          r_meas       = rm;
        }
        serialPos = 0;
      }
    } else {
      if (serialPos < (int)sizeof(serialBuf) - 1) {
        serialBuf[serialPos++] = c;
      } else {
        serialPos = 0;
      }
    }
  }
}

// ============================================================
// Ultrasonic polling
// ============================================================
float readUltrasonicCM(int trigPin, int echoPin) {
  digitalWrite(trigPin, LOW);
  delayMicroseconds(2);
  digitalWrite(trigPin, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin, LOW);

  unsigned long duration = pulseIn(echoPin, HIGH, echoTimeoutUs);
  if (duration == 0) {
    return -1.0f;
  }

  return (float)duration * 0.0343f * 0.5f;
}

void printSensorSweep() {
  for (int i = 0; i < NUM_SENSORS; i++) {
    Serial.print(distancesCm[i], 1);
    if (i < NUM_SENSORS - 1) {
      Serial.print(',');
    }
  }
  Serial.println();
}

void serviceSensors() {
  unsigned long nowUs = micros();
  if ((long)(nowUs - nextSensorTriggerUs) < 0) {
    return;
  }

  distancesCm[currentSensor] = readUltrasonicCM(trigPins[currentSensor], echoPins[currentSensor]);
  currentSensor++;

  if (currentSensor >= NUM_SENSORS) {
    currentSensor = 0;
    printSensorSweep();
  }

  nextSensorTriggerUs = micros() + sensorGapUs;
}

// ============================================================
// Fixed 20 Hz motor control loop
// ============================================================
void serviceControl() {
  unsigned long now = millis();
  if (now - lastControlMs < controlPeriodMs) {
    return;
  }
  lastControlMs = now;

  float target_speed = clampFloat(action_speed, -1.0f, 1.0f);
  float target_yaw   = clampFloat(action_yaw,   -1.0f, 1.0f);

  action_speed_limited = rateLimit(target_speed, prev_action_speed, max_speed_cmd_step);

  float yawStep = max_yaw_cmd_step;
  if ((target_yaw > 0.0f && prev_action_yaw < 0.0f) ||
      (target_yaw < 0.0f && prev_action_yaw > 0.0f)) {
    yawStep = max_yaw_reverse_step;
  }
  action_yaw_limited = rateLimit(target_yaw, prev_action_yaw, yawStep);

  prev_action_speed = action_speed_limited;
  prev_action_yaw   = action_yaw_limited;

  ESCCommands out = computeThrusterESC(action_speed_limited,
                                       action_yaw_limited,
                                       u_meas,
                                       r_meas,
                                       Ts);

  escLeft.writeMicroseconds(out.left_us);
  escRight.writeMicroseconds(out.right_us);
}

// ============================================================
// Setup / loop
// ============================================================
void setup() {
  Serial.begin(115200);
  delay(1000);

  for (int i = 0; i < NUM_SENSORS; i++) {
    pinMode(echoPins[i], INPUT);
    pinMode(trigPins[i], OUTPUT);
    digitalWrite(trigPins[i], LOW);
  }

  escLeft.setPeriodHertz(50);
  escRight.setPeriodHertz(50);
  escLeft.attach(escLeftPin, ESC_MIN, ESC_MAX);
  escRight.attach(escRightPin, ESC_MIN, ESC_MAX);

  escLeft.writeMicroseconds(ESC_STOP);
  escRight.writeMicroseconds(ESC_STOP);
  delay(3000);

  nextSensorTriggerUs = micros();
}

void loop() {
  readSerialLines();
  serviceSensors();
  serviceControl();
}
