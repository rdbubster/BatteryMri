// BatteryMRI - ESP32 charge logger  (TEMPLATE - not yet tested on a real cell; test with a protected 18650 under supervision)
// Hardware: ESP32 DevKit, INA219 (I2C: SDA=21, SCL=22) in series with the cell's charge path, DS18B20 on pin 4 (4.7k pull-up to 3V3).
// The ESP32 is a PASSIVE LOGGER. Charging itself is done by a protected charger module (e.g. TP4056 with protection) - never by this code.
// Output (115200 baud) is a CSV that the dashboard "Diagnose your data" tab reads:  t,v,i,temp   (seconds, volts, amps, deg C)
#include <Wire.h>
#include <Adafruit_INA219.h>
#include <OneWire.h>
#include <DallasTemperature.h>

Adafruit_INA219 ina;
OneWire oneWire(4);
DallasTemperature ds(&oneWire);
unsigned long t0;

void setup() {
  Serial.begin(115200);
  ina.begin();                       // default 32V / 2A range; use ina.setCalibration_16V_400mA() or a 0.01 ohm shunt for >2A
  ds.begin();
  ds.setWaitForConversion(false);
  Serial.println("t,v,i,temp");
  t0 = millis();
}

void loop() {
  static unsigned long last = 0;
  if (millis() - last >= 1000) {     // 1 sample per second is plenty for ICA
    last = millis();
    ds.requestTemperatures();
    float v = ina.getBusVoltage_V() + ina.getShuntVoltage_mV() / 1000.0;   // battery-side voltage
    float i = ina.getCurrent_mA() / 1000.0;
    float T = ds.getTempCByIndex(0);
    Serial.printf("%.1f,%.4f,%.4f,%.2f\n", (millis() - t0) / 1000.0, v, i, T);
  }
}
