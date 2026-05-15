import safe_exit as se
import serial
import serial.tools.list_ports
import time

class Relay():
    '''
    A class for controlling a simple Arduino relay for turning the stir plate on and off
    This is modified from example in PySerial documentation (https://www.pyserial.com/docs/arduino-integration)

    This assumes the following code has been previously sent to the Arduino (using digital pin 13 as voltage output):
    void setup() {
        Serial.begin(9600);
        Serial.flush();
        pinMode(13, OUTPUT);
        digitalWrite(13, LOW);
    }

    void loop() {
        if (Serial.available()) {
            String command = Serial.readString();
            command.trim();

            if (command == "RELAY_ON") {
                digitalWrite(13, HIGH);
                Serial.println("RELAY ON");
            } else if (command == "RELAY_OFF") {
                digitalWrite(13, LOW);
                Serial.println("RELAY OFF");
            } else if (command == "PING") {
                Serial.println("PONG");
            }
        }
    }

    Functions:
        init: establishes connection, sets relay to off and registers the safe_exit condition (off)
        findPort: identifies the USB port the Arduino is plugged into
        sendCommand: sends a command to the arduino
        on: switches the relay on (USB disabled)
        off: switches the relay off (USB enabled)
    '''

    def __init__(self, baud):

        # find the USB port
        self.port = self.findPort()
        if self.port == None:
            raise RuntimeError("Relay error: unable to find an arduino plugged into a USB port.")

        self.arduinoSerial = serial.Serial(self.port, baud, timeout = 1)
        se.register(self.disconnect)
        self.off()

    def findPort(self):

        for port in serial.tools.list_ports.comports():
            if 'Arduino' in port.description or 'CH340' in port.description:
                return port.device

        return None

    def sendCommand(self, cmd):

        self.arduinoSerial.write((cmd + '\n').encode())
        return self.arduinoSerial.readline().decode().strip()

    def off(self):

        res = self.sendCommand("RELAY_OFF")

        # if res != "RELAY OFF":
        #     raise RuntimeWarning("Arduino relay gave unexpected response. Expected 'RELAY OFF', received '" +
        #                          str(res) + "'")

    def on(self):

        res = self.sendCommand("RELAY_ON")

        # if res != "RELAY ON":
        #     raise RuntimeWarning("Arduino relay gave unexpected response. Expected 'RELAY ON', received '" +
        #                          str(res) + "'")

    def ping(self):

        res = self.sendCommand("PING")

        print("Pinged Arduino relay.\nResponse: " + str(res))

    def disconnect(self):
        ''' Make sure the relay is set to off, then close the Serial connection'''
        try:
            self.off()
            self.arduinoSerial.close()
        except serial.serialutil.PortNotOpenError:
            print("Relay serial connection already closed")

#todo: define a subclass for stirring?

# relay = Relay(9600)
# time.sleep(1)
# relay.on()
# time.sleep(1)
# relay.disconnect()
