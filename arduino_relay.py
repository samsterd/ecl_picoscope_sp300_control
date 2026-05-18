import safe_exit as se
import serial
import serial.tools.list_ports

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
            } else if (command == "RELAY_OFF") {
                digitalWrite(13, LOW);
            } else if (command == "PING") {
                Serial.println("PONG");
            }
        }
    }

    Functions:
        init: finds port, establishes connection, sets relay to off and registers the safe_exit condition
        findPort: identifies the USB port the Arduino is plugged into
        sendCommand: sends a command string to the arduino
        validCommandQ: checks if an input command string is valid
        on: switches the relay on (USB disabled)
        off: switches the relay off (USB enabled)
        ping: sends the ping command and reads the response from the arduino
        disconnect: switches relay off and closes serial connection
    '''

    def __init__(self, baud : int = 9600):
        '''
        Initialize connection to simple Arduino relay.
        First the serial port is identified, then the Serial object is created.
        For safety, the disconnect function for the Relay instance is registered with safe_exit immediately,
        then the relay is set to off in case it was left on previously.

        Args:
            baud (int) : the baud rate of the connection. This must match the rate specified by the arduino code
        Returns:
            None
        '''

        # find the USB port
        self.port = self.findPort()
        if self.port == None:
            raise RuntimeError("Relay error: unable to find an arduino plugged into a USB port.")

        self.arduinoSerial = serial.Serial(self.port, baud, timeout = 1)
        se.register(self.disconnect)
        self.off()

    def findPort(self):
        '''
        Identifies the name of the USB port the Arduino is plugged into.

        Args:
            None
        Returns:
            str or None: either a string name of the port usable by PySerial, or None if the port is not found
        '''

        for port in serial.tools.list_ports.comports():
            if 'Arduino' in port.description or 'CH340' in port.description:
                return port.device

        return None

    def sendCommand(self, cmd):
        '''
        Sends specified command to the arduino. Since reading the text response requires ~1s of overhead, this is not performed
        in general and is instead handled by the 'PING' command.

        If cmd is not a valid command, it is not sent and an warning message is printed

        Args:
            cmd (str): string to send to the arduino. Valid options are "RELAY_ON", "RELAY_OFF", and "PING"
        Returns:
            None
        '''
        if self.validCommandQ(cmd):
            self.arduinoSerial.write((cmd + '\n').encode())
        else:
            print("arduino_relay Warning: specified command '" + str(cmd) + "' is not a valid command for the "
                                                                            "arduino relay. Command ignored.")

    @staticmethod
    def validCommandQ(cmd : str):
        '''
        Checks if a given command string is on the list of valid commands.
        The list is defined in this function

        Args:
            cmd (str) : the command string to test
        Returns:
            bool : is the command a recognized command for the Arduino relay
        '''
        cmdList = ["RELAY_ON", "RELAY_OFF", "PING"]

        return cmd in cmdList

    def off(self):
        '''
        Turns off relay by sending "RELAY_OFF" command
        This sets digital output pin 13 to low voltage

        Args:
            None
        Returns:
            None
        '''

        self.sendCommand("RELAY_OFF")


    def on(self):
        '''
        Turns on relay by sending "RELAY_ON" command
        This sets digital output pin 13 to high voltage

        Args:
            None
        Returns:
            None
        '''

        self.sendCommand("RELAY_ON")

    def ping(self, printRes = False):
        '''
        Sends a PING command, then optionally prints the response and returns the string.
        This is used primarily for debugging

        Args:
            None
        Returns:
            str : response from Arduino
        '''

        self.sendCommand("PING")

        res = self.arduinoSerial.readline().decode().strip()

        if printRes:
            print("Pinged Arduino relay.\nResponse: " + str(res))

        return res

    def disconnect(self):
        '''
        Closes the connection safely. First the relay is set to off and then the Serial connection is closed.
        This is done in a try/except block to handle cases where disconnect is called when the serial connection is already
        closed (or wasn't opened yet). This allows disconnect to be safely called by safe_exit
        '''
        try:
            self.off()
            self.arduinoSerial.close()
        except serial.serialutil.PortNotOpenError:
            print("Relay serial connection already closed")
