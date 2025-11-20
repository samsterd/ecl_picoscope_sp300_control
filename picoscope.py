# Code base to handle connecting to the oscilloscope and running experiments
#      Based heavily off of the code used in ultrasonicTesting project
# Interactions with the scope will be through the picoscope class
# Class functions:
#   connect
#   initialize channels
#   program AWG
#   run experiment / collect data
#   functionToArbitraryWaveform (static method) : convert a python function to a buffer suitable to input to ps2000aSetSigGenArbitrary()
#   close
# Class variables:
#   chandle (identifier for connection)

# todo: we are running without errors, but not producing data on the channels. need to investigate what is actually coming out
#       possible culprits: streaming callback function, triggering
#   first problem: scope triggering is not going to run - it is circular right now?
#       But the auto trigger should be running?
# todo: add trigger on channel D - will use the biologic trigger out function

import ctypes
import numpy as np
import math
from picosdk.ps2000a import ps2000a as ps
from picosdk.functions import adc2mV, assert_pico_ok
import time


class Picoscope():
    '''
    A class for interacting the Picoscope 2405A. Contains functions for connecting via USB, setting up the AWG, and running
    experiments

    Functions:
        init(params : dict) : create connection by calling openPicoscope, save input params as a class variable
        openPicoscope() : initialize USB connection to the scope, saves the chandle as a variable
        resolveSampleInterval() : helper function that checks the values of the requested sample interval and converts it
                                  to the SDK-required format
        runExperiment() : sets up channels and AWG, allocates data buffers, runs streaming mode, returns data
        initChannels() : set up the measurement and trigger channels
        voltageIndexFromRange(voltageRange : float) : helper function to round an input voltage range to the nearest allowed value
        initAWG() : set up the arbitrary wave generator (AWG)
        vtDataToArbitraryWaveform() : converts vt function and time limits in experiment params into a buffer suitable to input to the AWG
        initDataBuffers() : allocate data buffers to save data in streaming mode
        streamingCallback(handle, numberOfSamples, startIndex, overflow, triggerAT, triggered, autoStop, pParameter) :
            function called by streaming function after gathering data from the scope in order to copy data from buffer into memory
        runStream() : runs streaming mode
        closePicoscope() : closes pico connection
    '''

    def __init__(self):
        '''
        Connect to the picoscope and define a few constants

        Args:
            None
        Returns:
            None
        '''
        self.maxDataBufferSize = 10000  # maximum number of samples each channel's buffer can hold
        # safe guess based on 2405B memory of 48 kS (divide by 3 channels, with overhead)

        # open picoscope. this also initializes self.cHandle
        self.openPicoscope()

        # gather min/max waveform values
        self.minBufferVal = ctypes.c_int16()
        self.maxBufferVal = ctypes.c_int16()
        self.minBufferSize = ctypes.c_uint32()
        self.maxBufferSize = ctypes.c_uint32()
        # inputs: cHandle, pointers to output the min/max buffer values and sizes
        minMaxStatus = ps.ps2000aSigGenArbitraryMinMaxValues(self.cHandle,
                                                             ctypes.byref(self.minBufferVal),
                                                             ctypes.byref(self.maxBufferVal),
                                                             ctypes.byref(self.minBufferSize),
                                                             ctypes.byref(self.maxBufferSize))
        assert_pico_ok(minMaxStatus)

    def loadExperiment(self, params : dict):
        '''
        Loads experimental parameters and does some basic error checking

        Args:
            params (dict) : input parameters dict, as defined in main.py
        Returns:
            None
        '''
        # gather the required parameters from the input dict for convenience
        # todo: make a gatherParams function that iterates through all needed keys and raises an error with all missing values
        self.vtFunc = params['vtFunc']
        self.vtFuncArgs = params['vtFuncArgs']
        self.vtFuncKwargs = params['vtFuncKwargs']
        self.vtPeriod = params['vtPeriod']
        self.tStep = params['tStep']
        self.experimentTime = params['experimentTime']
        self.targetSamples = params['scopeSamples']
        self.channelARange = params['detectorVoltageRange0']
        self.channelBRange = params['detectorVoltageRange1']
        self.channelCRange = params['potentiostatVoltageRange']
        self.channelDRange = 20  # hardcoding trigger channel range to max
        self.targetInterval = self.experimentTime / self.targetSamples
        self.resolveSampleInterval()
        self.params = params  # this is redundant but might be helpful for debugging


    def resolveSampleInterval(self):
        '''
        Helper function that determines the units of the target sample interval and provides a copy of the target interval
        that can be re-written by ps2000aRunStreaming(). Also adjusts the scopeSamples parameter to reach as the targeted
        experimentTime as closely as possible.

        Args:
            None. Requires self.targetInterval (float) : the target scope sampling interval determined by the input experimentTime and scopeSamples
        Returns:
            0. Saves self.sampleInterval and self.sampleUnits : the target interval as an uint32, the Pico constant corresponding to the target units
        '''
        # handle case where target interval is less than 16 ns
        #   16 ns is minimum value that does not result in a PICO_INVALID_SAMPLE_INTERVAL error
        if self.targetInterval < 16e-9:
            print(
                "Warning: requested sample interval (experimentTime/scopeSamples) is less than the sampling limit of the "
                "Picoscope (16 ns). A 16 ns interval will be used and number of samples will be adjusted to match experimentTime")
            self.scopeSamples = math.floor(self.experimentTime / 16e-9)
            self.sampleInterval = ctypes.c_uint32(16)
            self.sampleUnits = ps.PS2000A_TIME_UNITS['PS2000A_NS']
            self.sampleUnitVals = 1e-9
            return 0

        unitConstants = [ps.PS2000A_TIME_UNITS['PS2000A_S'], ps.PS2000A_TIME_UNITS['PS2000A_MS'], ps.PS2000A_TIME_UNITS['PS2000A_US'], ps.PS2000A_TIME_UNITS['PS2000A_NS']]
        unitVals = np.array([1, 1e-3, 1e-6, 1e-9])

        # find the largest unit that is below the target interval
        # error handling done in previous step, array cannot be empty since targetInverval >= 2e-9
        unitIndex = np.argmax(unitVals <= self.targetInterval)
        self.sampleUnits = unitConstants[unitIndex]
        self.sampleUnitVals = unitVals[unitIndex]

        # convert target interval to the sample units, round to nearest int and save
        convertedInterval = math.floor(self.targetInterval / unitVals[unitIndex])
        self.sampleInterval = ctypes.c_uint32(convertedInterval)

        # determine number of samples needed to reach the target experimentTime
        self.scopeSamples = math.floor(self.experimentTime / (convertedInterval * unitVals[unitIndex]))

        return 0

    def openPicoscope(self):
        '''
        Establishes connection to the picoscope, saves the chandle (the unique 16 bit identifier used by the PicoSDK for
        communicating with the scope).

        Args:
            None
        Returns:
            None
        '''

        # create cHandle
        self.cHandle = ctypes.c_int16()

        # Open the unit with the cHandle ref. None for second argument means it will return the first scope found
        # The outcome of the operation is recorded in cHandle
        self.openUnit = ps.ps2000aOpenUnit(ctypes.byref(self.cHandle), None)

        # Print plain explanations of errors
        if self.cHandle == -1:
            print("Picoscope failed to open. Check that it is plugged in and not in use by another program.")
        elif self.cHandle == 0:
            print("No Picoscope found. Check that it is plugged in and not in use by another program.")

        # Raise errors and stop code
        assert_pico_ok(self.openUnit)

    def initStream(self):
        '''
        Sets up and runs stream but does not start data collection loop.
        Initializes channels and AWG, allocates data buffers, calls ps2000aRunStreaming

        Args: None
        Returns: None
        '''
        # initialize channels, AWG, and data buffers
        self.triggered = False
        self.initChannels()
        self.initDataBuffers()
        self.initAWG()

        # run stream args
        #   handle (int16)
        #   sampleInterval (uint32 pointer byref(self.sampleInterval)) -  target sampling interval. Will get overwritten with actual value after run
        #   sample interval time units (constant self.sampleUnits) - calculated by init helper function
        #   maxPreTriggerSamples (uint32 - 0)
        #   maxPostTriggerSamples (uint32 - scopeSamples)
        #   autoStop (int16 - 1) - stops streaming when buffer is full
        #   downSampleRatio (uint32) - not using downsampling
        #   downsample ratio mode (constant) - not using downsampling
        #   overviewBufferSize (uint32) - = bufferLth passed to setDataBuffer()
        runStreamingStatus = ps.ps2000aRunStreaming(self.cHandle, ctypes.byref(self.sampleInterval), self.sampleUnits,
                                                    0, self.scopeSamples, 1, 1, 0, self.dataBufferSize)
        assert_pico_ok(runStreamingStatus)


    def runStream(self):
        '''
        Collects data from an ongoing streaming experiment.
        Streaming data gathering loop adapted from example script in PicoSDK.

        Args: None (initStream must have been called beforehand)
        Returns: data arrays (channel A, channel B, channel C, channel D, time)
        '''
        #todo: write checks that initStream was called
        # self.startTime = time.time()
        # self.autoTriggerTime = self.startTime + self.experimentTime + 0.033

        # convert the callback function to a C function pointer
        callbackPointer = ps.StreamingReadyType(self.streamingCallback)
        # gather data in a loop
        while self.nextSample < self.scopeSamples and not self.autoStopOuter:
            self.wasCalledBack = False
            getValsStatus = ps.ps2000aGetStreamingLatestValues(self.cHandle, callbackPointer, None)
            if not self.wasCalledBack:
                # check back based on 10% of the approximate time to fill the buffer
                time.sleep(self.approxStreamingInterval / 10)

        # stop AWG and collection
        stopStatus = ps.ps2000aStop(self.cHandle)
        assert_pico_ok(stopStatus)

        # convert data to mV
        maxADC = ctypes.c_int16()
        maxValStatus = ps.ps2000aMaximumValue(self.cHandle, ctypes.byref(maxADC))
        assert_pico_ok(maxValStatus)

        # note that the raw data are 16-bit ints. They need to be converted to 32- or 64- bit to avoid overflows
        self.channelAData = np.array(adc2mV(self.channelARawData.astype(np.int_), self.aRange, maxADC))
        self.channelBData = np.array(adc2mV(self.channelBRawData.astype(np.int_), self.bRange, maxADC))
        self.channelCData = np.array(adc2mV(self.channelCRawData.astype(np.int_), self.cRange, maxADC))
        self.channelDData = np.array(self.channelDRawData.astype(np.int_)) # we want the channel D data in raw ADC to judge the triggering set point

        # generate time data. Starts at 0 since there is no delay after trigger
        self.time = np.linspace(0, (self.scopeSamples - 1) * self.sampleInterval.value * self.sampleUnitVals, self.scopeSamples)
        # self.time = np.linspace(0, self.experimentTime, self.scopeSamples)

        return self.channelAData, self.channelBData, self.channelCData, self.channelDData, self.time

    def initChannels(self):
        '''
        Initializes the measurement channels and triggers on the Picoscope.
        A - Photodetector 0
        B - Photodetector 1
        C - Potentiostat Out / Trigger

        Args:
            None. Requires 'experimentTime' and 'scopeSamples' parameters in experimentParameters
        Returns:
            None
        '''
        # gather voltage ranges for each channel
        self.aRange = self.voltageIndexFromRange(self.channelARange)
        self.bRange = self.voltageIndexFromRange(self.channelBRange)
        self.cRange = self.voltageIndexFromRange(self.channelCRange)
        self.dRange = self.voltageIndexFromRange(self.channelDRange)

        # set channel args:
        #   handle (int16)
        #   channel (constants 0-3)
        #   enabled (1 for enabled)
        #   type (1 for DC)
        #   range (calculated above)
        #   offset (float, default to 0, may change later)
        chAStatus = ps.ps2000aSetChannel(self.cHandle, 0, 1, 1, self.aRange, 0)
        chBStatus = ps.ps2000aSetChannel(self.cHandle, 1, 1, 1, self.bRange, 0)
        chCStatus = ps.ps2000aSetChannel(self.cHandle, 2, 1, 1, self.cRange, 0)
        chDStatus = ps.ps2000aSetChannel(self.cHandle, 3, 1, 1, self.dRange, 0)

        # set trigger. For now, triggering on Channel D - idea is to trigger when potentiostat starts chronoamperometry
        # args:
        #   handle (int16)
        #   enable (1)
        #   source (3 - Channel D)
        #   threshold (int16 - will need to play with this a bit)
        #   direction 0 (ABOVE)
        #   delay (0 - can't use delay in streaming mode)
        #   autoTrigger_ms (int16 - 10000s - trigger after 10 s)
        triggerStatus = ps.ps2000aSetSimpleTrigger(self.cHandle, 1, 3, 1000, 0, 0, 1000)

        # error check
        assert_pico_ok(chAStatus)
        assert_pico_ok(chBStatus)
        assert_pico_ok(chCStatus)
        assert_pico_ok(chDStatus)
        assert_pico_ok(triggerStatus)

    @staticmethod
    def voltageIndexFromRange(voltageRange):
        '''
        Helper function that rounds an input voltage range to the nearest allowed value and then converts to the corresponding
        constants used as input to ps2000aSetChannel(). Raises an error if the input is greater than the maximum allowed

        Args:
            voltageRange (float) : the input value from experimental parameters
        Returns:
            constant : the constant defined in the PicoSDK corresponding to the nearest allowed value
        '''

        # voltageLimits taken from API ps2000aSetChannel() documentation, they are hard coded in the picoscope
        voltageLimits = np.array([0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20])
        voltageConstants = [ps.PS2000A_RANGE['PS2000A_20MV'], ps.PS2000A_RANGE['PS2000A_50MV'], ps.PS2000A_RANGE['PS2000A_100MV'],
                            ps.PS2000A_RANGE['PS2000A_200MV'], ps.PS2000A_RANGE['PS2000A_500MV'],
                            ps.PS2000A_RANGE['PS2000A_1V'], ps.PS2000A_RANGE['PS2000A_2V'], ps.PS2000A_RANGE['PS2000A_5V'],
                            ps.PS2000A_RANGE['PS2000A_10V'], ps.PS2000A_RANGE['PS2000A_20V']]

        # get the first voltage that is above the voltageRange input
        try:
            voltageIndex = np.argmax(voltageLimits >= voltageRange)
        except ValueError:
            raise ValueError("Input voltage range exceeds the 20V limit. Verify the input is correct. Do not use the"
                               "Picoscope if expecting inputs exceeding 20V.")

        return voltageConstants[voltageIndex]

    @staticmethod
    def voltageToPotentiostatCurrent(voltageArray, currentRange):
        '''
        Help function to convert the I_Monitor data from the Biologic (in mV) into current based on the current range of the
        potentiostat

        Args:
            voltageArray (array) : array of voltages (measured from Channel C in a typical experiment) (mV)
            currentRange (int) : the current range of the Biologic, an int from 0 to 9

        Returns:
            array (currents) : an array of current values, in amps
        '''
        maxCurrents = [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
        maxCurrent = maxCurrents[currentRange]

        return maxCurrent * voltageArray / 1000 # divide by 1000 to convert mV to V

    def initAWG(self):
        '''
        Initializes the arbitrary waveform generator (AWG)

        Args:
            None. Requires AWG parameters in the experimental params were properly filled out
        Returns:
            0 if successful, else -1
        '''
        # initialize the buffer
        self.generateAWGBuffer()

        # calculate the number of shots needed to run for vtPeriod time. Print a warning if the amount exceeds 2e32-1
        rawShots = math.floor(self.experimentTime / self.vtPeriod)
        if rawShots > 2e32-1:
            self.awgShots = 0xFFFFFFFF # max value of 32 bit int sets AWG to run continuously
            self.awgDuration = self.awgShots.value * self.vtPeriod
            print("AWG Warning: number of voltage function periods implied by vtPeriod and experimentTime settings exceeds " +
                  "the amount possible using the AWG (2e32-1). Experiment will with the AWG set to run continuously.")
        else:
            self.awgShots = rawShots
            self.awgDuration = self.awgShots * self.vtPeriod

        # call setsiggenarbitrary
        sigGenStatus = ps.ps2000aSetSigGenArbitrary(
            self.cHandle,  # scope identifier, int16
            self.awgOffset,  # offsetVoltage = 0 (int32) - set in generateAWGBuffer
            self.pkToPk,  # peak to peak in uV,  uint32 - calculated in generateAWGBuffer
            self.startDeltaPhase,  # (uint32) - calculated in generateAWGBuffer
            self.startDeltaPhase,  # stopDeltaPhase = startDeltaPhase (uint32) - only differs from start when sweeping
            0,  # deltaPhaseIncrement = 0 (uint32) - only non-zero when sweeping
            3,  # dwellCount (uint32) = ? (uint32)  - how long each step lasts when frequency sweeping. Set to minimum value (3), does not actually matter if not sweeping
            ctypes.byref(self.waveformBuffer),  # arbitraryWaveform = pointer to uint32 buffer -  voltage samples for the input vtFunc
            ctypes.c_int32(self.numberOfPoints),  # arbitraryWaveformSize = numberOfPoints (int32)
            ctypes.c_int32(0),  # sweepType = PS2000A_UP (shouldn't matter if not sweeping)
            0, #   # operation = PS2000A_ES_OFF (normal operation)
            0,  # indexMode = PS2000A_SINGLE (waveform buffer fully specifies signal, it isn't half of a mirrored signal)
            ctypes.c_uint32(self.awgShots),# ctypes.c_uint32(self.awgShots), # number of repeats of the signal (implied by vtPeriod and experimentTime)
                                        # setting to max 0xFFFFFFFF runs continuously
            0,  # sweeps = 0  (we're doing a set number of shots, not sweeps)
            ctypes.c_int32(0),  # triggerType = Rising? (hoping this is ignored when using scope trigger)
            ctypes.c_int32(1),  # triggerSource = 1 (PS2000A_SIGGEN_SCOPE_TRIG) (verify this means using a trigger from ps2000aSetSimpleTrigger()
            #                   else _SOFT_TRIG maybe?
            1  # extInThreshold = 0 (not using external trigger)
        )

        assert_pico_ok(sigGenStatus)

    def generateAWGBuffer(self):
        '''
        Converts voltage/time function describing the voltage profile to be applied through the potentiostat into a properly
        formatted buffer to be used as the arbitraryWaveform input to ps2000aSigGenArbitrary(). See the PicoSDK
        documentation for further information.

        Args:
            None, requires the following params in the input params dict:
                vtFunc (callable) : a function which outputs a voltage (in V) for an array of times input (in s)
                vtPeriod (float) : specify the time range to generate outputs for vtFunc
                tStep (float) : the timesteps to call vtFunc. Will get rounded to the nearest 50 ns
                    vtPeriod and tStep specify the times uses in the AWG by np.linspace(0, vtPeriod, vtPeriod / round(tStep) + 1)
                    Note that minArbitraryWaveformSize <= vtPeriod / round(tStep) + 1 <= maxArbitraryWaveformSize or an error
                    will be raised
                *funcArgs, **funcKwargs : additional args and kwargs to pass into vtFunc, if needed

        Returns:
            -1 for error, 0 for normal operation

        How is the wave represented:
            arbitraryWaveform is a buffer (array) of data, where each sample (point) is a value directly proportional to
            the voltage to be output
            The AWG steps through the buffer at a certain frequency and outputs the voltage based on the sample value in
            the buffer
            The timing is determined by several parameters describing the phase:
                startDeltaPhase : the increment added to the phase accumulator (ie index in buffer). this + buffer size determine
                    rate the waveform buffer is output i.e. its frequency
                    use ps2000aSigGenFrequencyToPhase() to get the correct value of this given a specified buffer size and desired frequency
                stopDeltaPhase, deltaPhaseIncrement, dwellCount : these parameters are only used if sweeping the waveform frequency
                    At least initially, we only want a single frequency, so these are ignored
                    When EIS is implemented as a test this will probably be useful
            The voltage value is calculated from the buffer value by
                vout = 1uV * (pkToPk / 2) * (sample_val/32767) + offsetVoltage
                pkToPk : peak-to-peak (i.e. max - min) of the wave
                offsetVoltage : constant added/subtracted
                vout is always clipped to +- 2V (2e6 since in units of uV)

        :param wavefunc:
        :return:
        '''
        # calculate number of points implied by vtPeriod/tStep, check it is within the bounds
        self.numberOfPoints = math.floor(self.vtPeriod / self.tStep) # this isn't saved as a proper Ctype because it needs
                                                                     # to be signed or unsigned depending on the function using it :(
        if self.numberOfPoints < self.minBufferSize.value:
            print("functionToArbitraryWaveform: not enough time points specified. Inputs imply " +
                  str(self.numberOfPoints) + " points but AWG requires " + str(self.minBufferSize) +
                  " points. Consider lowering the value of tStep or increasing vtPeriod.")
            return -1
        elif self.numberOfPoints > self.maxBufferSize.value:
            print("functionToArbitraryWaveform: too many time points specified. Inputs imply " +
                  str(self.numberOfPoints) + " points but AWG has a maximum of " + str(self.maxBufferSize.value) +
                  " points. Consider raising the value of tStep or decreasing vtPeriod.")
            return -1

        # calculate frequency based on vtPeriod, use to calculate startDeltaPhase
        targetFreq = (1 / self.vtPeriod)
        self.startDeltaPhase = ctypes.c_uint32()  # value will be written by next line

        # inputs:
        #   chandle
        #   frequency (double)
        #   indexMode (built in constant)
        #   bufferLength (uint32)
        #   phase (output, uint32 pointer)
        phaseStatus = ps.ps2000aSigGenFrequencyToPhase(
            self.cHandle,
            ctypes.c_double(targetFreq),
            0, # INDEX_MODE = SINGLE
            ctypes.c_uint32(self.numberOfPoints),
            ctypes.byref(self.startDeltaPhase)
        )

        # handle errors
        assert_pico_ok(phaseStatus)

        # create linspace of times, use to output voltages from vtFunc
        self.awgTime = np.linspace(0, self.vtPeriod, self.numberOfPoints)
        voltages = self.vtFunc(self.awgTime, *self.vtFuncArgs, **self.vtFuncKwargs) * 1e6  # convert to uV
        self.awg = voltages # used for saving later

        # convert values to awg buffer sample values
        # formula in SDK is vout = 1uV * (pkToPk / 2) * (sample value / 32767) + offsetVoltage
        #   This scales the Vout values to the peak-to-peak value and divides into a 16-bit int (32767 = 2e15-1, one bit is sign)
        # We will set the offset to be max + min / 2 so that the waveform is optimally positioned in the middle of the range
        # sample value = (65534 * (vout - offset)) /  pkToPk
        self.pkToPk = ctypes.c_uint32(math.floor(np.max(voltages) - np.min(voltages)))  # peak-to-peak voltage rounded to nearest uV
        self.awgOffset = ctypes.c_int32(math.floor((np.max(voltages) + np.min(voltages))/2))

        waveform = (65534 * (voltages  - self.awgOffset.value)) / self.pkToPk.value

        # to avoid overflow, need to cut max and min values to 32767 and -32768.
        ceilArray = np.where(waveform <= 32767, waveform, 32767)
        floorArray = np.where(ceilArray >= -32768, ceilArray, -32768).astype(ctypes.c_int16)

        self.waveformBuffer = np.ctypeslib.as_ctypes(floorArray)
        self.awgBuffer = self.waveformBuffer # renaming for later saving

        return 0

    def initDataBuffers(self):
        '''
        Allocate space for the data buffers and full data arrays that the scope will send data to, then connect those buffers to the scope
        via ps2000aSetDataBuffer().

        Streaming mode requires both a data buffer and a full data array - if the requested amount of data is larger than
        the memory on the picoscope, it needs to be gathered and transferred in chunks to the data buffer. The data buffer
        is copied into the full data array and then overwritten by the next chunk of data. The streaming data can become
        discontinuous if the scope memory fills faster than the buffer can be copied to computer memory.

        Args:
            None. Requires self.scopeSamples
        Returns:
            None. self.dataBufferA/B/C and self.dataA/B/C are saved as class variables
        '''
        # determine data buffer size
        if self.scopeSamples < self.maxDataBufferSize:
            self.dataBufferSize = self.scopeSamples
            self.approxStreamingInterval = self.dataBufferSize * self.targetInterval
        else:
            self.dataBufferSize = self.maxDataBufferSize
            self.approxStreamingInterval = self.dataBufferSize * self.targetInterval
            # experiment will fill more than one buffer, therefore check that there is a reasonable amount of time to execute
            # the copy to memory step. If there isn't, print a warning
            # according to Picotech, maximum sampling rate for 2205A is 1 MS/s, so check that
            self.sampleRate = self.scopeSamples / self.experimentTime

            if self.sampleRate > 1e6:
                print("Warning: requested sampling rate is greater than max specifications. This may result in discontinuous " +
                      "sampling due to USB and memory bottlenecks. Double check results and consider reducing scopeSamples.")

        # allocate data arrays
        self.channelARawData = np.zeros(self.scopeSamples, dtype=ctypes.c_int16)
        self.channelBRawData = np.zeros(self.scopeSamples, dtype=ctypes.c_int16)
        self.channelCRawData = np.zeros(self.scopeSamples, dtype=ctypes.c_int16)
        self.channelDRawData = np.zeros(self.scopeSamples, dtype=ctypes.c_int16)

        # allocate streaming buffers
        self.channelABuffer = np.ctypeslib.as_ctypes(np.zeros(self.dataBufferSize, dtype=ctypes.c_int16))
        self.channelBBuffer = np.ctypeslib.as_ctypes(np.zeros(self.dataBufferSize, dtype=ctypes.c_int16))
        self.channelCBuffer = np.ctypeslib.as_ctypes(np.zeros(self.dataBufferSize, dtype=ctypes.c_int16))
        self.channelDBuffer = np.ctypeslib.as_ctypes(np.zeros(self.dataBufferSize, dtype=ctypes.c_int16))

        # initialize trackers that will be used by the callback function to copy data from buffers to proper location in data arrays
        # these are based on the streaming example in the picosdk github
        self.nextSample = 0
        self.autoStopOuter = False
        self.wasCalledBack = False

        # set data buffer (points the scope to the allocated buffers for fast saving)
        # args:
        #   chandle (int16)
        #   channel (0-3 - each channel needs its own data buffer)
        #   buffer (pointer to array of int16) - location of data
        #   bufferLth (int32) - length of buffer arrays (i.e. self.scopeSamples)
        #   segmentIndex (uint32) - not used in streaming mode
        #   mode (constant) - used for downsampling
        bufferAStatus = ps.ps2000aSetDataBuffer(self.cHandle, 0, ctypes.byref(self.channelABuffer),
                                                self.dataBufferSize, 0, 0)
        bufferBStatus = ps.ps2000aSetDataBuffer(self.cHandle, 1, ctypes.byref(self.channelBBuffer),
                                                self.dataBufferSize, 0, 0)
        bufferCStatus = ps.ps2000aSetDataBuffer(self.cHandle, 2, ctypes.byref(self.channelCBuffer),
                                                self.dataBufferSize, 0, 0)
        bufferDStatus = ps.ps2000aSetDataBuffer(self.cHandle, 3, ctypes.byref(self.channelDBuffer),
                                                self.dataBufferSize, 0, 0)

        # error checking
        assert_pico_ok(bufferAStatus)
        assert_pico_ok(bufferBStatus)
        assert_pico_ok(bufferCStatus)
        assert_pico_ok(bufferDStatus)

    def streamingCallback(self, handle, numberOfSamples, startIndex, overflow, triggerAT, triggered, autoStop,
                          pParameter):
        '''
        Function that is called by ps2000aGetStreamingLatestValues every time it returns in order to move data from the
        streaming buffer into memory. Adapted from example code in PicoSDK

        PROBLEM: this needs to be turned into a pointer and fed into ps.StreamingReadyType. The arguments are dictated by
        the SDK. This means it probably can't use self as an argument, so can't access class variables
        Need to access self.channelData vars, would strongly prefer not making those globals
        Ugly possible solution: move initBuffers into runStream function? then variables are accessible?
        In the example, the streaming_callback function uses variables not defined in the function -
            doesn't raise errors at start because they are globals, but it is promising?
        If this doesn't work, will need to make buffers and arrays globals (and then erase when done?)
        For now: implementing as a class function. I'll see if that fails before implementing everything as globals

        Args:
            None. Requires that data arrays and buffers were properly set up
        Returns:
            None. Values copied to data arrays, nextSample, wasCalledBack, and autoStopOuter are updated
        '''
        # print('callback')
        # self.triggered is used to track whether the scope was triggered at any point during the experiment
        #   versus triggered (argument) refers to whether the trigger occurs on this particular callback
        self.triggered = self.triggered or triggered

        # self.triggered = True
        # self.autoTrigger = time.time() > self.autoTriggerTime

        self.wasCalledBack = True

        # this section is included in the example but really not sure what it is doing :(
        if autoStop:
            self.autoStopOuter = True

        # save data in three cases: was previously triggered, triggered in this callback, or the autotriggertime has elapsed
        if self.triggered and not triggered:
            destEnd = self.nextSample + numberOfSamples
            sourceEnd = startIndex + numberOfSamples
            self.channelARawData[self.nextSample: destEnd] = self.channelABuffer[startIndex: sourceEnd]
            self.channelBRawData[self.nextSample: destEnd] = self.channelBBuffer[startIndex: sourceEnd]
            self.channelCRawData[self.nextSample: destEnd] = self.channelCBuffer[startIndex: sourceEnd]
            self.channelDRawData[self.nextSample: destEnd] = self.channelDBuffer[startIndex: sourceEnd]
            self.nextSample += numberOfSamples

        elif self.triggered and triggered:
            # triggered on this callback. Only want data after startIndex + triggeredAT
            sourceStart = startIndex + triggerAT
            sourceEnd = startIndex + numberOfSamples
            triggeredSamples = sourceEnd - sourceStart
            destEnd = self.nextSample + triggeredSamples

            self.channelARawData[self.nextSample: destEnd] = self.channelABuffer[sourceStart: sourceEnd]
            self.channelBRawData[self.nextSample: destEnd] = self.channelBBuffer[sourceStart: sourceEnd]
            self.channelCRawData[self.nextSample: destEnd] = self.channelCBuffer[sourceStart: sourceEnd]
            self.channelDRawData[self.nextSample: destEnd] = self.channelDBuffer[sourceStart: sourceEnd]
            self.nextSample += triggeredSamples

        # elif self.autoTrigger:
        #     print('autotriggered')
        #     destEnd = self.nextSample + numberOfSamples
        #     sourceEnd = startIndex + numberOfSamples
        #     self.channelARawData[self.nextSample: destEnd] = self.channelABuffer[startIndex: sourceEnd]
        #     self.channelBRawData[self.nextSample: destEnd] = self.channelBBuffer[startIndex: sourceEnd]
        #     self.channelCRawData[self.nextSample: destEnd] = self.channelCBuffer[startIndex: sourceEnd]
        #     self.channelDRawData[self.nextSample: destEnd] = self.channelDBuffer[startIndex: sourceEnd]
        #     self.nextSample += numberOfSamples

    def closePicoscope(self):
        '''
        Closes connection to Picoscope.

        Args:
            None. Requires self.cHandle
        Returns:
            None. Error is raised if close operation is not successful
        '''
        closeStatus = ps.ps2000aCloseUnit(self.cHandle)
        assert_pico_ok(closeStatus)

# a test input function for the AWG
def testVT(times, freq = 100, amp = 0.1):
    return amp * np.sin(2 * np.pi * times * freq)