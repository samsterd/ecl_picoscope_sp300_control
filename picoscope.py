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


# todo: implement downsampling to get around USB data bottleneck (aggregation w/ averaging?)
#   changes:
#       xdefault to 1GHz samplng, expose only downsampling ratio to user
#           change numberOfSamples to requestedTimeInterval (ns)
#           change downsamplingRatio to requestedDownsamplingInterval (ns)
#           resolve timeInterval first, then get downsamplingInterval (rounded up) then numberOfSamples
#       xinit: change default values
#           testing: check values of min/max buffers. function runs fine
#       -loadExperiment: change handling of default values, sampling ratio
#             figure out if downsampleQ is still needed
#       xresolveSampleInterval: simplify, may not even be necessary
#           check that 1 ns is an acceptable timebase
#       xresolveDownsampling: simplify since number of samples is not exposed to user
#           delete the above functions once everything is tested
#       -condensed above functions into resolveSampling
#           return to this once all variables needed for later running are known
#           ADD CALCULATION OF AWG TRIGGER INDEX
#       -openPicoscope: change function name
#           finish troubleshooting connection: needs a combination of updating system path, restart, and running as administrator
#       -initStream: change function name, make downsampling handle correctly
#           check that rounding issues do not occur due to sample interval being in seconds
#       -runStream: make sure awg triggering still works
#           can
#       -initChannels: will want to play with trigger settings and check out if 50 ohm coupling (set in init()) works better
#           may run into issue where not all voltage ranges are allowed w/ 50 ohm coupling - cross that bridge when we get there
#       xvoltageIndexFromRange -
#           replaced with rangeMax: we need the nearest allowable voltage range, in nV, as an int64
#       voltageToPotentiostatCurrent - should be fully unchanged. will need to check that voltage conversion works with 50ohm connector?
#           absolute numbers may need adjustment!
#       -initAWG: test meaning of a few parameters (cycles on trigger, enabled on apply)
#       -generateAWGBuffer: test
#           test constant profile still works!
#       -initDataBuffers: delete commented out code once tested
#           potential source of issues - I do not fully understand what the action option means
#       xstreamingCallback: delete once everything is tested
#       xclosePicoscope: change function name

# fixing streaming memory constraints:
#       initDataBuffers: create local array for holding the data, create multiple buffers for each channel to hold the
#                           streaming data
#                       size of hardware buffers: 20% of max memory?
#                           issue isn't the assigned buffers but the memory allocated to samples (not downsamples)
#                           is this alleviated by clear|add action?
#       runStream: copy back the callback function -
#           handle triggered prev, triggered on current, etc
#           handle changing over buffers? or just let it be


#       self.downsamplingQ if downsamplingRatio != 1
#       check handling of resolveSampleInterval
#       initDataBuffers: use setDataBuffer() to include downsampled channels
#           calculate number of samples to allocate each buffer by ?
#       initStream: branch if downsampling or not
#       streamingCallback: branch to only copy data from the downsampling buffers
#           the real data buffers will have been filled and overwritten multiple times for each callback at max sampling rate
#           this shouldn't matter as long as we grab data fast enough before the downsample buffer overwrites
#           this gets to calculation for buffer sizes:
#               avg 125 samples gets 1 us resolution
#           safe assumption for callback function rate is 1 ms
#       change time reporting to match downsampling timestep
#       change all safety/error checking to account for downsampling
#   interface:
#       expose sampling rate and downsampling ratio
#           rate * ratio must be <250 kHz (4 us)
#       this is a change from number of samples... may want to fully overhaul this
#           or have samples = -1 for maximum rate
#           and ratio = -1 for minimum ratio
#           and failing that implement a check when resolving timebase that also checks and adjusts (with a message) the ratio
#   problem:
#       too much downsampling causes trigger to be missed sometimes?


import ctypes
import numpy as np
import math
from picosdk.psospa import psospa as ps
from picosdk.PicoDeviceEnums import picoEnum as enums
from picosdk.PicoDeviceStructs import picoStruct as structs
from picosdk.functions import adc2mVV2, assert_pico_ok, mV2adcV2
from picosdk.constants import PICO_STATUS
import safe_exit as se
from copy import copy
from scipy.signal import square


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
        # define some device
        self.maxDataBufferSize = 400e6  # maximum number of samples each channel's buffer can hold
                                        # safe guess based on 2GS memory (divide by 4 channels, with overhead)
        self.bitResolution = enums.PICO_DEVICE_RESOLUTION["PICO_DR_8BIT"]#enums.PICO_DEVICE_RESOLUTION["PICO_DR_8BIT"] # we will run in 10-bit mode for speed. If more speed is needed, switch to 8-bit mode
        self.coupling = 1 # options are 0 (1Mohm AC), 1 (1Mohm DC), and 50 (50 ohm DC)

        # todo: figure out what actual value is
        self.minAwgTimeStep = 5e-9      # minimum time step is 5 ns based on the value of ddsPeriod for Picoscope 3415E
        self.channelDRange = 20         # voltage range for channel D (the trigger channel)
        self.autoSampleInterval = 8 # sampling interval to aim for when scopeSamples = -1
        self.autoDownsampleInterval = 10000 # final sample interval to achieve with downsampling if downsampleRatio = -1
        # todo: check this formatting is correct
        self.channelEnabledBitfield = 0b1111 # bitfield indicated Channel A-D are on
        self.numberOfChannels = 4

        # open picoscope. this also initializes self.cHandle
        self.openPicoscope()
        se.register(self.closePicoscope)


        # gather min/max waveform values
        self.minBufferVal = ctypes.c_double()
        self.maxBufferVal = ctypes.c_double()
        self.bufferValStep = ctypes.c_double()
        self.minBufferSize = ctypes.c_double()
        self.maxBufferSize = ctypes.c_double()
        self.bufferSizeStep = ctypes.c_double()
        self.minAWGVolts = ctypes.c_double()
        self.maxAWGVolts = ctypes.c_double()
        self.awgVoltsStep = ctypes.c_double()
        bufferValStatus = ps.psospaSigGenLimits(self.cHandle,
                                                enums.PICO_SIGGEN_PARAMETER["PICO_SIGGEN_PARAM_SAMPLE"],
                                                ctypes.byref(self.minBufferVal),
                                                ctypes.byref(self.maxBufferVal),
                                                ctypes.byref(self.bufferValStep)
                                                )
        bufferSizeStatus = ps.psospaSigGenLimits(self.cHandle,
                                                enums.PICO_SIGGEN_PARAMETER["PICO_SIGGEN_BUFFER_LENGTH"],
                                                ctypes.byref(self.minBufferSize),
                                                ctypes.byref(self.maxBufferSize),
                                                ctypes.byref(self.bufferSizeStep)
                                                )
        awgVoltsStatus = ps.psospaSigGenLimits(self.cHandle,
                                                enums.PICO_SIGGEN_PARAMETER["PICO_SIGGEN_PARAM_OUTPUT_VOLTS"],
                                                ctypes.byref(self.minAWGVolts),
                                                ctypes.byref(self.maxAWGVolts),
                                                ctypes.byref(self.awgVoltsStep)
                                                )
        assert_pico_ok(bufferValStatus)
        assert_pico_ok(bufferSizeStatus)
        assert_pico_ok(awgVoltsStatus)

    def loadExperiment(self, params : dict):
        '''
        Loads experimental parameters and does some basic error checking

        Args:
            params (dict) : input parameters dict, as defined in main.py
        Returns:
            None
        '''
        # gather the required parameters from the input dict for convenience
        # also do some minor error catching
        # todo: make a gatherParams function that iterates through all needed keys and raises an error with all missing values
        self.awgFunc = params['awgFunc']
        self.awgFuncArgs = params['awgFuncArgs']
        self.awgFuncKwargs = params['awgFuncKwargs']

        if params['awgPeriod'] == None:
            if 'freq' in self.awgFuncKwargs.keys():
                self.awgPeriod = 1 / self.awgFuncKwargs['freq']
            else:
                raise ValueError("picoscope.loadExperiment: 'awgPeriod' was set to None but 'awgFuncKwargs' does not contain "
                                 "the key 'freq'. Experiment terminated.")
        else:
            self.awgPeriod = params['awgPeriod']

        awgRequestedSamples = params['awgSamples']
        # first resolve default value of -1
        if awgRequestedSamples == -1:
            # set such that we are sampling every 5 ns within awgPeriod
            awgTargetSamples = math.floor(self.awgPeriod / 5e-9)
        else:
            awgTargetSamples = awgRequestedSamples

        # next make sure target samples is within limits
        if awgTargetSamples > self.maxBufferSize.value:
            if awgRequestedSamples != -1:
                # only print warnings for non-default inputs
                print("Warning: requested number of AWG samples is greater than the max allowed by the hardware (" +
                  self.maxBufferSize.value + "). AWG samples set to this value.")
            self.awgSamples = int(self.maxBufferSize.value)

        elif awgTargetSamples < self.minBufferSize.value:
            if awgRequestedSamples != -1:
                # only print warnings for non-default inputs
                print("Warning: requested number of AWG samples is less than the min allowed by the hardware (" +
                  self.minBufferSize.value + "). AWG samples set to this value.")
            self.awgSamples = int(self.minBufferSize.value)

        else:
            self.awgSamples = awgTargetSamples # should be an int since targetSamples is floored

        if self.awgPeriod / self.awgSamples < 5e-9:
            # NOTE: this should not be possible if inputs are -1
            print("Warning: requested AWG sampling rate is too fast (awgPeriod/awgSamples < 5e-9). Consider lowering awgSamples.")

        self.awgDuration = params['awgDuration']
        self.awgDelayRaw = params['awgDelay'] # raw input separated to resolve edge case where a delay is input but awgFunc == None
        self.delayQ = not(self.awgDelayRaw == None) and not(self.awgDelayRaw == 0) and not(self.awgFunc == None)
        self.awgDelay = 0 if not(self.delayQ) else self.awgDelayRaw
        self.requestedDownsampleInterval = params['requestedDownsampleInterval']
        # self.downsampleQ = (self.downsampleRatioInput != 1 and self.downsampleRatioInput != None)
        self.experimentTime = params['experimentTime']
        self.awgStopQ = (self.awgDelay + self.awgDuration < self.experimentTime)
        self.requestedSampleInterval = params['requestedSampleInterval']
        self.channelARange = params['detectorVoltageRange0']
        self.channelBRange = params['detectorVoltageRange1']
        self.channelCRange = params['potentiostatVoltageRange']
        self.channelDRange = self.channelDRange  # hardcoding trigger channel range to max
        self.resolveSampling()
        self.params = params  # this is redundant but might be helpful for debugging

    def resolveSampling(self):
        '''
        Helper function that resolves input sample interval and downsampled interval to allowable values for the hardware.
        Also calculates the AWG triggering parameters, if needed
        Performs some error checking on the inputs and modifies the input requested sample information to the nearest
        hardware compatible values

        Args:
            None. Requires self.requestedSampleInterval and self.rawDownsampleInterval are informed
        Returns:
            None. Fills in the following class variables which are essential for later operation:
                sampleInterval (float) : time, in ns, between samples (pre-averaging)
                sampleIntervalSeconds (float) : time, in s, between samples
                sampleIntervalUnits (enum) : always seconds since that is what the functions return
                timebase (int) : hardware level timebase the sampleInterval. In ps
                downsampleRatio (int) : number of samples to be averaged for each data point returned to computer
                downsampleInterval (float) : time, in ns, between data points reported to computer after averaging
                numberOfDownsamples (int) : number of downsampled data points to save per experiment
                numberOfSamples (int) : number of samples the scope will collect per experiment
                correctedExperimentTime (float) : actual experiment time, in s, due to rounding of sample intervals
        '''

        # find nearest sample interval by calling built-in function. If sample interval is -1, use the minimum value
        self.timebase = ctypes.c_uint32()
        self.sampleIntervalSeconds = ctypes.c_double()
        self.sampleIntervalUnit = enums.PICO_TIME_UNITS["PICO_S"]

        # resolve user input sample interval based on defaults and bounds
        if self.requestedSampleInterval == -1:
            requestedInverval = ctypes.c_double(self.autoSampleInterval / 1e9)
        elif self.requestedSampleInterval < 4:
            print("Picoscope warning: minimum sampling interval while streaming with downsampling on 4 channels is 4 ns." +
                  "Sample interval changed to 4 ns for this experiment.")
            requestedInterval = ctypes.c_double(4.0e-9)
        # else find the nearest possible value, rounding down
        else:
            # first properly format the requested interval (convert from ns to s and make sure its a double)
            requestedInterval = ctypes.c_double(self.requestedSampleInterval / 1e9)

        # find nearest sample interval to requested interval
        # args: cHandle
        # channel enabled bitfield
        # requested interval (double, seconds)
        # round faster (uint8, 1 for faster, 0 for slower)
        # bit resolution
        # pointer to timebase
        # pointer to time interval
        sampleIntervalStatus = ps.psospaNearestSampleIntervalStateless(
            self.cHandle,
            self.channelEnabledBitfield,
            requestedInterval,
            1,
            self.bitResolution,
            ctypes.byref(self.timebase),
            ctypes.byref(self.sampleIntervalSeconds)
        )

        assert_pico_ok(sampleIntervalStatus)
        # sample interval is rounded for readability. Since the max sample rate is 5 GS/s, minimum value should be 0.2 ns/sample
        self.sampleInterval = round(self.sampleIntervalSeconds.value * 1e9, 2)

        # resolve defaults for downampling
        if self.requestedDownsampleInterval == -1:
            self.rawDownsampleInterval = self.autoDownsampleInterval
        else:
            self.rawDownsampleInterval = self.requestedDownsampleInterval

        # error handling: print a warning if downsampling interval < sampling interval
        if self.rawDownsampleInterval < self.sampleInterval:
            print("resolveSampling Warning: requested downsampling interval (" + str(self.rawDownsampleInterval) +
                  " ns) is shorter than the sampling interval (" + str(self.sampleInterval) +
                  " ns). A downsampling ratio of 1 will be used.")
            self.downsampleRatio = 1

        # calculate actual values, rounding down to the nearest int
        else:
            self.downsampleRatio = max(math.floor(self.rawDownsampleInterval / self.sampleInterval), 1)

        # error handling: sample interval constraints are longer when there isn't downsampling
        if self.downsampleRatio == 1 and self.sampleInterval < 50:
            print("Warning: requested sample interval of " + str(self.sampleInterval) + " ns may be too fast for " +
                  "USB streaming without downsampling. Experiment will run, but data may be discontinuous or missing.")

        # calculate actual downsample interval, in ns, for readability
        self.downsampleInterval = self.sampleInterval * self.downsampleRatio

        # calculate actual number of samples and downsamples to be taken
        # first calculate number of downsamples from experiment time (rounding up) and then use that number
        #   to calculate number of samples. This avoids cutting off experiments early due to rounding
        self.numberOfDownsamples = int(math.ceil(self.experimentTime / (self.downsampleInterval / 1e9)))
        self.numberOfSamples = self.downsampleRatio * self.numberOfDownsamples
        # calculate actual experiment time. use numberOfSamples - 1 since time starts at 0
        self.correctedExperimentTime = self.sampleIntervalSeconds.value * (self.numberOfSamples - 1)

        # if correctedExperimentTime deviates from the requested time by more than 1% print an error
        if abs(self.experimentTime - self.correctedExperimentTime) > 0.01 * self.experimentTime:
            print("resolveSampling Warning: calculated experiment time (" + str(self.correctedExperimentTime) +
                  " s) devaites by more than 1% from requested experiment time (" + str(self.experimentTime) +
                  " s) due to rounding. No action is needed unless down time at the end of the experiment is short.")

        # handle awg delay info
        # first put in filler data if a delay is not used
        if not(self.delayQ):
            self.awgTriggerSamples = 0
            self.awgDelayIndex = 0
            # we do need to calculate how many samples to run before stopping the AWG if it is stopping early
            if self.awgStopQ:
                self.awgRunSamples = math.floor(self.awgDuration / (self.sampleIntervalSeconds.value * self.downsampleRatio))
        else:
            #todo: check if this should be samples or downsamples
            self.awgTriggerSamples = math.floor(self.awgDelay / (self.sampleIntervalSeconds.value * self.downsampleRatio))
            self.awgRunSamples = math.floor(self.awgDuration / (self.sampleIntervalSeconds.value * self.downsampleRatio))
            # note: not putting in a filler value for awgDelayIndex. This is added during streaming
            #   and if that fails, a -1 flag is added afterward

    # def resolveSampleInterval(self):
    #     '''
    #     Helper function that determines the units of the target sample interval and provides a copy of the target interval
    #     that can be re-written by ps2000aRunStreaming(). Also adjusts the scopeSamples parameter to reach as the targeted
    #     experimentTime as closely as possible.
    #
    #     Args:
    #         None. Requires self.targetInterval (float) : the target scope sampling interval determined by the input experimentTime and scopeSamples
    #     Returns:
    #         0. Saves self.sampleInterval and self.sampleUnits : the target interval as an uint32, the Pico constant corresponding to the target units
    #     '''
    #     # handle case where target interval is less than 16 ns
    #     #   16 ns is minimum value that does not result in a PICO_INVALID_SAMPLE_INTERVAL error
    #     if self.targetInterval < 16e-9:
    #         print(
    #             "Warning: requested sample interval (experimentTime/scopeSamples) is less than the sampling limit of the "
    #             "Picoscope (16 ns). A 16 ns interval will be used and number of samples will be adjusted to match experimentTime")
    #         self.scopeSamples = math.floor(self.experimentTime / 16e-9)
    #         self.sampleInterval = ctypes.c_uint32(16)
    #         self.sampleUnits = ps.PS2000A_TIME_UNITS['PS2000A_NS']
    #         self.sampleUnitVals = 1e-9
    #         return 0
    #
    #     unitConstants = [ps.PS2000A_TIME_UNITS['PS2000A_S'], ps.PS2000A_TIME_UNITS['PS2000A_MS'], ps.PS2000A_TIME_UNITS['PS2000A_US'], ps.PS2000A_TIME_UNITS['PS2000A_NS']]
    #     unitVals = np.array([1, 1e-3, 1e-6, 1e-9])
    #
    #     # find the largest unit that is below the target interval
    #     # error handling done in previous step, array cannot be empty since targetInverval >= 2e-9
    #     unitIndex = np.argmax(unitVals <= self.targetInterval)
    #     self.sampleUnits = unitConstants[unitIndex]
    #     self.sampleUnitVals = unitVals[unitIndex]
    #
    #     # convert target interval to the sample units, round to nearest int and save
    #     convertedInterval = math.floor(self.targetInterval / unitVals[unitIndex])
    #     self.sampleInterval = ctypes.c_uint32(convertedInterval)
    #     self.sampleIntervalSeconds = convertedInterval * unitVals[unitIndex]
    #
    #     # determine number of samples needed to reach the target experimentTime
    #     # todo: there is a persistent error when loading data in the buffer related to rounding of the scope samples and downsampling
    #     self.scopeSamples = int(math.ceil(self.experimentTime / self.sampleIntervalSeconds))
    #
    #
    # def resolveDownsampling(self):
    #     '''
    #     Helper function that resolves values related to downsampling. Requires that resolveSampleInterval has run.
    #     Sets values for downsampling ratio and mode, sets experimentSamples and experimentTimeInterval
    #     '''
    #
    #
    #     # calculate the downsampling ratio and number of samples to be saved (experimentSamples)
    #     # based on scopeSamples and downsamplingRatio
    #     if self.downsamplingQ:
    #
    #         if self.downsamplingRatioInput == -1:
    #             # automatically resolve so that downsampled time interval is 20 us
    #             # if the sample interval is greater than 20us, this resolves to 1 and no downsampling is used
    #             self.downsamplingRatio = int(math.ceil(self.autoDownsamplingInterval / self.sampleIntervalSeconds))
    #         else:
    #             self.downsamplingRatio = int(self.downsamplingRatioInput)
    #
    #         # resolve based on input value
    #         # ceiling is used to prevent later off-by-one errors in the buffer size
    #         self.experimentSamples = int(math.ceil(self.scopeSamples / self.downsamplingRatio))
    #
    #     else:
    #         # we are not downsampling so experimentSamples is just the number of scope samples
    #         self.downsamplingRatio = 1 # other values are calculated based on this, so we need a value even though we aren't downsampling
    #         self.experimentSamples = self.scopeSamples
    #
    #     # set downsample mode
    #     if self.downsamplingQ:
    #         self.downsamplingMode = 4 # PS2000A_RATIO_MODE_AVERAGE
    #     else:
    #         self.downsamplingMode = 0 # PS2000A_RATIO_MODE_NONE
    #
    #     self.approxExperimentInterval = self.sampleIntervalSeconds * self.downsamplingRatio
    #
    #     # print a warning if experiment interval is below 25 us
    #     if self.approxExperimentInterval < 4e-6:
    #         print("Picoscope warning: experiment sampling interval implied by scopeSamples and downsamplingRatio is less "+
    #               "than 4 us. This may cause issues due to data bottlenecking. If the program hangs, the trigger pulse"+
    #               " is being missed due to this issue. Consider running at lower sample rate by lowering scopeSamples or "+
    #               "raising downsamplingRatio.")


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
        self.openUnit = ps.psospaOpenUnit(ctypes.byref(self.cHandle),
                                           None, # no need for serial number - connect to the first device discovered
                                           self.bitResolution,
                                           None) # no need for power details

        # Print plain explanations of errors
        if self.cHandle.value == -1:
            print("Picoscope failed to open. Check that it is plugged in and not in use by another program.")
        elif self.cHandle.value == 0:
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
        #   handle (int16): self.cHandle
        #   sample interval (double): self.sampleIntervalSeconds
        #   sample interval units (enum) : self.sampleIntervalUnit
        #   max pre trigger samples (uint64) : 0
        #   max post trigger samples (uint64) : self.numberOfDownsamples
        #   autoStop (int16) : 1 (yes, stop at max samples)
        #   downsample ratio (uint64) : self.downsampleRatio
        #   downsample mode (enum) : 4 (enums.PICO_RATIO_MODE[PICO_RATIO_MODE_AVERAGE])
        runStreamingStatus = ps.psospaRunStreaming(
            self.cHandle,
            ctypes.byref(self.sampleIntervalSeconds),
            self.sampleIntervalUnit,
            ctypes.c_uint64(0),
            ctypes.c_uint64(self.numberOfDownsamples),
            ctypes.c_int16(0),
            ctypes.c_uint64(self.downsampleRatio),
            4
            )

        assert_pico_ok(runStreamingStatus)
        # todo: verify that sampleInterval doesn't get messed up by rounding errors
        # todo: here's the problem: sampleInterval seems to change despite using get nearest sample interval
        #       check that that function is working as intended. check that the output is being properly assigned
        #       if we can't rely on that: need to recalculate numberOfDownsamples, use that to inform the streaming loop,
        #           and then cut off unfilled data. If resulting number of downsamples is larger than the buffers... allocate more?
        print("actual sample interval: " + str(self.sampleIntervalSeconds.value))


    def runStream(self):
        '''
        Collects data from an ongoing streaming experiment.
        Streaming data gathering loop adapted from example script in PicoSDK.

        Args: None (initStream must have been called beforehand)
        Returns: data arrays (channel A, channel B, channel C, channel D, time)
        '''

        # initialize streaming data structs. This is done here rather than initBuffers since it is faster to reference
        #   local data than class variables and speed is priority during streaming
        streamData = (structs.PICO_STREAMING_DATA_INFO * 4)()
        # data struct contents:
        #   channel (0-3)
        #   ratio mode (enum) : 4
        #   data type (enum) : enums.PICO_DATA_TYPE["PICO_INT16_T"]
        #   number of samples, buffer index, start index, overflow : initialize to 0 (used during stream)
        streamData[0] = structs.PICO_STREAMING_DATA_INFO(0, 4, enums.PICO_DATA_TYPE["PICO_INT16_T"], 0, 0, 0, 0)
        streamData[1] = structs.PICO_STREAMING_DATA_INFO(1, 4, enums.PICO_DATA_TYPE["PICO_INT16_T"], 0, 0, 0, 0)
        streamData[2] = structs.PICO_STREAMING_DATA_INFO(2, 4, enums.PICO_DATA_TYPE["PICO_INT16_T"], 0, 0, 0, 0)
        streamData[3] = structs.PICO_STREAMING_DATA_INFO(3, 4, enums.PICO_DATA_TYPE["PICO_INT16_T"], 0, 0, 0, 0)

        # trigger data struct (all initialized to 0):
        #   triggerAt (uint64)
        #   triggered (int16 / bool)
        #   autoStop (int16 / bool)
        streamTrigger = structs.PICO_STREAMING_DATA_TRIGGER_INFO(0, 0, 0)

        # create a flag for triggering delayed AWG
        awgTriggered = False
        awgTriggerIndex = 0

        # create flag to track if memory overflow issues occurred during collection
        self.memoryOverflow = False

        if self.awgStopQ:
            if self.delayQ:
                # if we are delaying the start of the AWG, put in a filler value for now
                # this will get updated to the more precise timing once the AWG trigger is sent
                awgStopIndex = self.awgTriggerSamples + self.awgRunSamples
            else:
                # awg is running without a delay so stopping is based only on the number of samples it is running for
                awgStopIndex = self.awgRunSamples
        else:
            #   WE'RE NOT STOPPING WOOOOOOO
            awgStopIndex = self.numberOfDownsamples

        # initialize indices for tracking stream progress
        saveStartIndex = [0,0,0,0] # for tracking where to save in the raw data buffers
        bufferStartIndex = [0,0,0,0] # for tracking where to grab data from the data buffers
        timeIndex = 0 # furthest save index in time, closest we have to a proxy for actual experiment time. Used for triggering AWG
        minTimeIndex = 0 # slowest save index in time. Used for stopping the collection loop
        bufferResetThreshold = math.floor(0.75 * self.numberOfDownsamples)

        # streaming loop needs to be very fast. We can get some performance gains by converting all class variables to
        #   local variables. The ones that are altered during collection will be re-saved after the loop
        numberOfDownsamples = self.numberOfDownsamples
        numberOfChannels = self.numberOfChannels
        cHandle = self.cHandle
        memoryOverflow = self.memoryOverflow # UPDATE THIS ONE AFTER LOOP
        delayQ = self.delayQ
        awgTriggerSamples = self.awgTriggerSamples
        triggerType = enums.PICO_SIGGEN_TRIG_TYPE["PICO_SIGGEN_RISING"]
        triggerIndex = -1
        timeIndex = 0
        awgRunSamples = self.awgRunSamples
        awgStopQ = self.awgStopQ
        awgStopped = False
        # flags to account for whether the primary trigger fired one the current data collection cycle
        triggered = False
        clearBufferAction = enums.PICO_ACTION["PICO_CLEAR_THIS_DATA_BUFFER"]

        # gather data in a loop
        while timeIndex < numberOfDownsamples - 1:

            # move data from hardware to buffer
            # args:
            #   cHandle
            #   pointer to streaming data info structs : streamData
            #   nStreamingDataInfos (uint64) : number of structs in streamData list (4?)
            #       todo: verify this number, its a little weird that the example only has 1
            #   pointer to trigger info struct : streamTrigger
            getValsStatus = ps.psospaGetStreamingLatestValues(cHandle, ctypes.byref(streamData), 4,
                                                              ctypes.byref(streamTrigger))


            # check for memory overflow using error codes on return
            # 268435464: "Pico Device Memory Overflow: The memory on board the device has overflowed."
            if getValsStatus == 268435464:
                memoryOverflow = True

            elif getValsStatus == 407:
                print("data buffer ran out of memory. since data buffers have 4x expected memory usage, something probably"
                      "went wrong. maybe the trigger was missed? anyway, ending collection early.")
                break

            elif getValsStatus != 0:
                print("another error message occurred during streaming:")
                print(getValsStatus)

            # check if a trigger occurred. If it did, save what index it occurred on
            if streamTrigger.triggered:
                # this should only occur once per experiment
                # only gathers data after the trigger at index
                triggered = True
                triggerIndex = bufferStartIndex[0] + streamTrigger.triggerAt

            # update the buffer indices
            for i in range(numberOfChannels):
                bufferStartIndex[i] += streamData[i].noOfSamples

            # update the timing index only if trigger occurred
            if triggered:
                timeIndex = max(bufferStartIndex) - triggerIndex

            # next handle AWG triggering separately
            if delayQ and (not awgTriggered) and (timeIndex >= awgTriggerSamples):

                # unpause the AWG and fire the software trigger
                awgRestartStatus = ps.psospaSigGenRestart(cHandle)
                assert_pico_ok(awgRestartStatus)
                awgTriggerStatus = ps.psospaSigGenSoftwareTriggerControl(cHandle, triggerType)
                assert_pico_ok(awgTriggerStatus)

                # save parameters and set some flags
                awgDelayIndex = copy(timeIndex)
                awgTriggered = True
                awgStopIndex = awgDelayIndex + awgRunSamples


            if awgStopQ and not(awgStopped) and timeIndex >= awgStopIndex:
                # need to pause the AWG or it will apply a constant voltage equal to the last value applied
                #   for the rest of the experiment
                stopStatus = ps.psospaSigGenPause(cHandle)
                assert_pico_ok(stopStatus)
                awgStopped = True

            #     for i in range(numberOfChannels):
            #         # set index bounds
            #         # number of samples to collect is bufferStart + numberOfSamples - triggerAt
            #         # buffer - start at triggerAt, go to bufferStart + numberOfSamples (I think)
            #         # raw data - start at 0, go to numberOfSamples
            #         numberOfSamples = streamData[i].noOfSamples - streamTrigger.triggerAt
            #         bufferEndIndex = bufferStartIndex[i] + streamData[i].noOfSamples
            #         bufferStart = bufferStartIndex[i] + streamTrigger.triggerAt
            #         # print(numberOfSamples)
            #         # print(streamData[i].noOfSamples)
            #         # print(streamTrigger.triggerAt)
            #         print(bufferStartIndex[i])
            #
            #         self.rawData[i][0: numberOfSamples] = self.dataBuffers[i][bufferStart: bufferEndIndex]
            #
            #         # update trackers
            #         saveStartIndex[i] = numberOfSamples
            #         bufferStartIndex[i] = bufferEndIndex
            #
            #     timeIndex = max(saveStartIndex)
            #     minTimeIndex = min(saveStartIndex)
            #
            #     # previously triggered, full returned data
            #     # streamdata: noOfSamples, bufferIndex, startIndex, overflow
            #     # iterate through channels and add data
            #     for i in range(numberOfChannels):
            #         # set index endpoints
            #         numberOfSamples = streamData[i].noOfSamples
            #         saveEnd = saveStartIndex[i] + numberOfSamples
            #         if saveEnd < numberOfDownsamples:
            #             saveEndIndex = saveStartIndex[i] + numberOfSamples
            #             bufferEndIndex = bufferStartIndex[i] + numberOfSamples
            #         else:
            #             # need to account for buffer being larger than save arrays
            #             saveEndIndex = numberOfDownsamples - 1
            #             adjustedNumberOfSamples = numberOfSamples - (saveEnd - saveEndIndex)
            #             bufferEndIndex = bufferStartIndex[i] + adjustedNumberOfSamples
            #
            #         # gather data
            #         self.rawData[i][saveStartIndex[i]: saveEndIndex] = self.dataBuffers[i][bufferStartIndex[i] : bufferEndIndex]
            #
            #         # update index trackers
            #         saveStartIndex[i] = saveEndIndex
            #         bufferStartIndex[i] = bufferEndIndex
            #
            #     timeIndex = max(saveStartIndex)
            #     minTimeIndex = min(saveStartIndex)
            #
            # elif streamTrigger.triggered:
            #     # this should only occur once per experiment
            #     # only gathers data after the trigger at index
            #     triggered = True
            #     for i in range(numberOfChannels):
            #         # set index bounds
            #         # number of samples to collect is bufferStart + numberOfSamples - triggerAt
            #         # buffer - start at triggerAt, go to bufferStart + numberOfSamples (I think)
            #         # raw data - start at 0, go to numberOfSamples
            #         numberOfSamples = streamData[i].noOfSamples - streamTrigger.triggerAt
            #         bufferEndIndex = bufferStartIndex[i] + streamData[i].noOfSamples
            #         bufferStart = bufferStartIndex[i] + streamTrigger.triggerAt
            #         # print(numberOfSamples)
            #         # print(streamData[i].noOfSamples)
            #         # print(streamTrigger.triggerAt)
            #         print(bufferStartIndex[i])
            #
            #         self.rawData[i][0: numberOfSamples] = self.dataBuffers[i][bufferStart: bufferEndIndex]
            #
            #         # update trackers
            #         saveStartIndex[i] = numberOfSamples
            #         bufferStartIndex[i] = bufferEndIndex
            #
            #     timeIndex = max(saveStartIndex)
            #     minTimeIndex = min(saveStartIndex)
            #
            # else:
            #     # before triggers, just need to update the buffer indices
            #     for i in range(numberOfChannels):
            #         bufferStartIndex[i] += streamData[i].noOfSamples



            # finally, do some memory management. If the bufferStart indices are approaching a threshold of fullness, clear them
            # for i in range(numberOfChannels):
            #     if bufferStartIndex[i] > bufferResetThreshold:
            #         bufferStatus = ps.psospaSetDataBuffer(cHandle, i, ctypes.byref(self.dataBuffers[i]),
            #                                               ctypes.c_uint64(numberOfDownsamples),
            #                                               enums.PICO_DATA_TYPE["PICO_INT16_T"],
            #                                               0, 4, clearBufferAction)
            #         assert_pico_ok(bufferStatus)

            # print(str(timeIndex) + " current | needed: " + str(numberOfDownsamples))

        # stop streaming
        stopStatus = ps.psospaStop(cHandle)
        assert_pico_ok(stopStatus)

        # move data from buffers to the data arrays
        dataStart = triggerIndex
        dataStop = triggerIndex + numberOfDownsamples
        for i in range(numberOfChannels):
            self.rawData[i] = self.dataBuffers[i][dataStart:dataStop]

        # update some class variables that were made local for the streaming loop
        self.memoryOverflow = memoryOverflow
        self.triggerIndex = triggerIndex
        if delayQ:
            self.awgDelayIndex = awgDelayIndex
            self.awgStopIndex = awgStopIndex

        # get ADC limits
        maxADC = ctypes.c_int16()
        minADC = ctypes.c_int16()
        adcLimitStatus = ps.psospaGetAdcLimits(self.cHandle, self.bitResolution, ctypes.byref(minADC), ctypes.byref(maxADC))
        assert_pico_ok(adcLimitStatus)

        # generate time data (in s)
        self.time = np.linspace(0, self.correctedExperimentTime, self.numberOfDownsamples)

        # note that the raw data are 16-bit ints. They need to be converted to 32- or 64- bit to avoid overflows
        self.channelAData = np.array(adc2mVV2(self.rawData[0], self.aRangeMax.value, maxADC))
        self.channelBData = np.array(adc2mVV2(self.rawData[1], self.bRangeMax.value, maxADC))
        self.channelCData = np.array(adc2mVV2(self.rawData[2], self.cRangeMax.value, maxADC))
        self.channelDData = np.array(self.rawData[3]) # we want the channel D data in raw ADC to judge the triggering set point

        # catch if delay was missed and put in a filler delay index to avoid later crashes
        if not hasattr(self, 'awgDelayIndex'):
            self.awgDelayIndex = -1

        if self.memoryOverflow:
            print("warning: memory overflow occurred")

        return self.channelAData, self.channelBData, self.channelCData, self.channelDData, self.time

    def initChannels(self):
        '''
        Initializes the measurement channels and triggers on the Picoscope.
        A - Photodetector 0
        B - Photodetector 1
        C - Potentiostat Out
        D - Potentiostat Trigger

        Args:
            None. Requires 'experimentTime' and 'scopeSamples' parameters in experimentParameters
        Returns:
            None
        '''
        # gather voltage ranges for each channel. Convert to nV
        self.aRangeMax = self.rangeMax(self.channelARange)
        self.bRangeMax = self.rangeMax(self.channelBRange)
        self.cRangeMax = self.rangeMax(self.channelCRange)
        self.dRangeMax = self.rangeMax(self.channelDRange)

        # min = -1 * max. there's probably a more elegant way to do this but...
        self.aRangeMin = ctypes.c_int64(-1 * self.aRangeMax.value)
        self.bRangeMin = ctypes.c_int64(-1 * self.bRangeMax.value)
        self.cRangeMin = ctypes.c_int64(-1 * self.cRangeMax.value)
        self.dRangeMin = ctypes.c_int64(-1 * self.dRangeMax.value)

        # set channel args:
        #   handle (int16): self.cHandle
        #   channel (constants 0-3 for A-D)
        #   coupling (enum) : self.coupling
        #   rangeMin (int64) : self.(a-d)RangeMin
        #   rangeMax (int64) : self.(a-d)RangeMax
        #   rangeType (enum) : 0 (PICO_PROBE_NONE_NV)
        #   analog offset (double) : 0
        #   bandwidth limiter (enum) : enums.PICO_BANDWIDTH_LIMITER["PICO_BW_FULL"]
        chAStatus = ps.psospaSetChannelOn(self.cHandle, 0, self.coupling, self.aRangeMin, self.aRangeMax,
                                          0, 0, enums.PICO_BANDWIDTH_LIMITER["PICO_BW_FULL"])
        chBStatus = ps.psospaSetChannelOn(self.cHandle, 1, self.coupling, self.bRangeMin, self.bRangeMax,
                                          0, 0, enums.PICO_BANDWIDTH_LIMITER["PICO_BW_FULL"])
        chCStatus = ps.psospaSetChannelOn(self.cHandle, 2, self.coupling, self.cRangeMin, self.cRangeMax,
                                          0, 0, enums.PICO_BANDWIDTH_LIMITER["PICO_BW_FULL"])
        chDStatus = ps.psospaSetChannelOn(self.cHandle, 3, self.coupling, self.dRangeMin, self.dRangeMax,
                                          0, 0, enums.PICO_BANDWIDTH_LIMITER["PICO_BW_FULL"])

        assert_pico_ok(chAStatus)
        assert_pico_ok(chBStatus)
        assert_pico_ok(chCStatus)
        assert_pico_ok(chDStatus)

        # set trigger. For now, triggering on Channel D - idea is to trigger when potentiostat starts chronoamperometry
        # args:
        #   handle (int16)
        #   enable (1)
        #   source (3 - Channel D)
        #   threshold (int16 - will need to play with this a bit)
        #       for 20V range, 1V is roughly an ADC of 1600
        #   direction 0 (ABOVE)
        #   delay (uint32 - ignored for data collection)
        #   autoTrigger_us (int32 - 10000s - trigger after 2 s)
        triggerStatus = ps.psospaSetSimpleTrigger(self.cHandle, 1, 3, 500000, 0, 0, ctypes.c_uint32(20000))

        assert_pico_ok(triggerStatus)

    @staticmethod
    def rangeMax(requestedVoltageRange):
        '''
        Helper function that rounds an input voltage range (in V) to the nearest allowed value in nV.
        Returns a 64 bit int for use in later functions

        Args:
            requestedVoltageRange (float) : the input value from experimental parameters, in V
        Returns:
            int64 : the nearest allowable voltage range above the requested value, converted to nV
        '''
        # voltageLimits taken from Picoscope3415E spec sheet
        voltageLimits = np.array([0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20])

        try:
            voltageRange = voltageLimits[voltageLimits >= requestedVoltageRange][0]
        except IndexError:
            raise ValueError("Input voltage range exceeds the 20V limit. Verify the input is correct. Do not use the"
                               "Picoscope if expecting inputs exceeding 20V.")

        return ctypes.c_int64(int(1e9 * voltageRange))

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

        # voltageLimits taken from Picoscope3415E spec sheet
        voltageLimits = np.array([0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20])
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
        Initializes the arbitrary waveform generator (AWG). Gathers input parameters and calls sig gen set up functions:
        # psospaSigGenRange - sets peakTopeak and offset
        # psospaSigGenTrigger - set to software control (use rising type), sets number of cycles
        # psospaSigGenWaveform - set wavetype to arbitrary, fill in values in range -32767 to 32767
        # psospaSigGenFrequency - sets frequency. need to think about how to gather this from vt func
        #   check w/ psospaSigGenFrequencyLimits
        # psospaSigGenApply - applies all of the settings from prev functions
        #       disable sweep, set frequency, do not use stopFreq, freqIncrement, and dwellTime
        Args:
            None. Requires AWG parameters in the experimental params were properly filled out
        Returns:
            0 if successful, else -1
        '''

        # resolve certain parameters based on awgFunc input
        if self.awgFunc == None:
            # skip this whole step if not using AWG, filling in dummy data to return for later operations
            self.awgTime = np.linspace(0, self.correctedExperimentTime, self.numberOfDownsamples)
            self.awg = np.zeros(self.numberOfDownsamples)
            self.delaySamples = 0
            return 0
        # if using AWG, resolve what wave type to use
        elif self.awgFunc == 'picoSine':
            self.awgWaveType = 0x00000011
        elif self.awgFunc == 'picoSquare':
            self.awgWaveType = 0x00000012
        else:
            self.awgWaveType = 0x10000000

        # calculate the number of shots needed to run for awgDuration time. Print a warning if the amount exceeds 2e64-1 (unlikely)
        rawShots = max(math.floor(self.awgDuration / self.awgPeriod), 1)

        if rawShots > 2e64-1:
            self.awgShots = 0 # setting to 0 runs continuously
            self.awgDurationAdjusted = self.awgShots * self.awgPeriod
            print("AWG Warning: number of voltage function periods implied by awgPeriod and experimentTime settings exceeds " +
                  "the amount possible using the AWG (2e64-1). Setting to run continuously. Maybe chill out on the data collection?")
        else:
            self.awgShots = rawShots
            self.awgDurationAdjusted = self.awgShots * self.awgPeriod

        # set up sig gen trigger depending on whether AWG start is delayed from experiment start
        #   when not delayed, we can use the scope trigger, but when delayed we need to send a software trigger
        #   during the streaming loop
        if self.delayQ:
            self.triggerSource = 4 # software trigger if AWG trigger is separate from rest of scope
        else:
            self.triggerSource = 1 # no delay so use scope trigger

        sigGenTriggerStatus = ps.psospaSigGenTrigger(
            self.cHandle,
            0, # RISING
            self.triggerSource, # 4 for software, 1 for scope
            ctypes.c_uint64(self.awgShots), # cycles. unsure if this correct
            0 # not used
        )
        assert_pico_ok(sigGenTriggerStatus)

        # need to set AWG parameters depending on the wave type
        if self.awgWaveType == 0x00000011:
            # first do the built-in sine function
            self.setAWGFreq()
            self.setAWGRange()
            self.setAWGWaveform()
        elif self.awgWaveType == 0x00000012:
            # square wave. need to add duty cycle
            self.setAWGFreq()
            self.setAWGRange()
            self.setAWGDuty()
            self.setAWGWaveform()
        else:
            # arbitrary waveform, need to set frequency and buffer
            self.setAWGBuffer()
            self.setAWGFreq()
            self.setAWGWaveform()

        # set up sig gen using SigGenApply
        # first initialize parameters to gather actual used values
        # we only care about the frequency since we aren't sweeping
        self.actualAWGFreq = ctypes.c_double()
        actualStopFreq = ctypes.c_double()
        actualFreqIncrement = ctypes.c_double()
        actualDwellTime = ctypes.c_double()

        sigGenApplyStatus = ps.psospaSigGenApply(
            self.cHandle,
            1, # sig gen enabled. May want to default to off instead if turned on by trigger
            0, # sweep not enabled
            1, # trigger enabled
            ctypes.byref(self.actualAWGFreq),
            ctypes.byref(actualStopFreq),
            ctypes.byref(actualFreqIncrement),
            ctypes.byref(actualDwellTime)
        )
        assert_pico_ok(sigGenApplyStatus)

        # need to pause immediately or the AWG will apply a constant voltage equal to the first value until it starts
        pauseStatus = ps.psospaSigGenPause(self.cHandle)
        assert_pico_ok(pauseStatus)

        # todo: delete once tested
        # else:
        #     # no delay so use the scope trigger
        #     sigGenTriggerStatus = ps.psospaSigGenTrigger(
        #         self.cHandle,
        #         0, # RISING
        #         4, # Software trigger
        #         ctypes.c_uint64(self.awgShots), # cycles. unsure if this correct
        #         0 # not used
        #     )
        #
        #     sigGenStatus = ps.ps2000aSetSigGenArbitrary(
        #         self.cHandle,  # scope identifier, int16
        #         self.awgOffset,  # offsetVoltage = 0 (int32) - set in generateAWGBuffer
        #         self.pkToPk,  # peak to peak in uV,  uint32 - calculated in generateAWGBuffer
        #         self.startDeltaPhase,  # (uint32) - calculated in generateAWGBuffer
        #         self.startDeltaPhase,
        #         # stopDeltaPhase = startDeltaPhase (uint32) - only differs from start when sweeping
        #         0,  # deltaPhaseIncrement = 0 (uint32) - only non-zero when sweeping
        #         3,
        #         # dwellCount (uint32) = ? (uint32)  - how long each step lasts when frequency sweeping. Set to minimum value (3), does not actually matter if not sweeping
        #         ctypes.byref(self.waveformBuffer),
        #         # arbitraryWaveform = pointer to uint32 buffer -  voltage samples for the input awgFunc
        #         ctypes.c_int32(self.numberOfPoints),  # arbitraryWaveformSize = numberOfPoints (int32)
        #         ctypes.c_int32(0),  # sweepType = PS2000A_UP (shouldn't matter if not sweeping)
        #         0,  # # operation = PS2000A_ES_OFF (normal operation)
        #         0,
        #         # indexMode = PS2000A_SINGLE (waveform buffer fully specifies signal, it isn't half of a mirrored signal)
        #         ctypes.c_uint32(self.awgShots),
        #         # ctypes.c_uint32(self.awgShots), # number of repeats of the signal (implied by awgPeriod and experimentTime)
        #         # setting to max 0xFFFFFFFF runs continuously
        #         0,  # sweeps = 0  (we're doing a set number of shots, not sweeps)
        #         ctypes.c_int32(0),  # triggerType = Rising (Enables simple control with sigGenSoftwareControl())
        #         ctypes.c_int32(4),# Trigger source - software. WHY THE FUCK IS IT 4 AND NOT 3?!? NEED TO FIGURE OUT HOW TO ACTUALLY USE CONSTANT ENUMS
        #         1  # extInThreshold  (not using external trigger, doesn't matter)
        #     )
        #
        # else:
        #     # No delay requested, so use scope trigger
        #     self.delaySamples = 0
        #     self.awgDelayIndex = 0
        #     sigGenStatus = ps.ps2000aSetSigGenArbitrary(
        #         self.cHandle,  # scope identifier, int16
        #         self.awgOffset,  # offsetVoltage = 0 (int32) - set in generateAWGBuffer
        #         self.pkToPk,  # peak to peak in uV,  uint32 - calculated in generateAWGBuffer
        #         self.startDeltaPhase,  # (uint32) - calculated in generateAWGBuffer
        #         self.startDeltaPhase,
        #         # stopDeltaPhase = startDeltaPhase (uint32) - only differs from start when sweeping
        #         0,  # deltaPhaseIncrement = 0 (uint32) - only non-zero when sweeping
        #         3,
        #         # dwellCount (uint32) = ? (uint32)  - how long each step lasts when frequency sweeping. Set to minimum value (3), does not actually matter if not sweeping
        #         ctypes.byref(self.waveformBuffer),
        #         # arbitraryWaveform = pointer to uint32 buffer -  voltage samples for the input awgFunc
        #         ctypes.c_int32(self.numberOfPoints),  # arbitraryWaveformSize = numberOfPoints (int32)
        #         ctypes.c_int32(0),  # sweepType = PS2000A_UP (shouldn't matter if not sweeping)
        #         0,  # # operation = PS2000A_ES_OFF (normal operation)
        #         0,
        #         # indexMode = PS2000A_SINGLE (waveform buffer fully specifies signal, it isn't half of a mirrored signal)
        #         ctypes.c_uint32(self.awgShots),
        #         # ctypes.c_uint32(self.awgShots), # number of repeats of the signal (implied by awgPeriod and experimentTime)
        #         # setting to max 0xFFFFFFFF runs continuously
        #         0,  # sweeps = 0  (we're doing a set number of shots, not sweeps)
        #         ctypes.c_int32(0),  # triggerType = Rising? (hoping this is ignored when using scope trigger)
        #         ctypes.c_int32(1),  # triggerSource = 1 (PS2000A_SIGGEN_SCOPE_TRIG)
        #         1  # extInThreshold  (not using external trigger, doesn't matter)
        #     )
        #
        # assert_pico_ok(sigGenStatus)

    def setAWGBuffer(self):
        '''
        Converts voltage/time function describing the voltage profile to be applied through the potentiostat into a properly
        formatted buffer to be used as the arbitraryWaveform input to psospaSigGen
        Uses input to call psospaSigGenRange and SigGenWaveform

        Args:
            None, requires the following params in the input params dict:
                awgFunc (callable) : a function which outputs a voltage (in V) for an array of times input (in s)
                awgPeriod (float) : specify the time range to generate outputs for awgFunc
                *funcArgs, **funcKwargs : additional args and kwargs to pass into awgFunc, if needed

        Returns:
            -1 for error, 0 for normal operation

        How is the wave represented:
            arbitraryWaveform is a buffer (array) of data, where each sample (point) is a value directly proportional to
            the voltage to be output
            The AWG steps through the buffer at a certain frequency and outputs the voltage based on the sample value in
            the buffer
        The voltage value is calculated from the buffer value by
            vout = (pkToPk / 2) * (sample_val/32767) + offsetVoltage
            pkToPk : peak-to-peak (i.e. max - min) of the wave, in volts
            offsetVoltage : constant added/subtracted, in volts
            vout is always clipped to +- 2V (2e6 since in units of uV)
        :return:
        '''

        # create linspace of times, use to output voltages from awgFunc
        # todo: check if end bound should be awgPeriod or awgPeriod - (1 sample period)
        self.awgTime = np.linspace(0, self.awgPeriod, self.awgSamples)
        self.rawAWG = self.awgFunc(self.awgTime, *self.awgFuncArgs, **self.awgFuncKwargs)

        # cut off awg to fit between max and min allowed voltage
        ceilArray = np.where(self.rawAWG <= self.maxAWGVolts, self.rawAWG, self.maxAWGVolts)
        self.awg = np.where(ceilArray >= self.minAWGVolts, ceilArray, self.minAWGVolts)

        # convert values to awg buffer sample values
        # formula in SDK is vout = 1uV * (pkToPk / 2) * (sample value / 32767) + offsetVoltage
        #   This scales the Vout values to the peak-to-peak value and divides into a 16-bit int (32767 = 2e15-1, one bit is sign)
        # We will set the offset to be max + min / 2 so that the waveform is optimally positioned in the middle of the range
        # sample value = (65534 * (vout - offset)) /  pkToPk
        self.pkToPk = abs(np.max(self.awg) - np.min(self.awg))  # abs since pk to pk should be positive
        self.awgOffset = (np.max(self.awg) + np.min(self.awg)) / 2 # offset is mean of peaks

        # set peak to peak and offset
        sigGenRangeStatus = ps.psospaSigGenRange(self.cHandle, ctypes.c_double(self.pkToPk),
                                                 ctypes.c_double(self.awgOffset))
        assert_pico_ok(sigGenRangeStatus)

        # generate and set the waveform buffer
        bufferValSpan = int(self.maxBufferVal.value - self.minBufferVal.value)
        waveform = (bufferValSpan * (self.awg - self.awgOffset)) / self.pkToPk

        # cut off any value in the waveofmr that exceeds limits
        #   this should already be safe from the previous cutoff but this avoids any rounding errors
        ceilArray = np.where(waveform <= self.maxBufferVal.value, waveform, self.maxBufferVal.value)
        floorArray = np.where(ceilArray >= self.minBufferVal.value, ceilArray, self.minBufferVal.value).astype(ctypes.c_int16)

        self.awgBuffer = np.ctypeslib.as_ctypes(floorArray)



    # def generateAWGBuffer(self):
    #     '''
    #     Converts voltage/time function describing the voltage profile to be applied through the potentiostat into a properly
    #     formatted buffer to be used as the arbitraryWaveform input to psospaSigGen
    #     Uses input to call psospaSigGenRange and SigGenWaveform
    #
    #     Args:
    #         None, requires the following params in the input params dict:
    #             awgFunc (callable) : a function which outputs a voltage (in V) for an array of times input (in s)
    #             awgPeriod (float) : specify the time range to generate outputs for awgFunc
    #             *funcArgs, **funcKwargs : additional args and kwargs to pass into awgFunc, if needed
    #
    #     Returns:
    #         -1 for error, 0 for normal operation
    #
    #     How is the wave represented:
    #         arbitraryWaveform is a buffer (array) of data, where each sample (point) is a value directly proportional to
    #         the voltage to be output
    #         The AWG steps through the buffer at a certain frequency and outputs the voltage based on the sample value in
    #         the buffer
    #       The voltage value is calculated from the buffer value by
    #             vout = 1uV * (pkToPk / 2) * (sample_val/32767) + offsetVoltage
    #             pkToPk : peak-to-peak (i.e. max - min) of the wave, in volts
    #             offsetVoltage : constant added/subtracted, in volts
    #             vout is always clipped to +- 2V (2e6 since in units of uV)
    #
    #     :param wavefunc:
    #     :return:
    #     '''
        # todo: delete this once everything is tested
        # calculate the number of time points to sample the awgFunc, using the minimum sample period (50 ns) or the
        #   maximum number of points, whichever smaller
        # self.numberOfPoints = min(math.floor(self.awgSamples), self.maxBufferSize.value)
        #
        # # error check the number of points
        # if self.numberOfPoints < self.minBufferSize.value:
        #     print("functionToArbitraryWaveform: not enough time points specified. Inputs imply " +
        #           str(self.numberOfPoints) + " points but AWG requires " + str(self.minBufferSize) +
        #           " points. awgPeriod value is too low to effectively sample.")
        #     return -1
        # elif self.numberOfPoints > self.maxBufferSize.value:
        #     # this condition SHOULD be impossible
        #     print("functionToArbitraryWaveform: too many time points specified. Inputs imply " +
        #           str(self.numberOfPoints) + " points but AWG has a maximum of " + str(self.maxBufferSize.value) +
        #           " points. It is unclear how this error could have happened, but consider adjusting awgPeriod to fix.")
        #     return -1

        #
        # self.startDeltaPhase = ctypes.c_uint32()  # value will be written by next line
        #
        # # inputs:
        # #   chandle
        # #   frequency (double)
        # #   indexMode (built in constant)
        # #   bufferLength (uint32)
        # #   phase (output, uint32 pointer)
        # phaseStatus = ps.ps2000aSigGenFrequencyToPhase(
        #     self.cHandle,
        #     ctypes.c_double(targetFreq),
        #     0, # INDEX_MODE = SINGLE
        #     ctypes.c_uint32(self.numberOfPoints),
        #     ctypes.byref(self.startDeltaPhase)
        # )
        #
        # # handle errors
        # assert_pico_ok(phaseStatus)
        #
        # # create linspace of times, use to output voltages from awgFunc
        # self.awgTime = np.linspace(0, self.awgPeriod, self.awgSamples)
        # voltages = self.awgFunc(self.awgTime, *self.awgFuncArgs, **self.awgFuncKwargs) * 1e6  # convert to uV
        # self.awg = voltages / 1e6 # used for saving later. Divided by 1e6 to convert from uV to V
        #
        # # convert values to awg buffer sample values
        # # formula in SDK is vout = 1uV * (pkToPk / 2) * (sample value / 32767) + offsetVoltage
        # #   This scales the Vout values to the peak-to-peak value and divides into a 16-bit int (32767 = 2e15-1, one bit is sign)
        # # We will set the offset to be max + min / 2 so that the waveform is optimally positioned in the middle of the range
        # # sample value = (65534 * (vout - offset)) /  pkToPk
        # self.pkToPk = ctypes.c_uint32(math.floor(np.max(voltages) - np.min(voltages)))  # peak-to-peak voltage rounded to nearest uV
        # self.awgOffset = ctypes.c_int32(math.floor((np.max(voltages) + np.min(voltages))/2))
        #
        # waveform = (65534 * (voltages  - self.awgOffset.value)) / self.pkToPk.value
        #
        # # to avoid overflow, need to cut max and min values to 32767 and -32768.
        # ceilArray = np.where(waveform <= 32767, waveform, 32767)
        # floorArray = np.where(ceilArray >= -32768, ceilArray, -32768).astype(ctypes.c_int16)
        #
        # self.waveformBuffer = np.ctypeslib.as_ctypes(floorArray)
        # self.awgBuffer = self.waveformBuffer # renaming for later saving
        #
        # return 0

    def setAWGWaveform(self):
        '''
        Formats call to psospaSigGenWaveform depending on the user input awg function
        If a built-in sine or square wave, uses those. Also puts in filler values for self.awg and self.awgTime
            for downstream processing
        For arbitrary profiles, it also uses the buffers specified in setAWGBuffer()
        :return:
        '''
        if type(self.awgFunc) == str:
            # using a built-in function, so no argument used for buffer or buffer length
            sigGenWaveformStatus = ps.psospaSigGenWaveform(self.cHandle,
                                                           self.awgWaveType,  # PICO_ARBITRARY
                                                           None,
                                                           ctypes.c_uint64(0))

            # generate filler sine and square waves for later use
            if self.awgWaveType == 0x00000011: #sine
                self.awgTime = np.linspace(0, self.awgPeriod, 100)
                self.awg = sinWave(self.awgTime, **self.awgFuncKwargs)
            elif self.awgWaveType == 0x00000012: #square
                self.awgTime = np.linspace(0, self.awgPeriod, 100)
                sqKwargs = copy(self.awgFuncKwargs)
                sqKwargs['delayQ'] = self.delayQ
                self.awg = squareWave(self.awgTime, **sqKwargs)
        else:
            # using an arbitrary waveform, need to use buffer spedified in setAWGBuffer()
            sigGenWaveformStatus = ps.psospaSigGenWaveform(self.cHandle,
                                                           self.awgWaveType, # PICO_ARBITRARY
                                                           ctypes.byref(self.awgBuffer),
                                                           ctypes.c_uint64(self.awgSamples))
        assert_pico_ok(sigGenWaveformStatus)

    def setAWGRange(self):
        '''
        Sets peak-to-peak and offset for built-in AWG functions. Does some basic error checking and then calls
        psospaSigGenRange() based on inputs in 'awgFuncKwargs' experimental parameter
        '''
        amp = self.awgFuncKwargs['amp']
        offset = self.awgFuncKwargs['offset']

        if amp + offset > self.maxAWGVolts.value or offset - amp < self.minAWGVolts.value:
            raise ValueError("setAWGRange: specified combination of amplitude and range are outside the AWG limits (" +
                             str(self.minAWGVolts.value) + " - " + str(self.maxAWGVolts.value) + " V).")

        rangeStatus = ps.psospaSigGenRange(self.cHandle,
                                           ctypes.c_double(amp), # peak - to - peak
                                           ctypes.c_double(offset)) # voltage offset
        assert_pico_ok(rangeStatus)

    def setAWGDuty(self):
        '''
        Sets the duty cycle for the built-in square wave. Converts input duty kwarg (0 -1 ) to a percent first
        '''

        dutyPercent = self.awgFuncKwargs['duty'] * 100
        dutyStatus = ps.psospaSigGenWaveformDutyCycle(self.cHandle,
                                                      ctypes.c_double(dutyPercent))
        assert_pico_ok(dutyStatus)

    def setAWGFreq(self):
        '''
        Resolves the AWG frequency based on user inputs and hardware limits, then sets the value

        Args:
            self. Requires class variables awgPeriod, cHandle
        Returns:
            None. Sets self.awgFreq
        '''
        # resolve the AWG frequency based on inputs and hardware limits using psospaSigGenFrequencyLimits
        targetFreq = (1 / self.awgPeriod)
        minFreq = ctypes.c_double()
        maxFreq = ctypes.c_double()
        # other required parameters we won't use since we aren't sweeping
        minFreqStep = ctypes.c_double()
        maxFreqStep = ctypes.c_double()
        minDwell = ctypes.c_double()
        maxDwell = ctypes.c_double()
        freqLimitsStatus = ps.psospaSigGenFrequencyLimits(self.cHandle,
                                                          self.awgWaveType,
                                                          ctypes.byref(ctypes.c_uint64(self.awgSamples)),
                                                          ctypes.byref(minFreq),
                                                          ctypes.byref(maxFreq),
                                                          ctypes.byref(minFreqStep),
                                                          ctypes.byref(maxFreqStep),
                                                          ctypes.byref(minDwell),
                                                          ctypes.byref(maxDwell))
        assert_pico_ok(freqLimitsStatus)

        # check and adjust target frequency if out of bounds, print a warning
        if targetFreq < minFreq.value:
            print("AWG Warning: target AWG frequency is too low. AWG will run at " + str(minFreq.value) + " Hz")
            self.awgFreq = minFreq
        elif targetFreq > maxFreq.value:
            print("AWG Warning: target AWG frequency is too high. AWG will run at " + str(maxFreq.value) + " Hz")
            self.awgFreq = maxFreq
        else:
            self.awgFreq = ctypes.c_double(targetFreq)

        # set the frequency
        freqStatus = ps.psospaSigGenFrequency(self.cHandle, self.awgFreq)
        assert_pico_ok(freqStatus)


    def initDataBuffers(self):
        '''
        Allocate space for the data buffers and full data arrays that the scope will send data to,
        then associate the computer buffers with the driver

        Streaming mode requires both a data buffer and a full data array - if the requested amount of data is larger than
        the memory on the picoscope, it needs to be gathered and transferred in chunks to the data buffer. The data buffer
        is copied into the full data array and then overwritten by the next chunk of data. The streaming data can become
        discontinuous if the scope memory fills faster than the buffer can be copied to computer memory.

        Args:
            None. Requires self.numberOfDownsamples and for all channels to be initialized
        Returns:
            None. self.dataBufferA/B/C and self.dataA/B/C are saved as class variables
        '''

        # allocate streaming buffers
        self.dataBuffers = [np.ctypeslib.as_ctypes(np.zeros(self.numberOfDownsamples * 4, dtype=ctypes.c_int16)) for i in range(self.numberOfChannels)]

        # arrays for copying down raw data during streaming
        self.rawData = [np.zeros(self.numberOfDownsamples, dtype = ctypes.c_int16) for i in range(self.numberOfChannels)]

        # call setDataBuffer
        # args:
        #   handle : self.cHandle
        #   channel : 0-3
        #   buffer (pointer) : pointer to streaming buffer
        #   nSamples (uint64) : numberOfDownsamples
        #   data type (enum) : enums.PICO_DATA_TYPE["PICO_INT16_T]
        #   waveform (uint64) : segment index (0, we aren't segmenting)
        #   ratio mode (enum) : downsampling mode. 4 for average
        #   action (enum) : method to create buffer
        #           clear then add for first buffer, then add for the rest
        #           enums.PICO_ACTION["PICO_CLEAR_ALL"] | enums.PICO_ACTION["PICO_ADD"]
        for i in range(self.numberOfChannels):
            if i == 0:
                # for first buffer, need to clear memory. after that we can just add
                bufferStatus = ps.psospaSetDataBuffer(self.cHandle, i, ctypes.byref(self.dataBuffers[i]),
                                               ctypes.c_uint64(self.numberOfDownsamples * 4), enums.PICO_DATA_TYPE["PICO_INT16_T"],
                                               0, 4, enums.PICO_ACTION["PICO_CLEAR_ALL"] | enums.PICO_ACTION["PICO_ADD"])
            else:
                bufferStatus = ps.psospaSetDataBuffer(self.cHandle, i, ctypes.byref(self.dataBuffers[i]),
                                       ctypes.c_uint64(self.numberOfDownsamples * 4), enums.PICO_DATA_TYPE["PICO_INT16_T"],
                                       0, 4, enums.PICO_ACTION["PICO_ADD"])
            assert_pico_ok(bufferStatus)

        # bufferAStatus = ps.psospaSetDataBuffer(self.cHandle, 0, ctypes.byref(self.channelABuffer),
        #                                        ctypes.c_uint64(self.numberOfDownsamples), enums.PICO_DATA_TYPE["PICO_INT16_T"],
        #                                        0, 4, enums.PICO_ACTION["PICO_CLEAR_ALL"] | enums.PICO_ACTION["PICO_ADD"])
        # bufferBStatus = ps.psospaSetDataBuffer(self.cHandle, 1, ctypes.byref(self.channelBBuffer),
        #                                        ctypes.c_uint64(self.numberOfDownsamples), enums.PICO_DATA_TYPE["PICO_INT16_T"],
        #                                        0, 4, enums.PICO_ACTION["PICO_ADD"])
        # bufferCStatus = ps.psospaSetDataBuffer(self.cHandle, 2, ctypes.byref(self.channelCBuffer),
        #                                        ctypes.c_uint64(self.numberOfDownsamples), enums.PICO_DATA_TYPE["PICO_INT16_T"],
        #                                        0, 4, enums.PICO_ACTION["PICO_ADD"])
        # bufferDStatus = ps.psospaSetDataBuffer(self.cHandle, 3, ctypes.byref(self.channelDBuffer),
        #                                        ctypes.c_uint64(self.numberOfDownsamples), enums.PICO_DATA_TYPE["PICO_INT16_T"],
        #                                        0, 4, enums.PICO_ACTION["PICO_ADD"])
        #
        # assert_pico_ok(bufferAStatus)
        # assert_pico_ok(bufferBStatus)
        # assert_pico_ok(bufferCStatus)
        # assert_pico_ok(bufferDStatus)


    # def streamingCallback(self, handle, numberOfSamples, startIndex, overflow, triggerAT, triggered, autoStop,
    #                       pParameter):
    #     '''
    #     Function that is called by ps2000aGetStreamingLatestValues every time it returns in order to move data from the
    #     streaming buffer into memory. Adapted from example code in PicoSDK.
    #     Triggering for data collection and AWG is handled here
    #
    #     Args:
    #         None. Requires that data arrays and buffers were properly set up
    #     Returns:
    #         None. Values copied to data arrays, nextSample, wasCalledBack, and autoStopOuter are updated
    #     '''
    #
    #     # self.triggered is used to track whether the scope was triggered at any point during the experiment
    #     #   versus triggered (argument) refers to whether the trigger occurs on this particular callback
    #     self.triggered = self.triggered or triggered
    #
    #     self.wasCalledBack = True
    #
    #     # this section is included in the example but really not sure what it is doing :(
    #     if autoStop:
    #         self.autoStopOuter = True
    #
    #     #todo: consistently getting value errors here where buffer array lengths are not matching
    #     #   I THINK this is a issue with rounding - buffer size is getting rounded vs the streaming length?
    #     # save data in two cases: was previously triggered or triggered in this callback
    #     if self.triggered and not triggered:
    #         destEnd = self.nextSample + numberOfSamples
    #         sourceEnd = startIndex + numberOfSamples
    #         self.channelARawData[self.nextSample: destEnd] = self.channelABuffer[startIndex: sourceEnd]
    #         self.channelBRawData[self.nextSample: destEnd] = self.channelBBuffer[startIndex: sourceEnd]
    #         self.channelCRawData[self.nextSample: destEnd] = self.channelCBuffer[startIndex: sourceEnd]
    #         self.channelDRawData[self.nextSample: destEnd] = self.channelDBuffer[startIndex: sourceEnd]
    #         self.nextSample += numberOfSamples
    #
    #     elif self.triggered and triggered:
    #         # triggered on this callback. Only want data after startIndex + triggeredAT
    #         sourceStart = startIndex + triggerAT
    #         sourceEnd = startIndex + numberOfSamples
    #         triggeredSamples = sourceEnd - sourceStart
    #         destEnd = self.nextSample + triggeredSamples
    #
    #         self.channelARawData[self.nextSample: destEnd] = self.channelABuffer[sourceStart: sourceEnd]
    #         self.channelBRawData[self.nextSample: destEnd] = self.channelBBuffer[sourceStart: sourceEnd]
    #         self.channelCRawData[self.nextSample: destEnd] = self.channelCBuffer[sourceStart: sourceEnd]
    #         self.channelDRawData[self.nextSample: destEnd] = self.channelDBuffer[sourceStart: sourceEnd]
    #         self.nextSample += triggeredSamples

    def closePicoscope(self):
        '''
        Closes connection to Picoscope.

        Args:
            None. Requires self.cHandle
        Returns:
            None. Error is raised if close operation is not successful
        '''
        closeStatus = ps.psospaCloseUnit(self.cHandle)
        assert_pico_ok(closeStatus)

# a test input function for the AWG
def testVT(times, freq = 100, amp = 0.1):
    return amp * np.sin(2 * np.pi * times * freq)

# copy this over from experiments to avoid circular import of module. there is probably a better way to handle this...
def squareWave(times : np.ndarray, freq = 100, amp = 0.1, offset = 0, duty = 0.5, delayQ = False):
    '''
    Returns a square wave voltage.

    Args:
        times (array): input time array
        freq (float): frequency of square wave
        amp (float): difference between minimum and maximum voltage / 2. Must be less than 2
            Note: Positive amp results in starting at the maximum voltage, while a negative amp starts at negative voltage
        offset (float) : constant added to wave to offset the average value from 0
        duty (float) : fraction of wave period spent at the voltage maximum. Must be between 0 and 1
        delayQ (bool) : will this function be used with a nonzero awgDelay parameter. If so, the first and last values will
                        be set to 0 to avoid outputting a constant nonzero voltage before and after running
    Returns:
        array : voltages of square wave of length equal to times
    '''

    # scipy implementation has a default period of 2pi so time input is stretched by factor of freq to match input
    # output value is -1 to +1, starting at +1
    # need to handle positive and negative amp separately since the meaning of duty gets reversed with negative amp
    if amp > 0:
        sqBase = square(2 * np.pi * freq * times, duty)
    elif amp < 0:
        sqBase = -1 * square(2 * np.pi * freq * times, 1 - duty)
    else:
        # handle case where amp is 0 because the user is being silly
        sqBase = np.zeros(len(times))

    # next, scale by abs(amp) since the sign was handled previously
    # then add the offset value, handle delay, and return
    output = (abs(amp) * sqBase) + offset

    if delayQ:
        output[0] = 0
        output[-1] = 0
        return output
    else:
        return output

def sinWave(times, freq=100, amp=0.1, offset=0):
    '''
    Returns a sine wave voltage profile

    :param times:
    :param freq:
    :param amp:
    :param offset:
    :return:
    '''
    return amp * np.sin(2 * np.pi * times * freq) + offset