# code infrastructure to run and analyze high frequency electrochemiluminescence experiments
# Hardware to interface with:
#   Biologic SP-300 (EC-Lab Developer Package)
#   Picoscope 2405A (PicoSDK)
# Control loop:
#   Connect to potentiostat / oscilloscope
#   Initialize experiments:
#       Chronoamperometry
#       Generate AWG
#       Trigger/collect scope data
#       Experiment types:
#           EIS, CV (for testing purposes)
#           Single ECL (ECL with single waveform)
#           Multi ECL (ECL with a list of waveforms)
#           Optimize ECL (change the ECL waveforms to optimize an objective function)
#       Save data efficiently (figure out how to handle different time steps of potentiostat vs scope data)
#   Analyze data:
#       Sync current/voltage/photovoltage
#   Control loop:
#       Generate new AWG conditions and rerun experiment

import numpy as np
import picoscope as pico

# Experiment parameters. Will be added anc sorted as they become apparent
experimentParameters = {
    # Oscilloscope params
    'vtFunc' : np.sin, # Callable, function that can input an array of time values and output voltages at those times
    'vtFuncArgs' : (), # Tuple, additional positional arguments that will be passed to vtFunc when it is evaluated
    'vtFuncKwargs' : {}, # Dict, additional keyword arguments that will be passed to vtFunc when it is evaluated
    'vtPeriod' : 1, # Duration in seconds of the voltage function. That max AWG frequency is 20 MHz, so minimum tStop is 5e-8 seconds
    'tStep' : 0.001, # Time step that the vtFunc will be sampled at. Minimum value is also 5e-8 seconds todo: check this
    #todo: add vtPeriods to specify how many times the AWG should repeat, with -1 repeating until experimentTime
    'experimentTime' : 5, # Duration that the vtFunc will be applied and photodetector / potentiostat measurements are done
    'scopeSamples' : 10000, # Number of voltage points that will be collected by the oscilloscope during experimentTime
                           # NOTE: Instrument limit is 2ns/sample. That limit may be higher with memory constraints
                            # and communication bottlenecks from streaming mode (48 kS memory -> need to send data every ~10kS).
                            # NOTE: the combination of time and samples will request a specific sampling interval. The actual
                            # sampling interval will be determined by communication speeds in streaming mode. The actual value will be saved
                            # TODO: determine empirical limits and add them here
    'detectorVoltageRange0' : 0.02, # maximum voltage expected on photodetector 0 (Channel A)
    'detectorVoltageRange1' : 0.02, # maximum voltage expected on photodetector 1 (Channel B)
    'potentiostatVoltageRange' : 5, # maximum voltage expected on the potentiostat output (Channel C)
                                    # Allowed values are 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20
                                    # Improper inputs will be rounded up to the nearest allowed value
    #todo: add voltage offsets? Might be useful for the photodetectors

    # potentiostat parameters
    'experimentType' : 'ocv', # str - 'ocv' or 'ca' for open circuit voltage or chronoamperometry
    'potentiostatChannel' : 1, # int - 1 or 2, channel on used on potentiostat
    'currentRange' : 8, # int 0-9, determines the current range in powers of 10 from 0 (100 pA), 1 (1 nA),... 7 (1 mA), 8 (10 mA)
                        # WARNING: auto range is not available. Selecting too low of a range WILL damage the potentiostat
    'dt': 0.1,  # float, time interval for recording current. Note that the picoscope will record much faster than this (if used)
    'trigger' : False, # boolean, should a trigger be sent out when the trigger is run
    'extInput' : False, # boolean, enable external voltage (i.e. picoscope AWG)

    # OCV params
    'duration' : 1.0, # float, duration in seconds


    # CA params
    'vStep' : [1.0], # voltage at each step
    'vStepTime' : [1.0], # time spent at each step. len(vStep) must equal len(vStepTime)
    'startStep' : 0, # step index to start on. Must be less than len(vStep) - 1
    'numberOfCycles' : 1 # number of times CA is repeated
}

def runExperiment(params : dict):
    '''
    Main control loop for running the ECL experiment

    Args:
        params (dict) : the input parameters to the experiment

    Returns:
        0 : data is saved, nothing returned?
    '''
    scope = pico.Picoscope(params)


    return 0

# runExperiment(experimentParameters)