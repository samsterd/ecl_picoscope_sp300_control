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
import biologic as bio
from matplotlib import pyplot as plt
import data
import experiments

# Experiment parameters. Will be added anc sorted as they become apparent
experimentParameters = {
    # Oscilloscope params
    'vtFunc' : pico.testVT, # Callable, function that can input an array of time values and output voltages at those times
    'vtFuncArgs' : (), # Tuple, additional positional arguments that will be passed to vtFunc when it is evaluated
    'vtFuncKwargs' : {'freq' : 1, 'amp' : 0.001, 'offset' : 0}, # Dict, additional keyword arguments that will be passed to vtFunc when it is evaluated
    'vtPeriod' : 1, # Duration in seconds of the voltage function. That max AWG frequency is 20 MHz, so minimum tStop is 5e-8 seconds
    'tStep' : 0.001, # Time step that the vtFunc will be sampled at. Minimum value is also 5e-8 seconds todo: check this
    #todo: add vtPeriods to specify how many times the AWG should repeat, with -1 repeating until experimentTime
    'experimentTime' : 5, # Duration that the vtFunc will be applied and photodetector / potentiostat measurements are done
    'scopeSamples' : 50000, # Number of voltage points that will be collected by the oscilloscope during experimentTime
                           # NOTE: Instrument limit is 16ns/sample when running on 4 channels. That limit may be higher with memory constraints
                            # and communication bottlenecks from streaming mode (48 kS memory -> need to send data every ~10kS).
                            # NOTE: the combination of time and samples will request a specific sampling interval. The actual
                            # sampling interval will be determined by communication speeds in streaming mode. The actual value will be saved
                            # TODO: determine empirical limits and add them here
    'detectorVoltageRange0' : 2, # maximum voltage expected on photodetector 0 (Channel A)
    'detectorVoltageRange1' : 2, # maximum voltage expected on photodetector 1 (Channel B)
    'potentiostatVoltageRange' : 1, # maximum voltage expected on the potentiostat output (Channel C)
                                    # Allowed values are 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20
                                    # Improper inputs will be rounded up to the nearest allowed value
    #todo: add voltage offsets? Might be useful for the photodetectors

    # potentiostat parameters
    'experimentType' : 'ca', # str - 'ocv' or 'ca' for open circuit voltage or chronoamperometry
    'potentiostatChannel' : 1, # int - 1 or 2, channel on used on potentiostat
    'currentRange' : 9, # int 4-9, determines the current range in powers of 10 from 0 (100 pA), 1 (1 nA),... 7 (1 mA), 8 (10 mA)
                        # WARNING: auto range is not available. Selecting too low of a range WILL damage the potentiostat
    'dt': 0.1,  # float, time interval for recording current. Note that the picoscope will record much faster than this (if used)
    'trigger' : True, # boolean, should a trigger be sent out when the trigger is run
    'extInput' : True, # boolean, enable external voltage (i.e. picoscope AWG)

    # OCV params
    'duration' : 1.0, # float, duration in seconds

    # CA params
    'vStep' : [0.5], # list: voltage at each step
    'vStepTime' : [5], # list: time spent at each step. len(vStep) must equal len(vStepTime)
    'startStep' : 0, # step index to start on. Must be less than len(vStep) - 1
    'numberOfCycles' : 0, # number of times CA is repeated

    # saving parameters
    'save' : False,    # should the data be saved (as an sqlite3 database)
    'experimentFolder' : 'C://Users//shams//Documents//ecl temp//20251120 ir test//', # folder to save in. Must end in //
    'experimentName' : 'ir 1mM tpra 10mM sine-sq-rand runs 60s rest', # base name of the file. Proper extensions will be appended by the experiment code

    # plotting parameters
    'plot' : True
}

def runExperiment(params : dict):
    '''
    Main control loop for running the ECL experiment

    Args:
        params (dict) : the input parameters to the experiment

    Returns:
        0 : data is saved, nothing returned?
    '''
    # initialize database
    if params['save']:
        db = data.Database(params)
        db.writeParameters(params, 0)

    # make connections to instruments
    # note that potentiostat outputs a voltage spike to the I_monitor when turning on, so this should be
    # done first to avoid accidental triggering
    pot = bio.Biologic(params)
    scope = pico.Picoscope()

    hwConf = pot.api.GetHardConf(pot.id, pot.channel)


    # get experiments ready to run
    scope.loadExperiment(params)
    pot.loadExperiment(params)
    scope.initStream()


    # run experiments. Note the stream on the picoscope has already started so the potentiostat must start quickly
    #   to avoid filling the scope memory
    pot.runExperimentWithoutData()
    a, b, c, d, t = scope.runStream()

    # plt.plot(scope.awgTime, scope.awgBuffer)
    # plt.show()
    # plt.plot(scope.awgTime, scope.awg)
    # plt.show()
    # this should be implemented as a function
    if params['save']:
        dataDict = {
            'detector0' : a,
            'detector1' : b,
            'potentiostatOut' : c,
            'potentiostatTrigger' : d,
            'time' : t,
            'awg' : scope.awg,
            'awgTime' : scope.awgTime
        }
        db.writeData(dataDict, False, 'experimentNumber', 0)
        db.close()
    #
    current = scope.voltageToPotentiostatCurrent(c, pot.currentRange)
    fig, ax = plt.subplots(2, 3)
    ax[0, 0].plot(scope.waveformBuffer)
    ax[0, 1].plot(t, a)
    ax[0, 2].plot(t, b)
    ax[1, 0].plot(t, c)
    ax[1, 1].plot(t, d)
    ax[1, 2].plot(t, current)
    plt.show()

    pot.close()
    scope.closePicoscope()
    return 0

# runExperiment(experimentParameters)
# experiments.threadedExperiment(experimentParameters)
# experiments.impedanceWithAutorange(experimentParameters, 4.5, 5.5, 5)
# experiments.impedanceTest(experimentParameters, 5, 6, 2)
# experiments.testPotentiostat(experimentParameters)
# if __name__ == '__main__':
#     # experiments.multiProcessExperimentsMain([experimentParameters])
#     experiments.multiProcessImpedanceExperiment(experimentParameters, 0, 5, 61)

#todo: add make range, awg time, etc reasonable for the experiment. lots of artifacting in awg when this isn't adjusted

if __name__ == '__main__':
    # initial test
    res = exp.runMultiParamList(experimentParameters,
                                    {"vtFunc" : [exp.sinWave, exp.squareWave, exp.squareWave, exp.randomStepProfile,exp.randomStepProfile],
                                        "vtFuncKwargs" : [
                                            {'freq': 1, 'amp': 1, 'offset': 0},
                                            {'freq': 1, 'amp': 1, 'offset': 0, 'duty' : 0.5},
                                            {'freq': 1, 'amp': -1, 'offset': 0, 'duty' : 0.3},
                                            {'numSteps': 8, 'voltageMin' : 0, 'voltageMax': 1},
                                            {'numSteps': 8, 'voltageMin': 0, 'voltageMax': 1}]}, 10)
    # long experiment
    #   sweep fine frequencies with sine, sweep frequencies and duty cycles with square, try lots of rands, repeat a few sines at end
    # 55 runs, 65 seconds each: ~1 hour
    # todo: generate this more efficiently!
    # res = exp.runMultiParamList(experimentParameters,
    #         {"vtFunc" : [exp.sinWave, exp.sinWave, exp.sinWave, exp.sinWave, exp.sinWave,
    #                      exp.sinWave, exp.sinWave, exp.sinWave, exp.sinWave, exp.sinWave,
    #                      exp.sinWave, exp.sinWave, exp.sinWave, exp.sinWave, exp.sinWave,
    #                      exp.sinWave, exp.sinWave, exp.sinWave, exp.sinWave, exp.sinWave,
    #                      exp.squareWave, exp.squareWave, exp.squareWave, exp.squareWave, exp.squareWave,
    #                      exp.squareWave, exp.squareWave, exp.squareWave, exp.squareWave, exp.squareWave,
    #                      exp.squareWave, exp.squareWave, exp.squareWave, exp.squareWave, exp.squareWave,
    #                      exp.squareWave, exp.squareWave, exp.squareWave, exp.squareWave, exp.squareWave,
    #                      exp.randomStepProfile, exp.randomStepProfile, exp.randomStepProfile, exp.randomStepProfile, exp.randomStepProfile,
    #                      exp.randomStepProfile, exp.randomStepProfile, exp.randomStepProfile, exp.randomStepProfile, exp.randomStepProfile,
    #                      exp.sinWave, exp.sinWave, exp.sinWave, exp.sinWave, exp.sinWave],
    #             "vtFuncKwargs" : [
    #             {'freq': 0.5, 'amp': 1, 'offset': 0},{'freq': 0.8, 'amp': 1, 'offset': 0},
    #             {'freq': 1, 'amp': 1, 'offset': 0},{'freq': 2, 'amp': 1, 'offset': 0},
    #             {'freq': 3, 'amp': 1, 'offset': 0}, {'freq': 4, 'amp': 1, 'offset': 0},
    #             {'freq': 5, 'amp': 1, 'offset': 0},{'freq': 8, 'amp': 1, 'offset': 0},
    #             {'freq': 10, 'amp': 1, 'offset': 0},{'freq': 20, 'amp': 1, 'offset': 0},
    #             {'freq': 50, 'amp': 1, 'offset': 0},{'freq': 80, 'amp': 1, 'offset': 0},
    #             {'freq': 100, 'amp': 1, 'offset': 0},{'freq': 200, 'amp': 1, 'offset': 0},
    #             {'freq': 500, 'amp': 1, 'offset': 0},{'freq': 800, 'amp': 1, 'offset': 0},
    #             {'freq': 1000, 'amp': 1, 'offset': 0},{'freq': 2000, 'amp': 1, 'offset': 0},
    #             {'freq': 1, 'amp': 1, 'offset': 0}, {'freq': 1, 'amp': 1, 'offset': 0},
    #             {'freq': 0.5, 'amp': -1, 'offset': 0, 'duty': 0.5},{'freq': 1, 'amp': -1, 'offset': 0, 'duty': 0.5},
    #             {'freq': 5, 'amp': -1, 'offset': 0, 'duty': 0.5},{'freq': 10, 'amp': -1, 'offset': 0, 'duty': 0.5},
    #             {'freq': 50, 'amp': -1, 'offset': 0, 'duty': 0.5},{'freq': 100, 'amp': -1, 'offset': 0, 'duty': 0.5},
    #             {'freq': 500, 'amp': -1, 'offset': 0, 'duty': 0.5},{'freq': 1000, 'amp': -1, 'offset': 0, 'duty': 0.5},
    #             {'freq': 1, 'amp': -1, 'offset': 0, 'duty': 0.5},{'freq': 1, 'amp': -1, 'offset': 0, 'duty': 0.5},
    #             {'freq': 1, 'amp': -1, 'offset': 0, 'duty': 0.1},{'freq': 1, 'amp': -1, 'offset': 0, 'duty': 0.9},
    #             {'freq': 10, 'amp': -1, 'offset': 0, 'duty': 0.1},{'freq': 10, 'amp': -1, 'offset': 0, 'duty': 0.9},
    #             {'freq': 100, 'amp': -1, 'offset': 0, 'duty': 0.1},{'freq': 100, 'amp': -1, 'offset': 0, 'duty': 0.9},
    #             {'freq': 1000, 'amp': -1, 'offset': 0, 'duty': 0.1},{'freq': 1000, 'amp': -1, 'offset': 0, 'duty': 0.9},
    #             {'freq': 1, 'amp': -1, 'offset': 0, 'duty': 0.3},{'freq': 1, 'amp': -1, 'offset': 0, 'duty': 0.7},
    #             {'numSteps': 10, 'voltageMin': 0, 'voltageMax': 1},{'numSteps': 10, 'voltageMin': 0, 'voltageMax': 1},
    #             {'numSteps': 10, 'voltageMin': 0, 'voltageMax': 1},{'numSteps': 10, 'voltageMin': 0, 'voltageMax': 1},
    #             {'numSteps': 10, 'voltageMin': 0, 'voltageMax': 1},{'numSteps': 10, 'voltageMin': 0, 'voltageMax': 1},
    #             {'numSteps': 10, 'voltageMin': 0, 'voltageMax': 1},{'numSteps': 10, 'voltageMin': 0, 'voltageMax': 1},
    #             {'numSteps': 10, 'voltageMin': 0, 'voltageMax': 1},{'numSteps': 10, 'voltageMin': 0, 'voltageMax': 1},
    #             {'freq': 1, 'amp': 1, 'offset': 0}, {'freq': 1, 'amp': 1, 'offset': 0},
    #             {'freq': 1, 'amp': 1, 'offset': 0}, {'freq': 1, 'amp': 1, 'offset': 0},{'freq': 1, 'amp': 1, 'offset': 0}],
    #         "vtPeriod": [2, 1.25, 1, 0.5, 1/3, 1/4, 1/5, 1/8, 1/10, 1/20, 1/50, 1/80, 1/100, 1/200, 1/500, 1/800, 1/1000, 1/2000, 1, 1,
    #                      2, 1, 0.2, 0.1, 0.02, 0.01, 0.002, 0.001, 1, 1, 1, 1, 0.1, 0.1, 0.01, 0.01, 0.001, 0.001, 1, 1,
    #                      5, 5, 0.5, 0.5, 0.05, 0.05, 0.005, 0.005, 5, 5, 1, 1, 1, 1, 1],
    #         "tStep" : [2/1000, 1.25/1000, 1/1000, 0.5/1000, 1/3000, 1/4000, 1/5000, 1/8000, 1/10000, 1/20000, 1/50000,
    #                     1/80000, 1/100000, 1/200000, 1/500000, 1/800000, 1/1000000, 1/2000000, 1/1000, 1/1000,
    #                     2/1000, 1/1000, 0.2/1000, 0.1/1000, 0.02/1000, 0.01/1000, 0.002/1000, 0.001/1000,
    #                     1/1000, 1/1000, 1/1000, 1/1000, 0.1/1000, 0.1/1000, 0.01/1000, 0.01/1000, 0.001/1000, 0.001/1000, 1/1000, 1/1000,
    #                     5/1000, 5/1000, 0.5/1000, 0.5/1000, 0.05/1000, 0.05/1000, 0.005/1000, 0.005/1000, 5/1000, 5/1000,
    #                     1/1000, 1/1000, 1/1000, 1/1000, 1/1000]
    #          }, 60)
    # res = experiments.runMultiParamList(experimentParameters,
    #                                     {"vtFuncKwargs": [{'startV': 0, 'endV': 0.5}, {'startV': 0, 'endV': 1},
    #                                                       {'startV' : 0, 'endV': -0.5}, {'startV':0, 'endV':-1}]})

# fresh todo:
#   limit to 250 kS/s on picoscope
#   add offset to AWG
#   separate AWG start from everything else-
#       implement software trigger for non-instantaneous start
#   create more generalized experiment framework
#   write basic experiments:
#       pause/stabilize then wave
#   start looking into DMD inputs / outputs


#todo: get pre-streaming sampling to work
# set up an impedance like experiment
# implement data saving!
# write a batch testing script, prepare for the real experiment!