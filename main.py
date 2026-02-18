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
import experiments as exp

# Experiment parameters. Will be added anc sorted as they become apparent
experimentParameters = {
    # Oscilloscope params
    'vtFunc' : pico.testVT, # Callable, function that can input an array of time values and output voltages at those times
    'vtFuncArgs' : (), # Tuple, additional positional arguments that will be passed to vtFunc when it is evaluated
    'vtFuncKwargs' : {'freq' : 1, 'amp' : 0.001, 'offset' : 0}, # Dict, additional keyword arguments that will be passed to vtFunc when it is evaluated
    'vtPeriod' : 1, # Duration in seconds of the voltage function. That max AWG frequency is 20 MHz, so minimum tStop is 5e-8 seconds
    'awgDelay' : 0.1, # Duration in seconds to wait after the trigger is received before starting the AWG. NOT TESTED YET
    'tStep' : 0.001, # Time step that the vtFunc will be sampled at. Minimum value is also 5e-8 seconds todo: check this
    #todo: add vtPeriods to specify how many times the AWG should repeat, with -1 repeating until experimentTime
    'experimentTime' : 5, # Duration that the vtFunc will be applied and photodetector / potentiostat measurements are done
    'scopeSamples' : 50000, # Number of voltage points that will be collected by the oscilloscope during experimentTime
                           # NOTE: Instrument limit is 16ns/sample when running on 4 channels. That limit may be higher with memory constraints
                            # and communication bottlenecks from streaming mode (48 kS memory -> need to send data every ~10kS).
                            # NOTE: the combination of time and samples will request a specific sampling interval. The actual
                            # sampling interval will be determined by communication speeds in streaming mode. The actual value will be saved
                            # NOTE: streaming maximum data transfer rate is 1MS/s, or 250kS per channel, leading to a minimum
                            #   timestep of 4 us. Empirically, the software struggles below 25 us
    'detectorVoltageRange0' : 2, # maximum voltage expected on photodetector 0 (Channel A)
    'detectorVoltageRange1' : 2, # maximum voltage expected on photodetector 1 (Channel B)
    'potentiostatVoltageRange' : 1, # maximum voltage expected on the potentiostat output (Channel C)
                                    # Allowed values are 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20
                                    # Improper inputs will be rounded up to the nearest allowed value
    #todo: add voltage offsets? Might be useful for the photodetectors

    # potentiostat parameters
    'experimentType' : 'ca', # str - 'ocv' or 'ca' for open circuit voltage or chronoamperometry
    'potentiostatChannel' : 1, # int - 1 or 2, channel on used on potentiostat
    'currentRange' : 8, # int 4-9, determines the current range in powers of 10 from 0 (100 pA), 1 (1 nA),... 7 (1 mA), 8 (10 mA)
                        # WARNING: auto range is not available. Selecting too low of a range WILL damage the potentiostat
    'dt': 0.1,  # float, time interval for recording current. Note that the picoscope will record much faster than this (if used)
    'trigger' : True, # boolean, should a trigger be sent out when the trigger is run
    'extInput' : True, # boolean, enable external voltage (i.e. picoscope AWG)

    # OCV params
    'duration' : 1.0, # float, duration in seconds

    # CA params
    'vStep' : [1], # list: voltage at each step
    'vStepTime' : [5], # list: time spent at each step. len(vStep) must equal len(vStepTime)
    'startStep' : 0, # step index to start on. Must be less than len(vStep) - 1
    'numberOfCycles' : 0, # number of times CA is repeated

    # saving parameters
    'save' : True,    # should the data be saved (as an sqlite3 database)
    'experimentFolder' : 'C://Users//shams//Documents//ecl temp//20251124 ru and ir test//', # folder to save in. Must end in //
    'experimentName' : 'ru 1mM ir 1 mM tpra 20mM 10 gain 550longpass filter chB sine-sq-rand runs 60s rest', # base name of the file. Proper extensions will be appended by the experiment code

    # plotting parameters
    'plot' : False
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
    res = exp.runMultiParamList(experimentParameters,
            {"vtFunc" : [exp.sinWave, exp.sinWave, exp.sinWave, exp.sinWave, exp.sinWave,
                         exp.sinWave, exp.sinWave, exp.sinWave, exp.sinWave, exp.sinWave,
                         exp.sinWave, exp.sinWave, exp.sinWave, exp.sinWave, exp.sinWave,
                         exp.sinWave, exp.sinWave, exp.sinWave, exp.sinWave, exp.sinWave,
                         exp.squareWave, exp.squareWave, exp.squareWave, exp.squareWave, exp.squareWave,
                         exp.squareWave, exp.squareWave, exp.squareWave, exp.squareWave, exp.squareWave,
                         exp.squareWave, exp.squareWave, exp.squareWave, exp.squareWave, exp.squareWave,
                         exp.squareWave, exp.squareWave, exp.squareWave, exp.squareWave, exp.squareWave,
                         exp.randomStepProfile, exp.randomStepProfile, exp.randomStepProfile, exp.randomStepProfile, exp.randomStepProfile,
                         exp.randomStepProfile, exp.randomStepProfile, exp.randomStepProfile, exp.randomStepProfile, exp.randomStepProfile,
                         exp.sinWave, exp.sinWave, exp.sinWave, exp.sinWave, exp.sinWave],
                "vtFuncKwargs" : [
                {'freq': 0.5, 'amp': 1, 'offset': 0},{'freq': 0.8, 'amp': 1, 'offset': 0},
                {'freq': 1, 'amp': 1, 'offset': 0},{'freq': 2, 'amp': 1, 'offset': 0},
                {'freq': 3, 'amp': 1, 'offset': 0}, {'freq': 4, 'amp': 1, 'offset': 0},
                {'freq': 5, 'amp': 1, 'offset': 0},{'freq': 8, 'amp': 1, 'offset': 0},
                {'freq': 10, 'amp': 1, 'offset': 0},{'freq': 20, 'amp': 1, 'offset': 0},
                {'freq': 50, 'amp': 1, 'offset': 0},{'freq': 80, 'amp': 1, 'offset': 0},
                {'freq': 100, 'amp': 1, 'offset': 0},{'freq': 200, 'amp': 1, 'offset': 0},
                {'freq': 500, 'amp': 1, 'offset': 0},{'freq': 800, 'amp': 1, 'offset': 0},
                {'freq': 1000, 'amp': 1, 'offset': 0},{'freq': 2000, 'amp': 1, 'offset': 0},
                {'freq': 1, 'amp': 1, 'offset': 0}, {'freq': 1, 'amp': 1, 'offset': 0},
                {'freq': 0.5, 'amp': -1, 'offset': 0, 'duty': 0.5},{'freq': 1, 'amp': -1, 'offset': 0, 'duty': 0.5},
                {'freq': 5, 'amp': -1, 'offset': 0, 'duty': 0.5},{'freq': 10, 'amp': -1, 'offset': 0, 'duty': 0.5},
                {'freq': 50, 'amp': -1, 'offset': 0, 'duty': 0.5},{'freq': 100, 'amp': -1, 'offset': 0, 'duty': 0.5},
                {'freq': 500, 'amp': -1, 'offset': 0, 'duty': 0.5},{'freq': 1000, 'amp': -1, 'offset': 0, 'duty': 0.5},
                {'freq': 1, 'amp': -1, 'offset': 0, 'duty': 0.5},{'freq': 1, 'amp': -1, 'offset': 0, 'duty': 0.5},
                {'freq': 1, 'amp': -1, 'offset': 0, 'duty': 0.1},{'freq': 1, 'amp': -1, 'offset': 0, 'duty': 0.9},
                {'freq': 10, 'amp': -1, 'offset': 0, 'duty': 0.1},{'freq': 10, 'amp': -1, 'offset': 0, 'duty': 0.9},
                {'freq': 100, 'amp': -1, 'offset': 0, 'duty': 0.1},{'freq': 100, 'amp': -1, 'offset': 0, 'duty': 0.9},
                {'freq': 1000, 'amp': -1, 'offset': 0, 'duty': 0.1},{'freq': 1000, 'amp': -1, 'offset': 0, 'duty': 0.9},
                {'freq': 1, 'amp': -1, 'offset': 0, 'duty': 0.3},{'freq': 1, 'amp': -1, 'offset': 0, 'duty': 0.7},
                {'numSteps': 10, 'voltageMin': -1, 'voltageMax': 1},{'numSteps': 10, 'voltageMin': -1, 'voltageMax': 1},
                {'numSteps': 10, 'voltageMin': -1, 'voltageMax': 1},{'numSteps': 10, 'voltageMin': -1, 'voltageMax': 1},
                {'numSteps': 10, 'voltageMin': -1, 'voltageMax': 1},{'numSteps': 10, 'voltageMin': -1, 'voltageMax': 1},
                {'numSteps': 10, 'voltageMin': -1, 'voltageMax': 1},{'numSteps': 10, 'voltageMin': -1, 'voltageMax': 1},
                {'numSteps': 10, 'voltageMin': -1, 'voltageMax': 1},{'numSteps': 10, 'voltageMin': -1, 'voltageMax': 1},
                {'freq': 1, 'amp': 1, 'offset': 0}, {'freq': 1, 'amp': 1, 'offset': 0},
                {'freq': 1, 'amp': 1, 'offset': 0}, {'freq': 1, 'amp': 1, 'offset': 0},{'freq': 1, 'amp': 1, 'offset': 0}],
            "vtPeriod": [2, 1.25, 1, 0.5, 1/3, 1/4, 1/5, 1/8, 1/10, 1/20, 1/50, 1/80, 1/100, 1/200, 1/500, 1/800, 1/1000, 1/2000, 1, 1,
                         2, 1, 0.2, 0.1, 0.02, 0.01, 0.002, 0.001, 1, 1, 1, 1, 0.1, 0.1, 0.01, 0.01, 0.001, 0.001, 1, 1,
                         5, 5, 0.5, 0.5, 0.05, 0.05, 0.005, 0.005, 5, 5, 1, 1, 1, 1, 1],
            "tStep" : [2/1000, 1.25/1000, 1/1000, 0.5/1000, 1/3000, 1/4000, 1/5000, 1/8000, 1/10000, 1/20000, 1/50000,
                        1/80000, 1/100000, 1/200000, 1/500000, 1/800000, 1/1000000, 1/2000000, 1/1000, 1/1000,
                        2/1000, 1/1000, 0.2/1000, 0.1/1000, 0.02/1000, 0.01/1000, 0.002/1000, 0.001/1000,
                        1/1000, 1/1000, 1/1000, 1/1000, 0.1/1000, 0.1/1000, 0.01/1000, 0.01/1000, 0.001/1000, 0.001/1000, 1/1000, 1/1000,
                        5/1000, 5/1000, 0.5/1000, 0.5/1000, 0.05/1000, 0.05/1000, 0.005/1000, 0.005/1000, 5/1000, 5/1000,
                        1/1000, 1/1000, 1/1000, 1/1000, 1/1000]
             }, 60)
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


#   write a multiprocess impedance, check at a wide frequency range
#       make sure to have a high floor on exp time to ensure trigger goes
#   write a more generalized multiprocess, multiparameter framework
#   plan out first experiments

# solved issue. fix is sad: streaming is limited by USB sampling rate, not scope time interval
#   need to limit samples to 2500 (10k buffer max / 4) and sampling rate below 250kS/s (1 MS/s USB max / 4)


#todo:
#   sipm data analysis is more complicated
#       need a spike counter. simple binned sum might work
#       could also rig something with counting triggers? this would be hacky but FAST
#       hardware: is there a simple circuit we can buy to do this for cheap?
#   correction: sipm might be fine, could be issue with fibers

#   check that threading fixed high freq problem
#       nope, freezes at 10 khz
#       next thing is to try multiprocessing
#           multiprocessing will be MUCH more involved - can't pass class methods through
#       multiprocessing did not fix everything, but seems to work a bit more consistently at high f
#           increasing the experiment time minimum to 1 ms also seems to help a lot:
#           I'm guessing there's an issue with the trigger consistency
#   fix low f problems:
#       overlapping AWG output
#           there is current response before trigger ?
#           562 hz is an example - both pre-trigger and overlapped signals
#           reverting awg to run for finite number of shots fixed this partially
#           may also be an issue with slow threading?
#       recording pre-trigger samples
#           check callback conditions, or whether its hitting max samples early due to slow triggering?
#       happen on single shot experiments - it isn't remnants from autoranging
#       happens with or without threading
#   check performance on long acquisitions

# todo: rerun impedance without current spike
#       hitting problems when f > 10kHz:
#           sample interval creating issues in runStream() - 10 us is too short?
#           stream hangs / stops workings on the waiting for callback. no data is collected?
#               could be experiment time is too short and it misses the trigger?
#       trigger is working on the potentiostat side
#       Problem as I understand it now: since streaming memory is FIFO and starts recording immediately,
#           if getlatestvalues is called too long after the experiment start, the trigger point is overwritten
#           as a result of how the callback function is handling it, no data gets saved, nextSample does not increase, and
#           the while loop goes infinitely.
#           Delay between calling startChannel() and running stream is not measureable by time.time...
#               calling startChannel takes 33 ms
#           What about just increasing the trigger length?
#               that solves the recording but the AWG will start earlier
#           Possible fixes: async or multithreading to start potentiostat and scope at closer times
#               b/c we are no longer recording pre-trigger data, runStream() could be async?
#               this is the harder fix. may not solve everything but would learn more
#           Is there a possible fix in the logic of the callback function?
#               is there a wasTriggered function in the pico sdk?
#               hacky fix would be to use a simpler callback function at shorter timescales that just runs right away?
#                   hacky fix is too slow - AWG already finished by the time startChannel finishes
#               gotta go async
#                   Best solution is to thread initStream and runExperimentWithoutSaving
#                       initStream is the threaded function, join at end
#               other option: 'manually' start AWG on stream start
#       strange thing is this worked fine at the first test (when the current spike was happening)
#       hanging problem is inconsistent - at same parameters, it sometimes hangs, sometimes doesn't, even at 10khz
#           it isn't loading time - sometimes the first few work and then the next doesn't w/o reconnecting
#           a related problem is that the trigger only seems to go off for the first experiment in series
#           this is even if waiting for a stop status before running again
#       from looking at low freq w/ 100 ms trigger durration, getting approx delay:
#           1khz: trig ends at 1.77 ms, off for ~ 0.75 ms, then restarts
#           1.7 khz: only catch last ~10 us of trigger
#           3.1 khz: get start
#           5.6 khz: quiet ~100 us between triggers
#           10 khz: hangs
#           delay b/w trigger and recording seems to be long (99ms????) but trigger also seems to be repeating???
#       strategy: define async runStream and runPotentiostat functions. Start runStream, yield to runPot, then go back to runStream?
#       what is weird is that the measured delay between end of runPot and start of streaming callback is < measureable time.time
#           the issue is then that the trigger goes out early in the ~30 ms runtime of startChannel() -
#           threading (NOT ASYNC) might fix this by devoting background resources to callback WHILE start channel is running
#           other option

#todo: deal with issues that arose in impedance testing:

# moving to potentiostat ground fixed current spike, but now the data is MUCH noisier
#      there is a 0.0263 period (~38 Hz) background
#       no background when detached. No background when running constant 0.01V CA, no background when 0.01 V constant from AWG
#       background freq is a function of awg sampling time?
#       scope samples reveals the underlying issue, which is the tStep -
#           AWG voltage changes in discrete points slower than sampling, so sampling picks up on the instantaneous reaction
#           of the dummy cell RC to step functions, which makes it look like a set of bumps!
#           solution is lower tStep + higher sampling?
#           also test on just resistor if possible