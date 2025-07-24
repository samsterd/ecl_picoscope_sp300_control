import data
import picoscope as pico
import biologic as bio
import copy
import numpy as np
import time
from matplotlib import pyplot as plt

# experiment functions will live here. Eventually this will become more systematic

def impedanceTest(params : dict, freqStart : float, freqStop : float, numberOfFreqs: int):
    '''
    Applies a low amplitude sine wave at different frequencies for testing purposes

    Args:
        params (dict) : the input parameters to the experiment
        freqStart : starting frequency (Hz)
        freqStop : stopping frequency (Hz)
        numberOfFreqs : number of frequencies in between start and stop (will be logspaced)

    Returns:
        0 : data is saved, nothing returned
    '''
    #todo: this works as a test, but modifying the parameter dict that is submitted on another script isn't super robust
    # generate frequency list
    freqs = np.logspace(freqStart, freqStop, numberOfFreqs)

    # overwrite initial parameters list to make this an impedance experiment
    runningParams = copy.copy(params)
    runningParams['vtFunc'] = pico.testVT
    runningParams['vtFuncArgs'] = ()
    runningParams['scopeSamples'] = 10000
    runningParams['vStep'] = [0]

    # initialize database
    if params['save']:
        db = data.Database(runningParams)


    # make connections to instruments
    # note that potentiostat outputs a voltage spike to the I_monitor when turning on, so this should be
    # done first to avoid accidental triggering
    # todo: need to break up initialization so that we don't reconnect each iteration
    #   need to separate opening instrument and loading parameters
    pot = bio.Biologic(runningParams)
    scope = pico.Picoscope()

    # iterate through frequencies, updating parameters as needed
    for i in range(len(freqs)):

        f = freqs[i]
        runningParams['vtFuncKwargs'] = {'freq' : f, 'amp' : 0.01}
        runningParams['vtPeriod'] = 1 / f
        runningParams['tStep'] = max(5e-8, runningParams['vtPeriod'] / 1000)
        runningParams['experimentTime'] = 10 * runningParams['vtPeriod'] + 0.035 # extra 35 ms to account for startup time
        runningParams['vStepTime'] = [10 * runningParams['vtPeriod']]

        if params['save']:
            db.writeParameters(runningParams, i)

        # get experiments ready to run
        scope.loadExperiment(runningParams)
        pot.loadExperiment()
        scope.initStream()

        # run experiments. Note the stream on the picoscope has already started so the potentiostat must start quickly
        #   to avoid filling the scope memory
        pot.runExperimentWithoutData()
        a, b, c, d, t = scope.runStream()
        current = scope.voltageToPotentiostatCurrent(c, pot.currentRange)

        # this should be implemented as a function
        if params['save']:
            dataDict = {
                'detector0' : a,
                'detector1' : b,
                'potentiostatOut' : c,
                'potentiostatTrigger' : d,
                'potentiostatCurrent' : current,
                'time' : t,
                'awg' : scope.awg,
                'awgTime' : scope.awgTime
            }
            db.writeData(dataDict, False, 'experimentNumber', i)

        #
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
    if params['save']:
        db.close()