import data
import picoscope as pico
import biologic as bio
import copy
import numpy as np
import time
from matplotlib import pyplot as plt
import scipy.signal
import math
import threading
from multiprocessing import Process, Queue, set_start_method, shared_memory

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
        print(f)
        runningParams['vtFuncKwargs'] = {'freq' : f, 'amp' : 0.01}
        runningParams['vtPeriod'] = 1 / f
        runningParams['tStep'] = max(5e-8, runningParams['vtPeriod'] / 1000)
        runningParams['experimentTime'] = 10 * runningParams['vtPeriod'] + 0.001 # extra 35 ms to account for startup time
        runningParams['vStepTime'] = [10 * runningParams['vtPeriod']]

        if params['save']:
            db.writeParameters(runningParams, i)

        # get experiments ready to run
        scope.loadExperiment(runningParams)
        pot.loadExperiment(runningParams)
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
        if params['plot']:
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

def impedanceWithAutorange(params : dict, freqStart : float, freqStop : float, numberOfFreqs: int):
    '''
    Applies a low amplitude sine wave at different frequencies for testing purposes. Repeats experiments when the max current
    is <0.1 V to maximize signal

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

        run = True

        while run and runningParams['currentRange'] >= 0:

            f = freqs[i]
            print(f)
            runningParams['vtFuncKwargs'] = {'freq' : f, 'amp' : 0.01}
            runningParams['vtPeriod'] = 1 / f
            runningParams['tStep'] = max(5e-8, runningParams['vtPeriod'] / 1000)
            runningParams['experimentTime'] = max(10.9 * runningParams['vtPeriod'], 0.0011) # add extra time. to preserve 10kS, min time is 16 ns * 10k = 160 us
            runningParams['vStepTime'] = [max(10 * runningParams['vtPeriod'], 0.001)] # minimum timebase for CA is 34 us
            runningParams['scopeSamples'] = 10000

            if params['save']:
                # this should overwrite if re-running
                db.writeParameters(runningParams, i)

            # make sure potentiostat is finished running
            while not pot.experimentDoneQ:
                time.sleep(0.1)

            # run experiment
            startTime = time.time()
            a, b, c, d, t = runThreaded(runningParams, pot, scope)
            # scope.loadExperiment(runningParams)
            # pot.loadExperiment(runningParams)
            # scope.initStream()
            #
            #
            # # run experiments. Note the stream on the picoscope has already started so the potentiostat must start quickly
            # #   to avoid filling the scope memory
            # pot.runExperimentWithoutData()
            # a, b, c, d, t = scope.runStream()
            current = scope.voltageToPotentiostatCurrent(c, pot.currentRange)

            # make sure the experiment time has elapsed before rerunning
            time.sleep(1)
            # currentTime = time.time()
            # while currentTime < startTime + runningParams['experimentTime']:
            #     time.sleep(0.1)
            #     currentTime = time.time()

            #  current is low and we could get better signal by reducing the range
            if np.max(abs(c)) < 100 and runningParams['currentRange'] > 4:

                runningParams['currentRange'] -= 1
                print('adjusting range')
                run = True

            else:

                # switch run to break the while loop
                run = False

                # todo: this should be implemented as a function
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

                # todo: this should be a function
                if params['plot']:
                    fig, ax = plt.subplots(2, 3)
                    ax[0, 0].plot(scope.waveformBuffer)
                    ax[0, 1].plot(t, a)
                    ax[0, 2].plot(t, b)
                    ax[1, 0].plot(t, c)
                    ax[1, 1].plot(t, d)
                    ax[1, 2].plot(t, current)
                    plt.show()

            # # check that the potentiostat is free before proceeding
            # print(pot.experimentStatus())
            # while not pot.experimentDoneQ:
            #     time.sleep(0.1)

    pot.close()
    scope.closePicoscope()
    if params['save']:
        db.close()

def multiprocessExperiment(params):
    '''
    Run a simple Chronoamperometry + AWG experiment with multiprocessing to enable streaming capture while potentiostat starts

    :param params:
    :return:
    '''
    # define names of shared memory
    shmNames = {'chA' : 'chA',
    'chB' : 'chB',
    'chC' : 'chC',
    'chD' : 'chD',
    'tName' : 't',
    'lenName' : 'len'}


    # set up db if saving
    if params['save']:
        db = data.Database(params)
        db.writeParameters(params, 0)

    # initialize instrument connections
    # note that potentiostat outputs a voltage spike to the I_monitor when turning on, so this should be
    # done first to avoid accidental triggering
    pot = bio.Biologic(params)


    # start multiprocess thread for running picoscope. This will run at the same time as biologic
    #todo: add args for process
    picoq = Queue() # create queue for getting data out
    picoProcess = Process(target = multiProcessPicoWrapper, args = [params, picoq], kwargs = shmNames)
    picoProcess.start()

    pot.loadExperiment(params)
    # using queue to wait until streaming starts to run
    startFlag = picoq.get(block = True)
    pot.runExperimentWithoutData()

    # wait for picoProcess to finish, check for data in shared memory
    # todo: this is causing problems. join makes parent process wait until child finishes, but if that happens,
    #   shared memory is getting garbage collected
    #   need to find a way to make child process wait until shared memory is accessed
    #   or just use Queue instead


    # gather data from shared memory
    # ashm = shared_memory.SharedMemory(name = shmNames['chA'])
    # bshm = shared_memory.SharedMemory(name = shmNames['chB'])
    # cshm = shared_memory.SharedMemory(name = shmNames['chC'])
    # dshm = shared_memory.SharedMemory(name = shmNames['chD'])
    # tshm = shared_memory.SharedMemory(name = shmNames['tName'])
    # len = shared_memory.SharedMemory(shmNames['lenName'])
    # a = np.frombuffer(ashm)
    # b = np.frombuffer(bshm)
    # current = np.frombuffer(cshm)
    # d = np.frombuffer(dshm, dtype = np.int_)
    # t = np.frombuffer(tshm)

    datReturn = picoq.get(block = True)
    print(len(datReturn))
    a = datReturn[0]
    b = datReturn[1]
    current = datReturn[2]
    d = datReturn[3]
    t = datReturn[4]

    picoProcess.join()


    print('here')
    # save data, if applicable
    # todo: add awgbuffer and time to shared memory if this process works
    if params['save']:
        dataDict = {
            'detector0' : a,
            'detector1' : b,
            'potentiostatOut' : current,
            'potentiostatTrigger' : d,
            'time' : t
            # 'awg' : scope.awg,
            # 'awgTime' : scope.awgTime
        }
        db.writeData(dataDict, False, 'experimentNumber', 0)
        db.close()

    # plot data, if applicable
    if params['plot']:
        fig, ax = plt.subplots(2, 3)
        # ax[0, 0].plot(scope.waveformBuffer)
        ax[0, 1].plot(t, a)
        ax[0, 2].plot(t, b)
        # ax[1, 0].plot(t, c)
        ax[1, 1].plot(t, d)
        ax[1, 2].plot(t, current)
        plt.show()

    # close and unlink shared memory
    # ashm.close()
    # bshm.close()
    # cshm.close()
    # dshm.close()
    # tshm.close()
    # ashm.unlink()
    # bshm.unlink()
    # cshm.unlink()
    # dshm.unlink()
    # tshm.unlink()

    # close the process
    picoProcess.close()

    pot.close()

def multiProcessPicoWrapper(params, queue, chA = 'chA', chB = 'chB', chC = 'chC', chD = 'chD', tName = 't', lenName = 'len'):
    '''
    Wrapper function that creates a Picoscope object, runs the specified experiment, saves results in a shared memory block,
    and closes the picoscope. This should be run AFTER connecting to the Biologic to avoid the current spike on startup
    :param params:
    :return:
    '''
    # todo: architecture for running multiple experiments
    #   connect
    #   execution while loop:
    #       look into queue for orders
    #       if 'stop': close everything
    #       if a dict: load. wait for biologic loaded flag, run with those parameters send 'starting' flag, send results into queue
    #   on other side:
    #       connect to biologic
    #       loop experiments:
    #           adjust params as needed, put them in queue
    #           load experiment, send flag and wait for streaming flag, run experiment
    #           get data from queue
    #   might be useful to learn how locks work - they might be better than sending flags through the queue
    scope = pico.Picoscope()
    scope.loadExperiment(params)

    # create shared memory for each of the return buffers
    # this is done after loadExperiments since the number of samples can change based on pico.resolveSampleInterval()
    # this is also used to send scopeSamples up to the parent process. There is probably a more efficient way to do this
    #   but I don't have time to learn it
    # exArray = np.zeros(scope.scopeSamples)
    # exIntArray = np.zeros(scope.scopeSamples, dtype = np.int_)
    # lenArray = np.array[scope.scopeSamples] # this is a silly way to pass a single int in shared memory, but here we are
    # ashm = shared_memory.SharedMemory(name = chA, create = True, size = exArray.nbytes)
    # bshm = shared_memory.SharedMemory(name = chB, create = True, size = exArray.nbytes)
    # cshm = shared_memory.SharedMemory(name = chC, create = True, size = exArray.nbytes)
    # dshm = shared_memory.SharedMemory(name = chD, create = True, size = exIntArray.nbytes)
    # tshm = shared_memory.SharedMemory(name = tName, create = True, size = exArray.nbytes)
    # lenshm = shared_memory.SharedMemory(name = lenName, create = True, size = lenArray.nbytes)
    # aBuffer = np.ndarray(exArray.shape, dtype = exArray.dtype, buffer = ashm.buf)
    # bBuffer = np.ndarray(exArray.shape, dtype = exArray.dtype, buffer = bshm.buf)
    # cBuffer = np.ndarray(exArray.shape, dtype = exArray.dtype, buffer = cshm.buf)
    # dBuffer = np.ndarray(exIntArray.shape, dtype = exIntArray.dtype, buffer = dshm.buf)
    # tBuffer = np.ndarray(exArray.shape, dtype = exArray.dtype, buffer = tshm.buf)
    # lenBuffer = np.ndarray(lenArray.shape, dtype = lenArray.dtype, buffer = lenshm.buf)

    # pass the len into shared memory
    # lenBuffer[:] = lenArray[:]

    scope.initStream()
    # send streaming start flag
    queue.put('start')
    a, b, c, d, t = scope.runStream()
    current = scope.voltageToPotentiostatCurrent(c, params['currentRange'])

    queue.put([a, b, current, d, t])


    # copy data into shared memory
    # aBuffer[:] = a[:]
    # bBuffer[:] = b[:]
    # cBuffer[:] = current[:]
    # dBuffer[:] = d[:]
    # tBuffer[:] = t[:]

    # close connection
    scope.closePicoscope()

    # close shared memory - unsure if this causes a problem if it wasn't linked to the parent process yet
    # ashm.close()
    # bshm.close()
    # cshm.close()
    # dshm.close()
    # tshm.close()
    #lenshm.close()

def threadedExperiment(params):
    '''
    Run a simple Chronoamperometry + AWG experiment with Threading to enable streaming capture while potentiostat starts

    :param params:
    :return:
    '''

    # set up db if saving
    if params['save']:
        db = data.Database(params)
        db.writeParameters(params, 0)

    # initialize instrument connections
    # note that potentiostat outputs a voltage spike to the I_monitor when turning on, so this should be
    # done first to avoid accidental triggering
    pot = bio.Biologic(params)
    time.sleep(1)
    scope = pico.Picoscope()

    # get experiments ready to run
    scope.loadExperiment(params)
    pot.loadExperiment(params)

    # initialize the picoscope thread before starting the stream to minimize delay
    #   we are threading the runStream and runExperiment functions to allow the callback function to run while the potentiostat
    #   experiment starts. This avoids issues at higher speeds where the first ~30 ms of data are missed
    #   due to the limited memory of the Picoscope and the longer run time of pot.runExperiment()
    picoThread = threading.Thread(target = scope.runStream, name = "Pico", daemon = False)
    # potThread = threading.Thread(target = pot.runExperimentWithoutData, name = "Pot", daemon = False)
    scope.initStream()

    # start streaming thread and then start the potentiostat
    picoThread.start()
    time.sleep(0.001)
    pot.runExperimentWithoutData()

    # wait for picoThread to finish
    picoThread.join()

    # gather data
    a, b, c, d, t = scope.channelAData, scope.channelBData, scope.channelCData, scope.channelDData, scope.time
    current = scope.voltageToPotentiostatCurrent(c, pot.currentRange)

    # save data, if applicable
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

    # plot data, if applicable
    if params['plot']:
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

def runThreaded(params, potObj, scopeObj):
    '''
    Threaded call to run an experiment that has already been initialized

    Args:
        params: experiment params dict
        potObj: a Biologic class object that has been initialized
        scopeObj: a Picoscope class object that has been initialized

    Returns:
        experiment output: channel a-d and time data
    '''

    # get experiments ready to run
    scopeObj.loadExperiment(params)
    potObj.loadExperiment(params)

    # initialize the picoscope thread before starting the stream to minimize delay
    #   we are threading the runStream and runExperiment functions to allow the callback function to run while the potentiostat
    #   experiment starts. This avoids issues at higher speeds where the first ~30 ms of data are missed
    #   due to the limited memory of the Picoscope and the longer run time of pot.runExperiment()
    # todo: do this in the correct multiprocessing syntax (if __name__ = '__main__')
    picoThread = threading.Thread(target = scopeObj.runStream, name = "Pico")
    # potThread = threading.Thread(target = pot.runExperimentWithoutData, name = "Pot", daemon = False)
    scopeObj.initStream()

    # start streaming thread and then start the potentiostat
    picoThread.start()
    time.sleep(0.1)
    potObj.runExperimentWithoutData()

    # wait for picoThread to finish
    picoThread.join()

    # gather data
    a, b, c, d, t = scopeObj.channelAData, scopeObj.channelBData, scopeObj.channelCData, scopeObj.channelDData, scopeObj.time

    return a, b, c, d, t

# def runMultiprocess(params, potObj, scopeObj):
#     '''
#     Threaded call to run an experiment that has already been initialized
#
#     Args:
#         params: experiment params dict
#         potObj: a Biologic class object that has been initialized
#         scopeObj: a Picoscope class object that has been initialized
#
#     Returns:
#         experiment output: channel a-d and time data
#     '''
#
#     # get experiments ready to run
#     scopeObj.loadExperiment(params)
#     potObj.loadExperiment(params)
#
#     # initialize the picoscope thread before starting the stream to minimize delay
#
#
#     # potThread = threading.Thread(target = pot.runExperimentWithoutData, name = "Pot", daemon = False)
#     scopeObj.initStream()
#
#     # start streaming thread and then start the potentiostat
#     #   we are threading the runStream and runExperiment functions to allow the callback function to run while the potentiostat
#     #   experiment starts. This avoids issues at higher speeds where the first ~30 ms of data are missed
#     #   due to the limited memory of the Picoscope and the longer run time of pot.runExperiment()
#     # todo: do this in the correct multiprocessing syntax (if __name__ = '__main__')
#     # if this fails: consider doing the entire pico init/load/run on a separate process, may be issues with shared resources
#     #   another option is to run the potentiostat through the multiprocess since we aren't grabbing data from it
#     #   would need to make a wrapper function that takes the parameters and the pot object.
#     #       BUT the connection probably isn't pickleable for any of these...
#     #       might be worth it to try, but will likely need a whole revision of the experiment architecture to run multiprocess
#     #       need 'reload' parallel process
#     # ... forking reruns whole process on windows
#     if __name__ == '__main':
#         set_start_method('fork')
#         picoProcess = Process(target = scopeObj.runStream)
#         picoProcess.start()
#
#     time.sleep(0.1)
#     potObj.runExperimentWithoutData()
#
#     # wait for picoThread to finish
#     picoProcess.join()
#
#     # gather data
#     #   unclear if the scopeObj from the process will be different from the original thread
#     #   may need to implement a connection object - would be limited to 32MB per array...
#     #   maybe use a shared ctypes array instead? or use shared_memory
#     a, b, c, d, t = scopeObj.channelAData, scopeObj.channelBData, scopeObj.channelCData, scopeObj.channelDData, scopeObj.time
#
#     return a, b, c, d, t

##########################################
##### analysis for impedance data #########
############################################

# steps: get awg, trigger, and current data @ frequency
#   get current after trigger
#   find phasor relation between awg and current
#       find value and amplitude of maxima
#       calculate avg shift and amplitude change
#       Magnitude |Z| : I / V (ratio of maxima, in ohms)
#       Phase phi: avg shift / period * 2pi
#       Z' = |Z| cos phi
#       Z'' = |Z| sin phi
def analyzeImpedance(dat):

    # initialize data lists
    f = []
    z1 = []
    z11 = []
    mag = []
    phase = []

    # iterate through data
    for key in dat.keys():

        if type(key) == int:

            # gather freq, awg, awgtime, trigger, time, and current data
            period = dat[key]['vtPeriod']
            freq = 1 / period

            trig = dat[key]['potentiostatTrigger']
            current = dat[key]['potentiostatCurrent']
            fullTime = dat[key]['time']

            # find trigger point
            triggerIndex = np.argmax(trig > 1000)
            triggerTime = fullTime[triggerIndex]

            # grab all current data after trigger
            expCurrent = current[triggerIndex:]

            # create x-axis for current data that starts at 0
            expTime = np.linspace(0, fullTime[-1] - triggerTime, len(expCurrent))

            plt.plot(expTime, expCurrent)
            plt.show()

            # grab locations and values of max and min for current
            # strategy: partition by period (starting a little offset to avoid the current spike on trigger)
            dt = expTime[1] - expTime[0]
            periodRange = math.floor(period / dt)
            extrema = []
            extremaTimes = []
            for i in range(math.floor(expTime[-1] / period)):
                startIndex = 100 + i * periodRange # starting a little offset to avoid trigger spike
                currentRange = expCurrent[startIndex : startIndex + periodRange]

                if len(currentRange) == 0:
                    break

                # max first to match sine wave
                extrema.append(np.max(currentRange))
                extrema.append(np.min(currentRange))

                maxIndex = np.argmax(currentRange)
                minIndex = np.argmin(currentRange)
                extremaTimes.append(expTime[minIndex + startIndex])
                extremaTimes.append(expTime[maxIndex + startIndex])

            if len(extrema) == 0:
                break

            f.append(freq)

            # calculate the maxima of a sine function with freq and no phase shift
            # max is 1/4 of period, min is 3/4 of period
            calcExtrema = np.arange(0.25 * period, expTime[-1], 0.5 * period)
            # need to trim calcExtrema to match length of extrema time
            awgExtrema = calcExtrema[:len(extremaTimes)]

            # calculate magnitude and phase. convert to z' and z''
            magArray = abs(np.array(extrema) / 0.01)
            magVal = np.mean(magArray)
            mag.append(np.mean(magVal))

            shiftArray = extremaTimes - awgExtrema
            phaseVal = (np.mean(shiftArray) / period) * 2 * np.pi
            phase.append(phaseVal)

            z1val = magVal * math.cos(phaseVal)
            z11val = -1 * magVal * math.sin(phaseVal)
            z1.append(z1val)
            z11.append(z11val)

            # write into data dict
            dat[key]['|Z|'] = magVal
            dat[key]['phase'] = phaseVal
            dat[key]["Z'"] = z1val
            dat[key]["-Z''"] = z11val

    # save data
    data.savePickle(dat)

    # show bode and nyquist plots
    fix, ax = plt.subplots(2, 2)
    ax[0, 0].plot(z1, z11)
    ax[0, 1].plot(f, mag)
    ax[1, 1].plot(f, phase)
    plt.show()


# Applies a Savitzky-Golay filter to the data and optionally takes its first or second derivative
# Inputs the y-data ('voltage'), x-data ('time') along with 3 auxiliary parameters: the window length of filtering (defaults to 9),
#       the order of polynomials used to fit (defaults to 3), and the derivative order (default to 0, must be less than polynomial order)
# outputs the filtered data or its requested derivative. The output has the same shape as the input
def savgolFilter(yDat, xDat, windowLength = 9, polyOrder = 3, derivOrder = 0):

    # calculate the spacing of the xData
    xDelta = xDat[1] - xDat[0]

    return scipy.signal.savgol_filter(yDat, windowLength, polyOrder, derivOrder)

# Similar to zeroCrossings, listExtrema takes y-values of a function, y-values of its derivative, and the x-values and returns
# an 2 x number of extrema array that correspond to the extrema of the function (i.e. the (x, y) values where the derivative crosses zero)
# inputs the yData array (i.e. 'voltage'), derivative of yData (i.e. 'savgol_1'), and the x-data ('time')
#   optional input minimum to specify the minimum y-value of an extremum to count it
# returns an array of coordinates
def listExtrema(yDat, deriv, xDat, minimum = None):

    # find the indices where zero crossing occurs
    # implementation taken from https://stackoverflow.com/questions/3843017/efficiently-detect-sign-changes-in-python
    zeroCrossingIndices = np.where(np.diff(np.signbit(deriv)))[0]

    extremaList = []
    for i in zeroCrossingIndices:
        if minimum == None:
            extremaList.append([xDat[i], yDat[i]])
        elif abs(yDat[i]) >= minimum:
            extremaList.append([xDat[i], yDat[i]])

    return np.array(extremaList)

def testPotentiostat(params):
    pot = bio.Biologic(params)
    time.sleep(1)
    pot.loadExperiment(params)
    time.sleep(1)
    pot.runExperimentWithoutData()
    time.sleep(1)
    pot.close()