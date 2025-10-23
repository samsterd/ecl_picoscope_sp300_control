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
from multiprocessing import Process, Queue, JoinableQueue, set_start_method, shared_memory

# experiment functions will live here. Eventually this will become more systematic
#todo: MULTI EXPERIMENTS ARE NOT SAVING DATA!!!!

def runMultiParamList(initParams: dict, paramLists: dict, downTime: float):
    '''
    Runs a series of experiments based off of an initial set of parameters and a list of parameters and the values to
    vary.
    NOTE: this has very little error checking right now. There is a lot of room for unexpected behavior, be very careful
    using this

    Args:
        initParams (dict): an experimentalParameters dict. The values in keys contained in paramLists will be ignored
        paramLists (dict): a dict with keys that are in the initParams dict and values as a list of experimental parameters to
            iterate through. The length of each list must be the same
        downTime (float): amount of time, in seconds, to wait between executing each experiment. During this time the
            potentiostat is not collecting data and no current is flowing through the setup

    Returns:
        None. Data saved in the file specified
    '''

    # error check the param lists dict
    paramListsLen = -1
    for val in paramLists.values():
        if type(val) != list and type(val) != np.ndarray:
            raise TypeError("runMultiParamList: a value in the paramLists dict is not a list")
        else:
            if paramListsLen == -1:
                paramListsLen = len(val)
            elif len(val) != paramListsLen:
                raise ValueError("runMultiParamList: " + str(val) + " does not match the length of other value lists.")

    # generate a list of dicts to pass into multiProcessExperimentMain
    # this whole design is kind of clunky. Will probably want to redesign the whole thing in the future
    # first generate copies of initial params
    paramDicts = [copy.copy(initParams) for i in range(paramListsLen)]
    # next iterate through and replace each key with the correct value
    for i in range(paramListsLen):
        for key in paramLists.keys():
            paramDicts[i][key] = paramLists[key][i]

    multiProcessExperimentsMain(paramDicts, downTime)


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

# in order to write autoranging, we need the paramList to change while running. impedance with autorange needs a standalone function
def multiProcessImpedanceExperiment(params, startFreq, endFreq, nFreqs):
    '''
    Implements an impedance-like experiment with autoranging current within the multiprocessing framework

    Args:
        params (dict) : starting experiment params
        startFreq (float) : starting frequency
        endFreq (float) : ending frequency
        nFreqs (int) : number of frequencies to test. These will be sampled in logspace
    Returns:
        None. Data saved or plotted as defined in the input params
    '''

    # create freq list
    freqs = np.logspace(startFreq, endFreq, nFreqs)

    # overwrite initial parameters list to make this an impedance experiment
    runningParams = copy.copy(params)
    runningParams['vtFunc'] = pico.testVT
    runningParams['vtFuncArgs'] = ()
    runningParams['vStep'] = [0]

    # initialize database
    if params['save']:
        db = data.Database(runningParams)

    # set up queue, start pico process
    multiQ = JoinableQueue()
    picoProcess = Process(target = multiProcessExperimentsPico, args = [multiQ])
    picoProcess.start()

    # connect to biologic
    pot = bio.Biologic(runningParams)

    # experiment loop
    for i in range(len(freqs)):

        run = True
        rerun = False # track whether this is the first run at a given frequency. this is needed for writing/overwriting parameters

        while run and runningParams['currentRange'] >= 4: # 4 is lower limit for current model

            f = freqs[i]
            print(f)
            runningParams['vtFuncKwargs'] = {'freq' : f, 'amp' : 0.01}
            runningParams['vtPeriod'] = 1 / f
            runningParams['tStep'] = max(5e-8, runningParams['vtPeriod'] / 1000)
            runningParams['experimentTime'] = max(10.9 * runningParams['vtPeriod'], 0.000160) # add extra time. to preserve 10kS, min time is 16 ns * 10k = 160 us
            runningParams['vStepTime'] = [max(10 * runningParams['vtPeriod'], 0.001)] # CA runs longer to avoid issues with trigger disappearance
            # runningParams['scopeSamples'] = 9999
            # set up scope samples to avoid going above 250kS/s
            maxSamples = 2.5e5 * runningParams['experimentTime']
            if maxSamples < 2500:
                runningParams['scopeSamples'] = 2500
            else:
                runningParams['scopeSamples'] = maxSamples

            if params['save']:

                if not rerun:
                    # only write the parameters if this is the first run
                    #NOTE: this will not save the correct currentRange if it is later adjusted, but the recorded current will be correct
                    db.writeParameters(runningParams, i)

            # send 'exp' flag and running params into queue, wait for signal
            flag = ('exp', runningParams)
            multiQ.put(flag)
            multiQ.join()

            # load experiment, wait for start flag
            pot.loadExperiment(runningParams)
            startFlag = multiQ.get(block=True)

            # error check: if flag is wrong, things have desynced and we should abandon ship
            if startFlag != 'start':
                print("multiProcessExperimentsMain: Desync detected. Subthread returned " + str(
                    startFlag) + " instead of " +
                      "'start'. \nExperiment aborted.")
                multiQ.task_done()
                break

            # mark flag as received, start experiment
            multiQ.task_done()  # unclear if the order this is done matters
            pot.runExperimentWithoutData()

            # wait until potentiostat finished. This may take longer than the picoscope
            while not pot.experimentDoneQ:
                time.sleep(0.1)

            # wait for data to enter the queue
            dat = multiQ.get(block = True)

            # dat should be a dict with the following keys:
            #   'detector0', 'detector1','potentiostatOut', 'potentiostatCurrent', 'potentiostatTrigger',
            #   'time', 'awg', 'awgTime'
            # todo: write a format checker

            # check that dat is correct. If incorrect, desync error
            if type(dat) != dict:
                print("multiProcessExperimentsMain: Desync detected. Subthread returned " + str(
                    dat) + " instead of a data dict."
                           "Experiment aborted.")
                break

            # load current channel
            c = dat['potentiostatOut']

            #  current is low (potentiostat < 100 mV) and we could get better signal by reducing the range
            if np.max(abs(c)) < 100 and runningParams['currentRange'] > 4:

                runningParams['currentRange'] -= 1
                print('adjusting range')
                run = True
                rerun = True

            else:

                # switch run to break the while loop
                run = False
                rerun = False

                if params['save']:
                    db.writeData(dat, False, 'experimentNumber', i)

                if params['plot']:
                    plotECL(dat)

            # signal data processing is done for either outcome
            multiQ.task_done()

    # experiment finished, close everything
    endFlag = ('end', None)
    multiQ.put(endFlag)

    pot.close()
    multiQ.close()
    picoProcess.join(timeout = 10)
    picoProcess.close()

def multiProcessExperimentsMain(paramList : list, downTime : float = 0):
    '''
    Function for performing experiments using multiprocessing. Runs in tandem with multiProcessExperimentsPico to run a
    series of ECL experiments using multiprocessing to ensure the Picoscope does not miss the trigger pulse at short
    experiment times.

    The general control flow is:
        Initialize saving DB
        Start PicoProcess -> Pico connects
        Connect to Biologic, send 'exp' flag
        Both processes enter experiment loop:
            Pico loads experiment, starts streaming, waits for 'start' flag to record
            Biologic loads experiment, sends the 'start' flag, runs experiment
            Pico puts data in queue while biologic waits
            Pico waits for next signal (stop or new experiment)
            Biologic saves/plots
            If at end of paramList:
                Send 'end' signal. Loop breaks: Pico closes connection, joins main thread, biologic disconnects, experiment over
            If more experiments:
                Send 'exp' signal, restart experiment loop

    todo for future: add error catching and timeouts to avoid program hanging if one of the instruments misses a trigger or something

    Args:
        paramList: a list of experiment param dicts. They will be executed in order
                NOTE: the save file will be defined by the first paramList. all experiments will be saved to the same file
                if the first experiment has save = False, no data will be saved
        downTime (float): amount of time, in seconds, to wait between executing each experiment. During this time the
            potentiostat is not collecting data and no current is flowing through the setup

    Returns:
        None. Data is plotted and saved as defined in the params
    '''
    #URGENT TODO: SAVING IS NOT WORKING PROPERLY. SAVES COLUMN NAMES BUT NO DATA! DOES NOT MAKE NEW ROWS FOR EACH EXPERIMENT!
    # check that input is a list of dicts
    # todo: write an experimentParamsQ function to check formatting
    for param in paramList:
        if type(param) != dict:
            raise TypeError("multiProcessExperimentsMain: invalid input, paramList must be a list of dicts.")

    firstExp = paramList[0]
    expNumber = 0

    # set up save file
    if firstExp['save']:
        db = data.Database(firstExp)

    # set up queue, start pico process
    multiQ = JoinableQueue()
    picoProcess = Process(target = multiProcessExperimentsPico, args = [multiQ])
    picoProcess.start()

    # connect to biologic
    pot = bio.Biologic(firstExp)

    # enter experiment loop
    for exp in paramList:

        # need to add the experimentNumber value to the experiment parameters for saving purposes
        exp['experimentNumber'] = expNumber

        # if saving, write parameters into db
        if exp['save']:
            print('saving')
            db.writeData(exp, newRow = True, keyCol = 'experimentNumber', keyVal = expNumber)

        # send 'exp' flag and first params into queue, wait for signal
        flag = ('exp', exp)
        multiQ.put(flag)
        multiQ.join()

        # load experiment, wait for start flag
        pot.loadExperiment(exp)
        startFlag = multiQ.get(block = True)

        # error check: if flag is wrong, things have desynced and we should abandon ship
        if startFlag != 'start':
            print("multiProcessExperimentsMain: Desync detected. Subthread returned " + str(startFlag) + " instead of " +
                 "'start'. \nExperiment aborted.")
            multiQ.task_done()
            break

        # mark flag as received, start experiment
        multiQ.task_done() # unclear if the order this is done matters
        pot.runExperimentWithoutData()

        # wait for data to enter the queue
        dat = multiQ.get(block = True)

        # dat should be a dict with the following keys:
        #   'detector0', 'detector1','potentiostatOut', 'potentiostatCurrent', 'potentiostatTrigger',
        #   'time', 'awg', 'awgTime'
        # todo: write a format checker

        # check that dat is correct. If incorrect, desync error
        if type(dat) != dict:
            print("multiProcessExperimentsMain: Desync detected. Subthread returned " + str(dat) + " instead of a data dict."
                 "Experiment aborted.")
            break

        # save and plot data
        if exp['save']:
            # this will raise an error if the data keys are not correct, but we should still check earlier
            print('saving')
            db.writeData(dat, newRow = False, keyCol = 'experimentNumber', keyVal = expNumber)

        if exp['plot']:
            plotECL(dat)

        # signal that data is done processing
        multiQ.task_done()

        if downTime > 0:
            time.sleep(downTime)

        expNumber += 1

    # finished all experiments
    # Send 'end' flag, disconnect, join threads
    endFlag = ('end', None)
    multiQ.put(endFlag)

    pot.close()
    multiQ.close()
    picoProcess.join(timeout = 10)
    picoProcess.close()

def multiProcessExperimentsPico(queue):
    '''
    Secondary thread of multiProcessExperimentsMain. Runs Picoscope, gathers data, and syncs with main function. See
    multiProcessExperimentsMain for documentation of overall control flow

    Args:
        queue: Multiprocess queue for communicating with main function. Three types of messages are sent:
            Experiment flag: used for controlling the central loop. Either ('exp', {params}) to run a new experiment or
                ('end', None) to break the loop and disconnect
            Control flag: sent back and forth to synchronize with main
            Data: a list of numpy arrays sent to main as the result of an experiment
    Returns:
         None. Data is saved or plotted in main
    '''
    # connect
    scope = pico.Picoscope()

    # enter main loop
    while True:

        # wait for experiment flag
        # todo: add a reasonable timeout?
        expFlag = queue.get(block = True)

        # check expFlag is formatted correctly
        if not validExpFlagQ(expFlag):
            print("multiProcessExperimentsPico: incorrect flag passed to sub-thread. Expected and expFlag, but received "+
                  str(expFlag) + ".\nAborting experiment.")
            queue.task_done()
            break

        # check for end flag and break
        if expFlag[0] == 'end':
            queue.task_done()
            break

        # load parameters, send task_done
        expParams = expFlag[1]
        scope.loadExperiment(expParams)
        queue.task_done()

        # start stream, send 'start' flag, run
        scope.initStream()
        queue.put('start')
        a, b, c, d, t = scope.runStream()

        # format data into dict
        current = scope.voltageToPotentiostatCurrent(c, expParams['currentRange'])
        # awg data needs to be converted to numpy arrays since pickle can't handle ctype arrays
        awg = np.array(scope.awgBuffer)
        awgTime = np.array(scope.awgTime)
        dat = {
            'detector0' : a,
            'detector1' : b,
            'potentiostatOut' : c,
            'potentiostatCurrent' : current,
            'potentiostatTrigger' : d,
            'time' : t,
            'awg' : awg,
            'awgTime' : awgTime
        }

        # enqueue data, wait for queue join() before restarting loop
        # note for control: it is essential that we check the data was taken off the queue before waiting for the next flag,
        #   otherwise we will just read the data as the next flag and it will be bad
        queue.put(dat)
        queue.join()

    # received 'end' or desync error: disconnect, join with main
    scope.closePicoscope()

def validExpFlagQ(flag):
    '''
    Helper function that evaluates if an input is a valid expFlag used to communicate between processes
    Checks it is a 2-tuple that is either ('exp', expParams) or ('end', None)

    Args:
         flag : thing to test
    Returns:
        bool : is flag an expFlag
    '''
    if type(flag) != tuple:
        return False

    if len(flag) != 2:
        return False

    if flag[0] == 'exp':
        if validExpParamQ(flag[1]):
            return True
        else: return False

    elif flag[0] == 'end':
        if flag[1] == None:
            return True
        else:
            return False

    return False

def validExpParamQ(param):
    '''
    Helper function that evaluates if an input is a valid experiment params dict
    todo:  expandf this, currently just testing if its a dict
    Args:
        param : anything we want to test
    Returns:
        bool : is it a valid exp param dict
    '''
    if type(param) == dict:
        return True
    else:
        return False

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

def plotECL(dat : dict):
    '''
    Plots the ECL data contained in a dict. Results in a 6 panel figure:
        0,0 : AWG wave
        0,1 : channel A
        0,2 : channel B
        1,0 : channel C
        1,1 : channel C converted to current
        1,2 : channel D

    Args:
        dat (dict) : must contain the following keys: 'detector0', 'detector1','potentiostatOut', 'potentiostatCurrent',
        'potentiostatTrigger', 'time', 'awg', 'awgTime'
    '''
    # unpack dat for easier use
    t = dat['time']
    a = dat['detector0']
    b = dat['detector1']
    c = dat['potentiostatOut']
    current = dat['potentiostatCurrent']
    d = dat['potentiostatTrigger']
    awg = dat['awg']
    awgTime = dat['awgTime']

    fig, ax = plt.subplots(2, 3)

    ax[0, 0].plot(awgTime, awg)
    ax[0, 1].plot(t, a)
    ax[0, 2].plot(t, b)
    ax[1, 0].plot(t, c)
    ax[1, 1].plot(t, current)
    ax[1, 2].plot(t, d)

    plt.show()

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