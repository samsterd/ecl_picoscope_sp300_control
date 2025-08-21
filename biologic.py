# control scripts for running the Biologic SP-300

import os
import sys
import time
import numpy as np
from dataclasses import dataclass

import kbio.kbio_types as KBIO
from kbio.c_utils import c_is_64b
from kbio.kbio_api import KBIO_api
from kbio.kbio_tech import ECC_parm
from kbio.kbio_tech import get_experiment_data
from kbio.kbio_tech import get_info_data
from kbio.kbio_tech import make_ecc_parm
from kbio.kbio_tech import make_ecc_parms
from kbio.utils import exception_brief
from kbio.kbio_types import ECC_PARM_ARRAY


# code adapted from example included with the EC-Lab package
# General approach:
#   Init: load the dll file, check the firmware
#   Load experiment: read experimental parameters, put them in the appropriate formate for LoadTechnique
#   Run experiment: run StartChannel, gather data, (save data)
#   Close: disconnect potentiostat
# Approach for ECL measurement:
#   Calculate CA parameters from ECL parameters
#   Init, Load Experiment, Load trigger
#   Get picoscope to runStream (put autotrigger delay high)
#   Run potentiostat experiment
#   Save data, autoanalyze, run new conditions

#todo 7/9:
#   test connection with instrument
#   implement triggering
#       trigger is a parameter passed to LoadTechnique
#   test OCV on dummy cell
#   write/test impedance on dummy cell
#   write full ECL experiment script

class Biologic:
    '''
    A class for interacting with the Biologic SP-300 through the EC-Lab developer's package. Primary experiments of
    interest are OCV and chronoamperometry

    '''

    def __init__(self, params : dict):
        '''
        Things to do when connecting:
            Open DLL file / check versions
            Connect to the potentiostat
        Running experiment:
            Retrieve channel info, test whether correct firmware is running
            Create potentiostat parameters from experiment parameters
            Load experiment into channel
            Start running
                Retrieve and save data
                Stop at correct time

        init: find and load essential files, check firmware version, etc. Based on example code in ex_tech_OCV.py
        '''
        # gather some parameters from input dict
        # some parameters are not loaded until the loadExperiment() function to allow them to be changed after loading firmware
        self.params = params
        self.channel = params['potentiostatChannel']
        self.experiment = params['experimentType']

        # do some basic error checking
        if self.experiment != 'ocv' and self.experiment != 'ca':
            raise ValueError('Invalid value for parameter \'experimentType\': ' + self.experiment +
                  '. Valid options are \'ca\' or \'ocv\'.')
        if self.channel != 1 and self.channel != 2:
            raise ValueError("potentiostatChannel must be 1 or 2.")

        # get path to driver files
        self.binaryPath = os.environ.get("ECLIB_DIR", f"C:{os.sep}EC-Lab Development Package{os.sep}lib")

        # get path to dll file
        if c_is_64b:
            DLL_file = "EClib64.dll"
            blfind = "blfind64.dll"
        else:
            DLL_file = "EClib.dll"
            blfind = "blfind.dll"

        self.dllPath = f"{self.binaryPath}{os.sep}{DLL_file}"
        self.blfindPath = f"{self.binaryPath}{os.sep}{blfind}"

        # initialize API
        self.api = KBIO_api(self.dllPath, self.blfindPath)
        self.version = self.api.GetLibVersion()

        # based on board_type, determine firmware filenames
        #todo: figure out connecting first - look into pyserial.list_ports, see what kinds of addresses api.Connect takes
        # find python version of BL_FINDECHEMDEV?
        self.usb = self.api.FindEChemUsbDev()[0]
        # check that an instrument was found
        if not isinstance(self.usb, KBIO.USB_device):
            raise RuntimeError("Potentiostat USB address not found. Check that it is connected and try again.")

        self.id, self.deviceInfo = self.api.Connect(self.usb.address)

        # todo: BE CONSISTENT WITH _ VERSUS camelCase
        self.board_type = self.api.GetChannelBoardType(self.id, self.channel)
        # self.board_type = KBIO.BOARD_TYPE.PREMIUM.value
        #todo: just try setting board_type to premium or premium_p and seeing if it loads?
        match self.board_type:
            case KBIO.BOARD_TYPE.ESSENTIAL.value:
                self.firmware_path = "kernel.bin"
                self.fpga_path = "Vmp_ii_0437_a6.xlx"
            case KBIO.BOARD_TYPE.PREMIUM.value:
                self.firmware_path = "kernel4.bin"
                self.fpga_path = "vmp_iv_0395_aa.xlx"
            case KBIO.BOARD_TYPE.DIGICORE.value:
                self.firmware_path = "kernel.bin"
                self.fpga_path = ""
            case _:
                print("> Board type detection failed")
                sys.exit(-1)

        # load firmware
        self.channel_map = self.api.channel_map({self.channel})
        firmwareRes = self.api.LoadFirmware(self.id, self.channel_map, firmware=self.firmware_path, fpga=self.fpga_path, force=True)
        self.channel_info = self.api.GetChannelInfo(self.id, self.channel)

        if not self.channel_info.is_kernel_loaded:
            print("> kernel must be loaded in order to run the experiment")
            sys.exit(-1)

        # define experiment parameter files (these get picked later based on the firmware)
        if self.experiment == 'ocv':
            essential_tech_file = "ocv.ecc"
            premium_tech_file = "ocv4.ecc"
            digicore_tech_file = "ocv5.ecc"
        elif self.experiment == 'ca':
            essential_tech_file = "ca.ecc"
            premium_tech_file = "ca4.ecc"
            digicore_tech_file = "ca5.ecc"
        # define trigger files
        essentialTriggerFile = "TO.ecc"
        premiumTriggerFile = "TO4.ecc"
        digicoreTechFile = "TO5.ecc"

        # match board type to the tech file
        match self.board_type:
            case KBIO.BOARD_TYPE.ESSENTIAL.value:
                self.tech_file = essential_tech_file
                self.triggerTechFile = essentialTriggerFile
            case KBIO.BOARD_TYPE.PREMIUM.value:
                self.tech_file = premium_tech_file
                self.triggerTechFile = premiumTriggerFile
            case KBIO.BOARD_TYPE.DIGICORE.value:
                self.tech_file = digicore_tech_file
                self.triggerTechFile = digicoreTechFile
            case _:
                print("> Board type detection failed")
                sys.exit(-1)

        # set to floating ground if using external input (for premium boards only)
        # todo: figure out proper settings for this
        if self.board_type == KBIO.BOARD_TYPE.PREMIUM.value:

            # set WE/CE connections to standard
            conn = 0 # KBIO_CONN_STD

            if params['extInput']:
                gnd = 1 # KBIO_MODE_FLOATING
                self.api.SetHardConf(self.id, self.channel, conn, gnd)
            else:
                gnd = 0 # KBIO_MODE_GROUNDED
                self.api.SetHardConf(self.id, self.channel, conn, gnd)

        # self.loadExperiment()

    def loadExperiment(self, params):
        '''
        Gather experimental parameters from input dict depending on 'experimentType', converts them to ECC_Parms and calls
        api.LoadTechnique()

        Every ECC_Parm requires a name (str) and a value
        Baseline parameters for all techniques:
            iRange : ECC_parm("I_Range", int)
                Constants: 0 (100 pA), 1 (1 nA), 2 (10 nA),...7 (1 mA), 8 (10 mA), 9 (100 mA)
            eRange : ECC_parm("E_Range", int)
                Constants: 0 (2.5), 1 (5), 2 (10), 3 (auto) (always set to 2?)
            timebase : ECC_parm("tb", single) (do I need to set this?)
            filter (make sure it is 0)
        Activate external control/output:
            xctr: bitfield for controlling certain options (p177 for info)
                0/1 (on/off)
                0b00001000 (enable external control)
            raux2: analog in 2 Erange (0)
            KBIO_MODE_FLOATING (1)
            Trigger_Logic, int (0 or 1)
            Trigger_Duration, single (timebase is 20 us)
        Parameters for OCV:
            "duration": ECC_parm("Rest_time_T", float),
            "record_dt": ECC_parm("Record_every_dT", float),
            "record_dE": ECC_parm("Record_every_dE", float),
            "E_range": ECC_parm("E_Range", int),
            "timebase": ECC_parm("tb", int),
        Parameters for CA:
            "vStep" : ECC_parm("Voltage_step", array of 100 single
            "vsInit" : ECC_parm("vs_initial", array of 100 boolean, voltage step vs initial one ???
            "durationStep" : ECC_parm("Duration_step", array of 100 single, duration of each step
            "numberOfSteps" : ECC_parm("Step_number", number of steps - 1 (0 to 98)
            "recorddT" : ECC_parm("Record_every_dT", record every dt seconds
            "recorddI" : ECC_parm("Record_every_dI", record every dI amps
            "numberOfCycles" : ECC_parm("N_Cycles", number of times the whole process is repeated
                returns time (convert w/ BL_ConvertTimeChannelNumbericIntoSeconds)
                E(WE) (convert w/ BL_ConvertChannelNumbericIntoSingle)
                cycle
        :return:
        '''
        # update the input parameters, load the current range
        # todo: come up with a more intelligent way to overwrite params?
        self.params = params
        self.currentRange = self.params['currentRange']

        if type(self.currentRange) != int:
            raise TypeError("currentRange must be an int")
        if self.currentRange < 0 or self.currentRange > 9:
            raise ValueError("currentRange must be between 0 and 9. If you are unsure what current range to expect, set to 9"
                             "to avoid damaging the potentiostat")

        # generate parameters for the external input
        ext_parms = {
            "xctr": ECC_parm("xctr", int),  # option control bitfield parameter
            "raux2": ECC_parm("raux2", int)  # analog input voltage range
        }
        # unsure if raux should be 0 (+/- 2.5 V) or 1 (+/- 5V)
        praux2 = make_ecc_parm(self.api, ext_parms['raux2'], 1)
        if self.params['extInput']:
            # generate the ecc_parm for external input on
            pxctr = make_ecc_parm(self.api, ext_parms['xctr'], 0b00001000)
        else:
            pxctr = make_ecc_parm(self.api, ext_parms['xctr'], 0b00000000)

        match self.experiment:

            case 'ocv':

                # gather inputs, make sure the ints and floats are correct
                self.dt = float(self.params['dt'])
                self.duration = float(self.params['duration'])

                # generate parameters dict
                # todo: camelCase vs _underscroe is all over the place
                ocv_parms = {
                    "duration": ECC_parm("Rest_time_T", float),
                    "record_dt": ECC_parm("Record_every_dT", float),
                    "record_dE": ECC_parm("Record_every_dE", float),
                    "E_range": ECC_parm("E_Range", int),
                    "timebase": ECC_parm("tb", int),
                    "I_range" : ECC_parm("I_Range", int)
                }

                # make ecc_parms for the user inputs
                eccDuration = make_ecc_parm(self.api, ocv_parms['duration'], self.duration)
                eccdt = make_ecc_parm(self.api, ocv_parms['record_dt'], self.dt)
                eRange = make_ecc_parm(self.api, ocv_parms['E_range'], KBIO.E_RANGE["E_RANGE_10V"].value) # hard coding this to 10V
                iRange = make_ecc_parm(self.api, ocv_parms['I_range'], self.currentRange)
                eccParms = make_ecc_parms(self.api, eccDuration, eccdt, eRange, iRange, pxctr, praux2)

            case 'ca':

                # gather inputs and cast to the correct types
                self.dt = float(self.params['dt'])
                self.vStep = [float(v) for v in self.params['vStep']]
                self.vStepTime = [float(t) for t in self.params['vStepTime']]
                self.numberOfSteps = len(self.vStep)
                self.startStep = int(self.params['startStep'])
                self.vInit = [i == self.startStep for i in range(self.numberOfSteps)] # bool array that is only true at start step
                self.numberOfCycles = int(self.params['numberOfCycles'])

                # do some basic error checking
                if self.numberOfSteps != len(self.vStepTime):
                    raise ValueError("'vStep' and 'vStepTime' must have equal lengths")
                if self.startStep < 0 or self.startStep >= self.numberOfSteps:
                    raise ValueError("'startStep' must be between 0 and len(vStep) - 1")

                # convert the steps specified in input to the right format
                stepParm = [VoltageStep(self.vStep[i], self.vStepTime[i], self.vInit[i]) for i in range(self.numberOfSteps)]

                # generate parameters dict
                ca_parms = {
                    "vStep" : ECC_parm("Voltage_step", float),
                    "vStepTime" : ECC_parm("Duration_step", float),
                    "vsInit" : ECC_parm("vs_initial", bool),
                    "numberOfSteps" : ECC_parm("Step_number", int),
                    "dt" : ECC_parm("Record_every_dT", float),
                    "di" : ECC_parm("Record_every_dI", float),
                    "numberOfCycles" : ECC_parm("N_Cycles", int),
                    "iRange" : ECC_parm("I_Range", int)
                }

                # create ecc_parms from inputs. this can probably be turned into a more efficient function later
                pNumberOfSteps = make_ecc_parm(self.api, ca_parms['numberOfSteps'], self.numberOfSteps - 1) # the input parameter is 0-indexed
                pdt = make_ecc_parm(self.api, ca_parms['dt'], self.dt)
                pNumberOfCycles = make_ecc_parm(self.api, ca_parms['numberOfCycles'], self.numberOfCycles)
                piRange = make_ecc_parm(self.api, ca_parms['iRange'], self.currentRange)
                pSteps = []
                for i, step in enumerate(stepParm):
                    pSteps.append(make_ecc_parm(self.api, ca_parms['vStep'], step.voltage, i))
                    pSteps.append(make_ecc_parm(self.api, ca_parms['vStepTime'], step.duration, i))
                    pSteps.append(make_ecc_parm(self.api, ca_parms['vsInit'], step.vsInit, i))

                # todo: add 'xctr' parm for ext input

                eccParms = make_ecc_parms(self.api, *pSteps, pNumberOfSteps, pdt, pNumberOfCycles, piRange, pxctr, praux2)

        if self.params['trigger']:

            # calculate the duration of the trigger pulse based on the experiment type and timing
            if self.params['experimentType'] == 'ocv':
                expTime = self.params['duration']
            elif self.params['experimentType'] == 'ca':
                expTime = sum(self.params['vStepTime'])
            else:
                expTime = self.params['experimentTime']
            # set trigger duration to 5% of experiment time or the minimum time (20 us), whichever is larger
            self.triggerDuration = max(0.05 * expTime, 2e-5)

            # make ecc params for trigger out experiment
            to_parms = {
                "triggerLogic" : ECC_parm("Trigger_Logic", int),
                "triggerDuration" : ECC_parm("Trigger_Duration", float)
            }
            pTriggerLogic = make_ecc_parm(self.api, to_parms['triggerLogic'], 1)
            pTriggerDuration = make_ecc_parm(self.api, to_parms['triggerDuration'], self.triggerDuration) # hard code duration to 10 ms, duration doesn't matter too much
            triggerParms = make_ecc_parms(self.api, pTriggerLogic, pTriggerDuration)

            # load trigger and actual experiment. Trigger must be loaded first
            self.api.LoadTechnique(self.id, self.channel, self.triggerTechFile, triggerParms, first = True, last = False, display = False)
            self.api.LoadTechnique(self.id, self.channel, self.tech_file, eccParms, first = False, last = True, display = False)

        else:
            # load technique without trigger
            self.api.LoadTechnique(self.id, self.channel, self.tech_file, eccParms, first=True, last=True,
                                       display=False)

    def runExperiment(self):

        self.api.StartChannel(self.id, self.channel)

        dataDictList = [] # initialize data storage from buffer
        while True:
            # todo: check if self.data is a buffer that needs to write somewhere or all of the data gets safely recorded there
            #       probably not viable for big data sets, but should be fine for CA?
            dataBuffer = self.api.GetData(self.id, self.channel)
            status, techName = get_info_data(self.api, dataBuffer)

            # unload data from bufferas a generator
            bufferGenerator = get_experiment_data(self.api, dataBuffer, techName, self.board_type)

            # convert the generator into a list of dicts and add to dataDictList
            bufferDictList = [d for d in bufferGenerator]
            dataDictList.extend(bufferDictList)

            if status == "STOP":
                break
            time.sleep(1)

        # convert data into a more useable format
        # todo: get_experiment_data only returns nicely formatted things for 'ocv' and 'cp'
        #   that said, we can probably dump it b/c we want the data on the scope anyway + we can't run this control loop while
        #       also running scope streaming without a lot of annoyance developing a parallel algorithm
        self.data = self.processData(dataDictList)
        return self.data

    def runExperimentWithoutData(self):
        '''
        Runs experiment without a control loop. This will result in only the first ~1000 data points being recovered
        due to the potentiostat memory filling

        :return:
        '''
        self.api.StartChannel(self.id, self.channel)

    def experimentDoneQ(self):
        '''
        Checks the status of a running experiment and returns True when get_info_data returns a status of "STOP"

        Args: None
        Returns: bool
        '''
        data = self.api.GetData(self.id, self.channel)
        status, tech = get_info_data(self.api, data)

        if status == "STOP":
            return True
        else:
            return False

    def experimentStatus(self):
        '''
        Repots the status of the experiment

        Args: none
        Returns: str status
        '''
        data = self.api.GetData(self.id, self.channel)
        status, tech = get_info_data(self.api, data)
        return status

    def gatherData(self):
        '''
        Gathers data after experiment has been run and formats it
        The potentiostat only has room for ~1000 data points, so this will not gather everything efficiently

        :return:
        '''
        dataBuffer = self.api.GetData(self.id, self.channel)

        status, techName = get_info_data(self.api, dataBuffer)

        # unload data from buffer as a generator of data rows
        bufferGenerator = get_experiment_data(self.api, dataBuffer, techName, self.board_type)
        # be careful about generating list of rows for large data sets
        bufferDictList = [d for d in bufferGenerator]

        if self.experiment == 'ocv':

            self.data = self.processData(bufferDictList)
            return self.data

        elif self.experiment == 'ca':

            # bufferDictList is raw, unlabeled data. Need to format it nicely
            rows = len(bufferDictList)
            cols = len(bufferDictList[0])

            # gather some info for data conversion
            timebase = dataBuffer[0].TimeBase

            # initialize data dict
            self.data = {'t' : np.empty(rows),
                    'Ewe' : np.empty(rows),
                    'Iwe' : np.empty(rows),
                    'cycle' : np.empty(rows)}

            # iterate through bufferDictList and populate data dict
            for i in range(rows):

                # the rows are strings of hex numbers for some ungodly reason and so must be converted to usable numbers
                rowStrs = bufferDictList[i]
                row = [int(d, 16) for d in rowStrs]

                # convert time data based on 'CP' branch of get_experiment_data
                tHigh, tLow = row[0], row[1]
                tRel = (tHigh << 32) + tLow
                self.data['t'][i] = tRel * timebase
                self.data['Ewe'][i] = self.api.ConvertChannelNumericIntoSingle(row[2], self.board_type)
                self.data['Iwe'][i] = self.api.ConvertChannelNumericIntoSingle(row[3], self.board_type)
                self.data['cycle'][i] = row[4]

            return self.data

    def processData(self, listOfDicts):
        '''
        Takes the output from runExperiment loop (a list of dicts) and converts it to a dict of arrays
        This is effectively a transpose and then converting to an array:
        The data is output as rows (with each column title being the keys of the dict), output has the same keys, but the values
        are an array of all row values

        :param experimentData:
        :return:
        '''

        # initialize the output dict as lists
        output = {}
        for key in listOfDicts[0].keys():
            output[key] = np.empty(len(listOfDicts))

        # iterate through datList and add the dat from each key
        for i in range(len(listOfDicts)):

            # we are going to assume the same keys exist for every data point
            row = listOfDicts[i]
            for key in row.keys():
                output[key][i] = row[key]

        return output

    def close(self):

        self.api.Disconnect(self.id)

# class for creating arrays of data for chronoamperometry experiments
@dataclass
class VoltageStep:
    voltage: float
    duration: float
    vsInit : bool = False