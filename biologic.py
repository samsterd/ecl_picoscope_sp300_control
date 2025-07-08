# control scripts for running the Biologic SP-300

import os
import sys
import time

import kbio.kbio_types as KBIO
from kbio.c_utils import c_is_64b
from kbio.kbio_api import KBIO_api
from kbio.kbio_tech import ECC_parm
from kbio.kbio_tech import get_experiment_data
from kbio.kbio_tech import get_info_data
from kbio.kbio_tech import make_ecc_parm
from kbio.kbio_tech import make_ecc_parms
from kbio.utils import exception_brief


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
        # gather parameters from input dict
        self.channel = params['potentiostatChannel']
        self.experiment = params['experimentType']
        if self.experiment != 'ocv' and self.experiment != 'ca':
            print('Invalid value for parameter \'experimentType\': ' + self.experiment +
                  '. Valid options are \'ca\' or \'ocv\'.')
            #todo: using sys.exit for errors to match convention of example code. Should be changed to raising exceptions
            sys.exit(-1)

        # get path to driver files
        self.binaryPath = os.environ.get("ECLIB_DIR", f"C:{os.sep}EC-Lab Development Package{os.sep}lib")

        # get path to dll file
        if c_is_64b:
            DLL_file = "EClib64.dll"
        else:
            DLL_file = "EClib.dll"

        self.dllPath = f"{self.binaryPath}{os.sep}{DLL_file}"

        # initialize API
        self.api = KBIO_api(self.dllPath)
        self.version = self.api.GetLibVersion()

        # based on board_type, determine firmware filenames
        #todo: figure out connecting first - look into pyserial.list_ports, see what kinds of addresses api.Connect takes
        # find python version of BL_FINDECHEMDEV?
        self.usb = self.api.FindEChemDev
        self.id, self.deviceInfo = self.api.Connect(self.usb)

        # todo: BE CONSISTENT WITH _ VERSUS camelCase
        self.board_type = self.api.GetChannelBoardType(self.id, self.channel)

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
        self.api.LoadFirmware(self.id, self.channel_map, firmware=self.firmware_path, fpga=self.fpga_path, force=True)
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

        # match board type to the tech file
        match self.board_type:
            case KBIO.BOARD_TYPE.ESSENTIAL.value:
                self.tech_file = essential_tech_file
            case KBIO.BOARD_TYPE.PREMIUM.value:
                self.tech_file = premium_tech_file
            case KBIO.BOARD_TYPE.DIGICORE.value:
                self.tech_file = digicore_tech_file
            case _:
                print("> Board type detection failed")
                sys.exit(-1)

        self.loadExperiment()

    def loadExperiment(self):
        '''
        Gather experimental parameters from input dict depending on 'experimentType'. Load parameters into instrument.
        Baseline parameters for all techniques:
            iRange : ECC_parm("I_Range", int)
                Constants: 0 (100 pA), 1 (1 nA), 2 (10 nA),...7 (1 mA), 8 (10 mA), 9 (100 mA)
            eRange : ECC_parm("E_Range", int)
                Constants: 0 (2.5), 1 (5), 2 (10), 3 (auto)
            timebase : ECC_parm("tb", single)
            filter (make sure it is 0)
        Activate external control/output:
            xctr: bitfield for controlling certain options (p177 for info)
                0/1 (on/off)
                0b00001000 (enable external control)
            raux2: analog in 2 Erange (0)
            KBIO_MODE_FLOATING (1)
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



        pass