# functions for saving / loading / maniulating data

# things we will want to save:
#       all input parameters
#       awg function values
#       picoscope outputs
#       (not even going to attempt to read potentiostat output)
# In an ideal experiment, above repeats several times
#   probably easiest to repeat ultrasound method:
#       SQL table (params will have multiple rows now, + column for "Experiment Number" as primary key)
#           Params + data all in one table
#               dump data as numpy array binaries for speed

import sqlite3
import numpy as np
import io
import time
import os
import copy
import pickle
from tqdm import tqdm
from typing import Callable
import json
import bottleneck as bn
from matplotlib import pyplot as plt
import math
import scipy.signal

# Adapted from github.com/samsterd/ultrasonicTesting/database.py
# Class for creating/saving into SQlite Database during experiments
# Contains functions for initializing databases, saving experimental parameters, and reformatting/saving data from dictionaries
class Database:

    def __init__(self, params : dict):
        """
        A class for creating/saving/loading an SQlite database during ECL experiments

        Class methods:
            init(params : dict) : initialize the Database object, extract necessary information from the experimental parameters dict,
                and create the database file to hold the experimental data
            adaptArray(arr: numpy array): defines an adapter for converting numpy arrays to an sqlite-usable format
            convertArray(text): defines a converter for reading numpy arrays that have been saved using adaptArray
            dataTableInitializer(params : dict): creates the data table based on the experimental parameters
            parseQuery(inputDict: dict, newRow : bool, keyCol : str, keyVal, table : str): generates an SQlite query string and a list of data in order to write the input dict into a table
                Can either create a new row or update an existing row depending on the value of newRow
            write(query : str, vals : list): takes the output string from parseQuery and writes it into the database
            writeData(dataDict : dict, newRow : bool, keyCol : str, keyVal, table : str): a wrapper function which combines parseQuery and write into a single function
            writeParameters(params : dict, experimentNumber : int, table : str) : writes the current values of the experimental parameters along with extra metadata
        Class variables:
            connection: sqlite3 database connection
            cursor: sqlite3 cursor for interacting with the database
        """
        # check that experimentFolder and experimentName are specified in parameters
        # note that we don't check if the resulting filename is a valid path since that is apparently a pain in the ass on Python
        if 'experimentFolder' not in params.keys() or 'experimentName' not in params.keys():
            raise ValueError("experimentParams dict must include the keys 'experimentFolder' and 'experimentName'.")

        # generate the filename in a copy of params to avoid altering inputs
        copyParams = copy.copy(params)
        copyParams['fileName'] = copyParams['experimentFolder'] + copyParams['experimentName']

        #create db connection, create cursor
        # first check if the requested filename already exists. If so print a warning and generate a new filename with the timestamp
        if os.path.exists(copyParams['fileName'] + '.sqlite3'):
            # this is kind of cursed, sorry
            fileName = copyParams['fileName'] + str(int(time.time())) + '.sqlite3'
            copyParams['fileName'] = fileName
            print("Database Warning: requested save file already exists. \nExperiment will be saved as " + fileName + " instead.")
        else:
            fileName = copyParams['fileName'] + '.sqlite3'

        self.connection = sqlite3.connect(fileName)
        self.cursor = self.connection.cursor()

        # register adapters for converting between numpy arrays and text
        # modified from https://stackoverflow.com/questions/18621513/python-insert-numpy-array-into-sqlite3-database
        # Converts np.array to TEXT when inserting
        sqlite3.register_adapter(np.ndarray, self.adaptArray)
        # Converts TEXT to np.array when selecting
        sqlite3.register_converter("array", self.convertArray)

        # create data table sql query string with correctly labeled columns
        dataTableInit = self.dataTableInitializer(copyParams)

        # create the data table
        self.cursor.execute(dataTableInit)

    # define adapters for converting numpy arrays to sqlite-usable format
    # copied from stackoverflow: https://stackoverflow.com/questions/18621513/python-insert-numpy-array-into-sqlite3-database
    @staticmethod
    def adaptArray(arr):
        """
        Define adapters for converting numpy arrays to sqlite-usable format
        Taken from http://stackoverflow.com/a/31312102/190597 (SoulNibbler)
        This operation of storing numpy arrays as raw binary in an sqlite3 table is mildly cursed, but it makes a MASSIVE
        speed improvement versus converting the data arrays to strings

        Args:
            arr (numpy array) : the array to be converted
        Returns:
            Raw binary of the numpy array
        """
        out = io.BytesIO()
        np.save(out, arr)
        out.seek(0)
        return sqlite3.Binary(out.read())

    # define adapters for converting numpy arrays to sqlite-usable format
    # copied from stackoverflow: https://stackoverflow.com/questions/18621513/python-insert-numpy-array-into-sqlite3-database
    @staticmethod
    def convertArray(text):
        """
        Defines a converter for reading binary "array" types in the saved sqlite file and interpreting them as numpy arrays
        copied from stackoverflow: https://stackoverflow.com/questions/18621513/python-insert-numpy-array-into-sqlite3-database
        This operation of storing numpy arrays as raw binary in an sqlite3 table is mildly cursed, but it makes a MASSIVE
        speed improvement versus converting the data arrays to strings

        Args:
            text ("array" binary read from sqlite3 table)
        Returns:
            numpy array
        """
        out = io.BytesIO(text)
        out.seek(0)
        return np.load(out)

    def dataTableInitializer(self, params : dict):
        """
        Generates an SQL query string to intialize the data table based on the experimental parameters

        Args:
            params (dict): the experimental parameters dict
        Returns:
            query (str): an sqlite3 query string for creating the table with all appropriate columns
        """

        # data is name of TABLE. Not sure if we want this hardcoded
        # general table structure that is true in all experiments
        initTable = "CREATE TABLE IF NOT EXISTS data (\n"

        # need to create rows for all of the data that will be saved:
        #   experimentNumber (int, primary key)
        #   time started (real)
        #   channelA-D (array) - data label on each channel is hardcoded for now
        #   time (array)
        #   input wave (array)
        #   input wave time (array)

        initTable = initTable + '''
            experimentNumber INTEGER PRIMARY KEY,
            timeCollected REAL,
            detector0 array,
            detector1 array,
            potentiostatOut array,
            potentiostatTrigger array,
            potentiostatCurrent array,
            time array,
            awg array,
            awgTime array,'''

        # next add a column for each parameter in the input parameters dict
        paramString = ""
        for key in params.keys():

            keyString = key
            # need to determine sqlite data type based on param value
            valType = type(params[key])
            if valType == float:
                keyType = ' REAL,\n'
            elif valType == int:
                keyType = ' INT,\n'
            else:
                keyType = ' TEXT,\n'

            paramString += keyString + keyType

        # need to replace ending ",\n" with a ")" and add to the initTable string
        initTable += paramString.removesuffix(",\n") + ")"

        return initTable

    @staticmethod
    def parseQuery(inputDict: dict, newRow : bool, keyCol : str, keyVal, table: str = 'data'):
        """
        Takes a dict and turns it into an SQL-readable format for writing the data. Converts the data to safe types for
        the database (ints, floats, arrays, or strings) and either writes a new row or updates an existing row with the
        data

        Args:
            inputDict (dict): dict where the keys are columns and values are the data to write in those columns
            newRow (bool) : should the input data be written as a new row or update an existing row?
                If True : the INSERT operation is performed
                If False: the UPDATE operation is performed
            keyCol (str) : name of the column used to find the correct row to insert if newRow = False
                For best performance, experimentNumber is recommended
            keyVal : value of keyCol at the row to write in if newRow = False
            table (str) : the name of the table to write in
        Returns:
            query (str), vals (list) : a query string and the values as a list to be executed on the db connection
        """

        dictKeys = inputDict.keys()
        keyString = ', '.join([key for key in dictKeys])

        qMarks = "(" + ("?," * (len(dictKeys)-1)) + "?)"

        vals = [inputDict[key] for key in dictKeys]

        # vals need extra formatting to ensure they are all either an int, float, or array. Anything that isn't one of these
        # i.e. tuples or None or callables are converted to strings
        safeVals = [
            val if (type(val) == int or type(val) == float or type(val) == np.ndarray) else str(val) for val in vals
        ]

        # generate an insert or update string depending on newRow
        if newRow:
            query = 'INSERT INTO ' + table + ' (' + keyString + ') VALUES ' + qMarks + ';'
        else:
            query = 'UPDATE ' + table + ' SET (' + keyString + ') = ' + qMarks + ' WHERE ' + keyCol + ' = ?'
            safeVals.append(keyVal)

        return query, safeVals

    def write(self, query: str, vals: list):
        """
        Use the output of parseQuery to write into the database file

        Args:
            query (str): an sqlite3-readable query string
            vals (list): a list of values to write into the table
        Returns:
            lastrowid of the sqlite3 cursor after the data has been written
        """

        self.cursor.execute(query, vals)
        self.connection.commit()

        return self.cursor.lastrowid

    def writeData(self, dataDict : dict, newRow : bool, keyCol : str, keyVal, table : str = 'data'):
        """
        Wrapper function to combine generating query strings and writing into the database

        Args:
            dataDict (dict): a dict whose keys are columns and values are data to write into the table
            newRow (bool) : should the input data be written as a new row or update an existing row?
                If True : the INSERT operation is performed
                If False: the UPDATE operation is performed
            keyCol (str) : name of the column used to find the correct row to insert if newRow = False
                For best performance, experimentNumber is recommended
            keyVal : value of keyCol at the row to write in if newRow = False
            table (str) : the name of the table to write into

        Returns:
            None
        """

        query, vals = self.parseQuery(dataDict, newRow, keyCol, keyVal, table)
        self.write(query, vals)

    def writeParameters(self, params : dict, experimentNumber : int, table : str = 'data'):
        """
        Writes the current values of the experiment params dict, along with the experimentNumber and experimentTime,
        as a new row in the database

        Args:
            params (dict) : experimental params dict defined in main.py
            experimentNumber (int) : unique identifier of the current experiment. Must be unique within the table or an
                error will be raised
            table (str) : name of the table to write into

        Returns:
            None
        """
        # add the experimentNumber and timeCollected values to the parameters dict
        writeParams = params
        writeParams['experimentNumber'] = experimentNumber
        writeParams['timeCollected'] = time.time()

        self.writeData(writeParams, True, 'experimentNumber', experimentNumber, table)

    def close(self):
        """
        Wrapper for closing database connection

        Args:
            None
        Returns:
            None
        """
        self.connection.close()

'''
Adapted from ultrasonicTesting/pickleJar.py and sqliteUtils.py
Code for interfacing with data as a pickled dict
Format of data dict: Top level keys are integers or the string 'fileName'
Integers are the experimentNumber data points and their values are dicts whose keys are the data and parameters of each 
experiment in a file ( = the columns of the Sqlite database)
'fileName' value is a string that is the path to the pickle file
Example:
 { 0 : {'timeCollected' : ..., 'detector0' : [...], ...},
   1 : {...},
   'fileName' : 'C://path//to//data.pickle'
   }
   
This copying and modifying is somewhat inefficient, but I'd rather not reinvent the wheel at this point
'''

# Open a connection to the database specified in filepath and initialize a cursor
# Returns the database connection object and the initialized cursor
def openDB(fileName):

    connection = sqlite3.connect(fileName, detect_types=sqlite3.PARSE_DECLTYPES)

    # register numpy adapters
    sqlite3.register_adapter(np.ndarray, Database.adaptArray)
    sqlite3.register_converter("array", Database.convertArray)

    cursor = connection.cursor()

    return connection, cursor

# Inputs a cursor and a name for a table
# Outputs a list of strings with the name of every column
# Use to extract a list of parameters for a given experiment, or to check what axes were scanned along
def columnNames(cursor, table : str):

    # Generate PRAGMA table_info query
    query = "PRAGMA table_info(" + table + ")"

    res = cursor.execute(query).fetchall()

    # The result will be a list of tuples. The name of the column is entry 1
    names = []
    for column in res:
        names.append(column[1])

    return names

# Inputs a cursor and table name
# Outputs the number of rows in the table
def numberOfRows(cursor, table: str):

    query = "SELECT COUNT(*) FROM " + table

    res = cursor.execute(query)

    return res.fetchone()[0]

# Convert sqlite database
# Inputs a filename with .sqlite3 extension
# Creates a data dict and saves it as the same filename .pickle
# Returns the dataDict
def sqliteToPickle(file : str):

    # Open db connection
    con, cur = openDB(file)

    # get column names
    colNames = columnNames(cur, 'data')

    # find the position of collection_index, which is used to create keys for the dataDict
    indexPosition = colNames.index('experimentNumber')

    dataDict = {}

    # Generate the filename by removing .sqlite3 extension and .pickle extension
    pickleFile = os.path.splitext(file)[0] + '.pickle'
    dataDict['fileName'] = pickleFile

    # check if the pickle file already exists. If it does, print an error message and exit early
    if os.path.isfile(pickleFile):
        print("sqliteToPickle Warning: pickle file " + pickleFile + " already exists. Conversion aborted.")
        return -1

    # Gather the number of rows in the db
    numRows = numberOfRows(cur, 'data')

    # Create and execute a SELECT query to pull all of the data from the table
    selectQuery = "SELECT " + ", ".join(colNames) + " FROM data"
    res = cur.execute(selectQuery)

    # Iterate through the result, convert the data to numpy arrays, apply the function, record in a list with keys
    for i in tqdm(range(numRows)):

        row = res.fetchone()

        #create a new dict for the given collection_index
        index = int(row[indexPosition])
        dataDict[index] = {}

        for i in range(len(colNames)):
            # some tables have blank columns due to code bugs. This skips over them
            # needs to first check if the value is an array b/c truth values don't apply to whole arrays
            if type(row[i]) == np.ndarray or row[i] != None:
                dataDict[index][colNames[i]] = stringConverter(row[i])
            else:
                pass

    # save the dataDict as a pickle. We checked if the file exists earlier, so this operation is safe
    with open(pickleFile, 'wb') as f:
        pickle.dump(dataDict, f)

    con.close()
    f.close()

    return dataDict

def multiSqliteToPickle(files : list):
    '''
    Converts a list of sqlite files to pickled dicts

    Args:
        files (list) : a list of file paths for each sqlite file
    Returns:
        None : pickles are saved but not returned
    '''
    for file in files:
        print("\nConverting " + file + "\n")
        sqliteToPickle(file)

def dirSqliteToPickle(dirName : str):
    '''
    Converts all sqlite files in a directory to pickled dicts

    Args:
        dirName (string) : path to the folder with sqlite files.
    Returns:
        None : pickles are saved but not returned
    '''

    fileNames = listFilesInDirectory(dirName, '.sqlite3')

    multiSqliteToPickle(fileNames)

def listFilesInDirectory(dirName : str, ext = '.pickle'):
    '''
    Lists all files in a directory with a specified extension. Useful for gathering all data files in a folder

    Args:
        dirName (string) : path to the folder
        ext (string) : file extension to look for, in format '.extension'. Default is '.pickle'
    Returns:
         list of strings : list of full file paths for each file in dirName with the specified extension
    '''

    files = os.listdir(dirName)
    fileNames = []
    for file in files:
        if file.endswith(ext):
            fileNames.append(os.path.join(dirName, file))

    return fileNames

# todo: update description - no longer just for strings!
# Helper function that takes in a string returned from a table lookup and attempts to convert it to the appropriate data type
# Only handles floats and lists right now. If it isn't recognized, it returns the unchanged string
def stringConverter(string):

    # first check if its an ndarray and return
    if type(string) == np.ndarray:
        return string

    # Next attempt a float
    try:
        data = float(string)
        return data
    except ValueError or TypeError:
        # it isn't a float, so try to convert a list
        # lists are deprecated but kept in for backward compatibility
        try:
            data = stringListToArray(string)
            return data
        # it isn't a list either. return the string unchanged
        except json.decoder.JSONDecodeError:
            return string

# Helper function to convert a 'stringified' list ('[1.1, 3.2, 4.3]') into a numpy array ([1.1, 3.2, 4.3])
# Inputs the string, outputs the list
# Used to convert sql-saved lists into numpy arrays
# TODO: in the future, saving and loading can be made smarter - save the data as binary and directly load it into an array
def stringListToArray(strList : str):

    # For some reason it is faster to use json.loads rather than np.fromstring
    return np.array(json.loads(strList))

# Saves a dataDict as a pickle. If the 'fileName' key is not informed, a warning message is printed
def savePickle(dataDict : dict):

    if 'fileName' not in dataDict.keys():
        print('savePickle Error: \'fileName\' is not a key in input dict, pickle cannot be saved. Manually add dataDict[\'fileName\'] = fileName and retry saving.'
              'In the future, using loadPickle() or sqliteToPickle() ensures the dataDict is properly formatted')
        return -1

    else:
        fileName = dataDict['fileName']
        with open(fileName, 'wb') as f:
            pickle.dump(dataDict, f)
        f.close()
        return 0

# Load a pickle specified in a filename
# This will also do some basic error checking:
#   Makes sure the pickle is a dict
#   Checks if the 'fileName' key exists
#   Checks if the 'fileName' value matches the input fileName.
#       If either of the last two are not true, the 'fileName' key is updated
# NOTE: the first warning message will be thrown for loading DataCubes. This should be fine
def loadPickle(fileName : str):

    with open(fileName, 'rb') as f:
        dataDict = pickle.load(f)

    pickleType = type(dataDict)

    if pickleType != dict:
        print('loadPickle Warning: loading ' + fileName + ' does not result in a dict. Data manipulation functions and scripts will likely fail.')

    # handle problems with dataDict fileName field
    elif pickleType == dict and 'fileName' not in dataDict.keys():
        print('loadPickle Warning: \'fileName\' not in list of dataDict keys. Updated dataDict[\'fileName\'] = ' + fileName)
        dataDict['fileName'] = fileName
        savePickle(dataDict)

    elif pickleType == dict and dataDict['fileName'] != fileName:
        print('loadPickle Warning: dataDict[\'fileName\'] does not match input fileName. Value of dataDict key has been updated to match new file location.')
        dataDict['fileName'] = fileName
        savePickle(dataDict)

    f.close()
    return dataDict

# apply function to key
# takes a dataDict, a function, the key to store the result in, a list of keys to use as the function arguments, and a list of additional arguments if needed
# NOTE: if dataDict[collection_index][resKey] already exists, it will be overwritten
def applyFunctionToData(dataDict : dict, func, resKey, dataKeys, *funcArgs):

    # check if dataKeys is a list. If it isn't convert it to one
    dataKeys = [dataKeys] if not isinstance(dataKeys, list) else dataKeys

    # Iterate through the keys (coordinates) in the dataDict
    for key in dataDict:

        # check that the key is a collection_index (an int), not 'fileName' or 'parameters"
        if type(key) == int:

            # Gather the data from dataKeys into a list to use as input to func
            funcInputs = [dataDict[key][dataKey] for dataKey in dataKeys]

            dataDict[key][resKey] = func(*funcInputs, *funcArgs)

        else:
            pass

    # Repickle the data
    savePickle(dataDict)

    return dataDict

# Applies multiple functions to a data set. This can be faster than calling applyFunctionToData multiple times because data only needs to be loaded once
# takes a dataDict as well as an input called the funcDict, which is a list of dicts with information on the funcs to be applied
#   funcDictList = [{'func': funcName, 'dataKeys' : ['list of data keys to input'], 'resKey' : 'name of key to store func result', 'funcArgs' : [optional key with additional arguments in order]},...]
#   As an example here's a funcDictList to apply absoluteSum and staltaFirstBreak:
#   [{'func' : pj.absoluteSum, 'dataKeys' : ['voltage'], 'resKey' : 'absSum'}, {'func': pj.staltaFirstBreak, 'dataKeys' : ['voltage', 'time'], 'resKey' : 'stalta_5_30_0d75', 'funcArgs' : [5,30,0.75]}]
# Applies the functions to the data and then returns the new dataDict
def applyFunctionsToData(dataDict : dict, funcDictList : list):

    # iterate through keys in datadict
    for key in dataDict:

        # check that the key is a data set, not parameters or fileName
        if type(key) == int:

            # iterate through functions in funcDictList
            for funcDict in funcDictList:

                # format the dataKeys to an iterable input
                funcInputs = [dataDict[key][dataKey] for dataKey in funcDict['dataKeys']]

                # calculate the value of func. Split depending on whether additional inputs are needed
                if 'funcArgs' in funcDict.keys() and 'funcKwargs' in funcDict.keys():
                    dataDict[key][funcDict['resKey']] = funcDict['func'](*funcInputs, *funcDict['funcArgs'], **funcDict['funcKwargs'])

                elif 'funcArgs' in funcDict.keys():
                    dataDict[key][funcDict['resKey']] = funcDict['func'](*funcInputs, *funcDict['funcArgs'])

                elif 'funcKwargs' in funcDict.keys():
                    dataDict[key][funcDict['resKey']] = funcDict['func'](*funcInputs, **funcDict['funcKwargs'])

                else:
                    dataDict[key][funcDict['resKey']] = funcDict['func'](*funcInputs)

    #repickle data
    savePickle(dataDict)

    return dataDict

# Apply a function to a list of files
def applyFunctionToPickles(fileNames : list,  func : Callable, resKey, dataKeys, *funcArgs):

    for i in tqdm(range(len(fileNames))):

        file = fileNames[i]
        dataDict = loadPickle(file)
        applyFunctionToData(dataDict, func, resKey, dataKeys, *funcArgs)

# same as above, but uses applyFunctionsToData and takes a funcDictList as input
def applyFunctionsToPickles(fileNames : list, funcDictList : list):
    ''' funcDictList = [{'func': funcName, 'dataKeys' : ['list of data keys to input'], 'resKey' : 'name of key to store func result', 'funcArgs' : [optional key with additional arguments in order]},...]
    '''
    for i in tqdm(range(len(fileNames))):

        file = fileNames[i]
        dataDict = loadPickle(file)
        applyFunctionsToData(dataDict, funcDictList)

# Apply a function to all of the .pickles in a directory
def applyFunctionToDir(dirName : str, func : Callable, resKey, dataKeys, *funcArgs):

    fileNames = listFilesInDirectory(dirName)

    applyFunctionToPickles(fileNames, func, resKey, dataKeys, *funcArgs)

# same as above, but for multiple functions using the funcDictList format
def applyFunctionsToDir(dirName : str, funcDictList : list):
    '''funcDictList = [{'func': funcName, 'dataKeys' : ['list of data keys to input'], 'resKey' : 'name of key to store func result', 'funcArgs' : [optional key with additional arguments in order]},...]
    '''
    fileNames = listFilesInDirectory(dirName)

    applyFunctionsToPickles(fileNames, funcDictList)

def compareDatsAtKey(datList : list, titles : list, expIndex : int, xKey : str, yKeys : list):
    '''
    Generates side-by-side plots from lists at the same experiment index

    Args:
        datList : list of data pickles
        titles: strings to name the data pickles
        expIndex : experiment index to plot
        xKey : data key for the x-axis of all plots
        yKeys : list of ykeys used for each plot
    Returns:
        None. Shows a len(datList) x len(yKeys) panelled plot
    '''
    fig, ax = plt.subplots(len(yKeys), len(datList))
    for i in range(len(datList)):
        for j in range(len(yKeys)):
            ax[j, i].plot(datList[i][expIndex][xKey], datList[i][expIndex][yKeys[j]])
            ax[j, i].set_ylabel(yKeys[j])
    for i in range(len(titles)):
        ax[0, i].set_title(titles[i])
    plt.show()

def plotData(dat : dict, index : int, caV : float = 0.):
    '''
    Generates a simple four panelled plot of experimental data at the specified experiment index.
    Plots Detector A and B on top row, applied voltage and measured current on the bottom row

    Args:
        dat : data dict
        index : experimentIndex to plot
        caV : optional, potential applied from the potentiostat via the chronoamperometry experiment
    Returns:
        None. Shows plot
    '''

    exp = dat[index]
    t = exp['time']
    da = exp['detector0']
    db = exp['detector1']
    c = exp['potentiostatCurrent']
    awgt = exp['awgTime']
    awg = exp['awg']
    v = interpolateVoltageProfile(awg, awgt, t[-1], t, caV)

    fig, ax = plt.subplots(2, 2)
    ax[0,0].plot(t, da)
    ax[0,0].set_title('Detector A (Unfiltered)')
    ax[0,1].plot(t, db)
    ax[0,1].set_title('Detector B (695 nm Longpass)')
    ax[1,0].plot(t, v)
    ax[1,0].set_title('Applied Voltage (V vs Ag/Ag+)')
    ax[1,1].plot(t, c)
    ax[1,1].set_title('Current (A)')
    plt.show()

def plot2by2(dat : dict, index : int, xKeys : tuple, yKeys : tuple, titles : tuple):
    '''
    Generates a simple 2 by 2 panelled plot of experimental data at the specified experiment index using user input keys

    Args:
        dat : data dict
        index : experimentIndex to plot
        xKeys : tuple of keys for x-axes. Must be length 4
        yKeys : tuple of keys for y-axes. Must be length 4
        titles : tuple of titles for plots. Must be length 4
    Returns:
        None. Shows plot
    '''
    if len(xKeys) != 4 or len(xKeys) != len(yKeys) or len(xKeys) != len(titles):
        raise ValueError('plot2by2: length of xKeys, yKeys, and titles must be equal to 4')

    exp = dat[index]

    fig, ax = plt.subplots(2, 2)
    for i in range(4):
        # we're getting needlessly fancy with the bitwise operators to index, but its fun
        ax[i >> 1, i & 1].plot(exp[xKeys[i]], exp[yKeys[i]])
        ax[i >> 1, i & 1].set_title(titles[i])

    plt.show()

################################################################################
################### Data Processing Functions ###################################
##################################################################################

# This section contains functions that can be used with applyFunctionToData() for processing experimental data
#   e.g. filtering, interpolating, and analyzing

def generateVoltageProfile(dat, vtFunc : callable):
    '''
    NOTE: DEPRECATED. USE INTERPOLATE VOLTAGEPROFILE, IT IS MORE ROBUST TO DIFFERENT INPUTS
    NOTE: THIS DIRECTLY EVALUATES STRINGS FROM DATA INPUTS AND IS UNSAFE IF YOU DO NOT TRUST YOUR DATA SOURCE
    Currently only handles a single chronoamperometry voltage

    Generates the applied voltage at the same time steps as measured on the oscilloscope. Note this is an interpolation
    does not precisely match with the AWG timing
    This combines information from the input vtFunc parameters and the chronoamperometry experiment

    Args:
        dat (dict) : experimental data dict imported from a pickle
        vtFunc (func) : function used to generate the awg input
    Returns:
        array, array : An array of time values and an array of voltages applied from the potentiostat + awg
                        Both are sampled at the oscilloscope frequency and cutoff by the end of the chronoamperometry experiment


    '''
    # gather vtfunc args and kwargs.Requires direct eval of strings - TRUST YOUR DATA BEFORE USING
    vtArgs = eval(dat['vtFuncArgs'])
    vtKwargs = eval(dat['vtFuncKwargs'])

    # gather chronoamperometry and other measurement parameters
    tStop = dat['vStepTime'][-1]
    caVoltage = dat['vStep'][0] # todo: this only handles a single voltage. Need to upgrade to make a piecewise function

    # determine number of time steps from time array and tStop value (tStop does not need to equal experimentTime)
    time = dat['time']
    boundedTime = time[np.nonzero(time <= tStop)]
    samples = len(boundedTime)

    # evaluate the awg applied voltage using vtFunc
    xVals = np.linspace(0, tStop, samples)
    awgVoltages = vtFunc(xVals, *vtArgs, **vtKwargs)

    return boundedTime, awgVoltages + caVoltage

def interpolateVoltageProfile(awg, awgTime, awgDuration, expTime, caVoltage, finalVoltage = None):
    '''
    Takes the awg profile and time data, as well as the time data to interpolate to (usually the experimental time),
    and interpolates the awg to all values in the target time array, assuming the awg is periodic

    Args:
        awg (array) : array of voltages applied by the awg, in V
        awgTime (array) : array of times for each voltage in awg. len(awg) must equal len(awgTime). Assumed to be in increasing order
        expTime (array) : array of times to interpolate the applied voltage. Assumed to be in the same time unit as
            awgTime and in increasing order. Also assumes expTime[0] >= awgTime[0]
        awgDuration (float) : length of time the AWG is on. When the AWG is off, finalVoltage is used as a constant
        caVoltage (float) : constant voltage applied by the potentiostat in the Chronoamperometry experiment that runs simultaneous
            Assumes the CA experiment was applied for the length of expTime
        finalVoltage (float) : final constant voltage that is applied after awgDuration. If not specified, caVoltage is used
    Returns:
        array : array of applied voltages from the potentiostat and AWG, at the time steps specified by expTime
    '''
    # resolve finalVoltage input
    if finalVoltage == None:
        fv = caVoltage
    else:
        fv = finalVoltage

    # grab the portion of expTime that occurs before awg turns off
    awgOnTime = expTime[expTime <= awgDuration]

    # generate the constant voltage profile that occurs after awg turns off
    awgOffLen = len(expTime) - len(awgOnTime)
    awgOff = np.full(awgOffLen, fv)

    # calculate period of the awg signal
    awgPeriod = awgTime[-1] - awgTime[0]

    # to make the interpolation periodic, take the modulus of the time data with the awg period, then use the
    #   periodic x-axis interpolation
    modOnTime = awgOnTime % awgPeriod

    awgOn = np.interp(modOnTime, awgTime, awg, period = awgPeriod) + caVoltage

    return np.concatenate((awgOn, awgOff))

def movingAvgFilter(arr, windowSize : int = 10):
    '''
    Returns the moving average along the array with the specified window size. Window is right-indexed: value at index i
    is the moving average of the i-windowSize:i elements

    Args:
        arr (array) : array to apply moving average
        windowSize (int) : size of the moving average window
    Returns:
        array : result of filter. Return array has the same length as input array. First windowSize - 1 elements will be
            the average of < windowSize elements
    '''
    return bn.move_mean(arr, windowSize, min_count = 1)

def binAvgReduction(arr, windowSize : int = 10):
    '''
    Reduces the data by returning the average of values within a bin determined by windowSize:
    [average(arr[0:windowSize]), average(arr[windowSize+1:2 * windowSize]),...]

    Args:
        arr (array) : array to apply reduction to
        windowSize (int) : size of bins for averaging
    Returns:
        array : binned data. Array length is floor(len(arr) / windowSize)
    '''
    movingAvg = movingAvgFilter(arr, windowSize)
    return movingAvg[windowSize::windowSize] # starting index is windowSize since moving average is right-indexed (see movingAvgFilter documentation)

def photonCounter(arr, threshold = 100):
    '''
    Attempts to convert raw voltage data to photon counts.

    Args:
        arr (array) : raw voltage array, assumed to be in mV
        threshold (float or int) : threshold value for a single photon detection (in mV) on the SiPM at the measured gain setting
    Returns:
        array : input array discretized to a floored photon count
    '''
    return np.fix(arr / threshold) # fix is a floor towards zero. prevents slightly negative values going to -1

def dehankel(m, d, s):
    '''
    'undo' the Hankel matrix arrangement performed by pydmd.utils.pseudo_hankel_matrix
    Used to extract values of snapshots that were Hankel-ized before DMD

    :param m: matrix to be de-Hankelized
    :param d: d parameter (number of snapshots generated) given to the function that Hankel-ized the matrix
    :param s: number snapshots to turn the matrix back into
        m.shape = r, c
        r = d * s
    :return: the de-Hankelized matrix
        shape = (s,
    '''
    # todo: this needs lots of testing of m.shape vs d and s if this ends up being useful

    # gather the first s rows up to the end. these will be used as-is
    dehankel = m[:s, :-1]

    # next iterate through each set of s rows, adding the final column to the output matrix
    # first calculate the number of sets of rows to iterate through based on the s parameter and shape of the matrix
    rowSets = math.floor(m.shape[0] / s)
    for i in range(rowSets):
        rowEnd = m[i*s:(i*s)+s,-1]
        dehankel = np.concatenate((dehankel, rowEnd[:,np.newaxis]), axis = 1)

    return dehankel

# creates a butterworth low pass filter, adapted from UT package
# takes the time as argument, with additional arguments for the transducer frequency (in MHz, default 2.25), filter order (default 5),
#   and fraction of the transducer frequency to create the filter critical frequency (default 0.2 e.g. 5 MHz transducer -> 1 MHz filter freq)
def createButterFilter(time, freq = 2.25, order = 5):

    sampleRate = time[1] - time[0] # convert ns to s
    sampleFreq = 1 / sampleRate

    # N = order of filter. Higher number = slower calculation but steeper cutoff
    # Wn = critical frequency (gain drops by -3 dB vs passband)
    # btype = lowpass
    # analog = False (this is a digital signal)
    # fs = sampling rate (500,000,000 Hz for 2 ns step size)
    return scipy.signal.butter(order, freq, btype = 'lowpass', analog = False, fs = sampleFreq, output = 'sos')

