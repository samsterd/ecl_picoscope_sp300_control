# functions for saving / loading data

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
            query = 'UPDATE ' + table + ' SET ' + keyString + ' = ' + qMarks + ' WHERE ' + keyCol + ' = ?'
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