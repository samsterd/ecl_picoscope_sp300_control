import data

# Simple analysis/plotting script for filtering data. Not well documented yet
#   For optimal use, comment in/out sections that will be used. This would probably work more naturally as a notebook

dir = 'C://path//to//data//'
file = dir +  'filename.sqlite3'
data.sqliteToPickle(file)
dat = data.loadPickle(file)

caVoltage = 1

funcDictList = [
    {'func': data.interpolateVoltageProfile, 'dataKeys' : ['recoveredAWG', 'awgTime', 'time'], 'resKey':'expAWG', 'funcArgs':[caVoltage]},
    {'func': data.binAvgReduction, 'dataKeys': ['detector0'], 'resKey':'d0_binned_1p18ms', 'funcArgs':[59]},
    {'func': data.binAvgReduction, 'dataKeys': ['detector1'], 'resKey': 'd1_binned_1p18ms', 'funcArgs': [59]},
    {'func': data.binAvgReduction, 'dataKeys': ['potentiostatCurrent'], 'resKey': 'c_binned_1p18ms', 'funcArgs': [59]},
    {'func': data.binAvgReduction, 'dataKeys': ['time'], 'resKey': 't_binned_1p18ms', 'funcArgs': [59]},
    {'func': data.binAvgReduction, 'dataKeys': ['expAWG'], 'resKey': 'v_binned_1p18ms', 'funcArgs': [59]},
]
#
comparisonKey = 0
data.compareDatsAtKey([dat],
                 ["Example data"],
                 comparisonKey,
                 't_binned_1p18ms',
                 ['v_binned_1p18ms','c_binned_1p18ms','d0_binned_1p18ms','d1_binned_1p18ms'])
