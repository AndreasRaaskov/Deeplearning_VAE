import os, re, glob, json, sys
import pandas as pd
from collections import defaultdict

def findEDF(DataDir):
    # find all .edf files
    pathRootInt = len(DataDir.split('\\'))
    paths = ['\\'.join(fDir.split('\\')[pathRootInt:]) for fDir in
                   glob.glob(DataDir + "/**/*.edf", recursive=True)]
    # construct defaultDict for data setting
    edfDefDict = defaultdict(dict)
    for n,path in enumerate(paths):
        edfDefDict[n]={"path": path.split(".")[0]}
    return edfDefDict