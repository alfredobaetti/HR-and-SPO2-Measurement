import json
import numpy
import numpy as np
import math
import pickle
from config import SUBJECT


def find_nearest(array,value):  
    
        idx = np.searchsorted(array, value, side="left")
        
        if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
            return array[idx-1], idx
        
        else:
            return array[idx], idx

def getTS():
    with open(f"C:/Users/baett/OneDrive/Desktop/Proyecto final/Dataset proyecto/{SUBJECT}.json") as jsonFile:
        jsonObject = json.load(jsonFile)
        jsonFile.close()

    HR = []
    SPO2 = []
    TShr = []
    TSfr = []
    TShraprox = []
    HRaprox = []
    SPO2aprox = []

    for c in range (len(jsonObject['/FullPackage'])):
        pulserate = jsonObject['/FullPackage'][c]['Value']['pulseRate']
        saturation = jsonObject['/FullPackage'][c]['Value']['o2saturation']
        tsheartrate = jsonObject['/FullPackage'][c]['Timestamp']
        HR.append(pulserate)
        SPO2.append(saturation)
        TShr.append(tsheartrate)

    for c in range (len(jsonObject['/Image'])):
        tsframe = jsonObject['/Image'][c]['Timestamp']
        TSfr.append(tsframe)

    for c in TSfr:
        ts, idx = find_nearest(TShr, c)
        TShraprox.append(ts)
        try:
            HRaprox.append(HR[idx])
            SPO2aprox.append(SPO2[idx])
        except:
            pass
    
    return TSfr, TShraprox, HRaprox, SPO2aprox

