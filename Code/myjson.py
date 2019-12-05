#import numpy
#import sys
#import cv2
#import skimage
import argparse
import os
import json
import sys
import warnings
from importlib import import_module
import datetime
import inspect
import psutil
from auxil import name_exist
from auxil import create_default_Name
from traindata import data_for_trainModel
from evaluationdata import data_for_evalModel


def memory_usage_psutil():
    # return the memory usage in MB
    #import psutil
    #process = psutil.Process(os.getpid())
   # mem = process.get_memory_info()[0]# / float(2 ** 20)
    mem = psutil.virtual_memory()
    memused=float(mem.used)/1048576.
    memused=round(memused)
    return memused



def to_logFile(logname, userstr = None):
        currentTime = str(datetime.datetime.now())
        #currentMem = memory_usage_resource()

        mem=memory_usage_psutil()
        currentMem=str(mem)+ 'MB '

        f=inspect.stack()[2][3]
        if(f=='<module>'):
            currentFunction='argumentProcessing'
        else:
            currentFunction =f
        #frame = inspect.currentframe()
        #currentFunction = 'argument Processing'
        position = 'Before method call'
        logstr = 'At time ' + currentTime + ' running in the function ' + currentFunction + ' at position ' + position + ' with memory used: ' + currentMem
        with open(logname, 'w') as lf:
            lf.write(logstr)
            if(userstr!=None):
                lf.write(logstr)
        return

'''
def name_exist(file_name):
    #import os.path
    if os.path.exists(file_name):
        return True
    else:
        return False
'''


def exam_json(input_file_name):
    rezult=dict.fromkeys(['logFile', 'state','error','response','warnings'])
    #now = datetime.datetime.now()
    #logstr = str(now)
    #logstr = logstr.replace(' ', '_')
    #logstr = logstr.replace(':', '_')
    logstr=create_default_Name('log')
    respstr='{}'
    dataflag=False
    methodflag=False
    #logfile = open(logstr, 'w')
   # rezult=dict.fromkeys(['state','name', 'method', 'param'])
    if(name_exist(input_file_name)):

        with open(input_file_name, "r") as inf:
            try:
                jdata = json.load(inf)
                #rezult['name']=jdata['name']
                rezult['name']=input_file_name
                rezult['response']=respstr
                dataflag=True
            except:
                rezult['state'] = False
                rezult['error'] = 'Input file is not json-type'
                return rezult


        if(dataflag):
            log_exists = 'logFile' in jdata
            if(log_exists):
                lognamestr=jdata['logFile']
            else:
                lognamestr=logstr
            rezult['logFile'] = lognamestr
            key_exists = 'method' in jdata
            if(key_exists):
                methodnamestr=jdata['method']
                rezult['state']=True
                rezult['method']=methodnamestr

                try:
                    mod = import_module(methodnamestr)
                    n=0
                except:
                    rezult['state'] = False
                    rezult['error']='script '+methodnamestr+'.py  does not exists'
                    return rezult
                try:
                    met = getattr(mod, methodnamestr)
                    #stlength = met(jdata)
                    methodflag=True;
                    to_logFile(lognamestr)

                    n = 2

                except :
                    rezult['state'] = False
                    rezult['error'] = 'method ' + methodnamestr + ' does not exists'
                    return rezult

                if(methodflag):
                    try:
                        rezult['response'] = met(jdata)
                    except ValueError as err:
                        emptydict={}
                        rezult['response']=emptydict
                        rezult['state'] = False
                        rezult['error']=err.args[0]
                        return rezult
            else:
                rezult['state']=False
                rezult['error']='no method name'
                return rezult


    else:
        rezult['state'] = False
        #rezult['logFile'] = input_file_name
        filestr = 'File ' + input_file_name + ' does not exists'
        rezult['error'] = filestr




    return rezult


if __name__ == "__main__":

        #name_exist("123")
        #print 'Number of arguments:', len(sys.argv), 'arguments.'
        #print 'Argument List:', str(sys.argv)

        paramdict={'modelFile':"e:\\jsnt\\123.bin",'predictedColumnName':'new_column','inputData':"e:\\jsnt\\123.csv",'outFile':"e:\\jsnt\\123out.csv",'progressFile':'progress_123.txt'}
        #jsondict={'name':'Jonson','method':'printonscreen','param':'Hello_From_Jonson'}
        #jsondict={'method':'batchPrediction','parameters':paramdict,'logFile':'123.log'}
        #jsondict=data_for_trainModel()
        jsondict=data_for_evalModel()

        l = len(sys.argv)
        if(l==5):

            parser = argparse.ArgumentParser(description='Process command line.')
            parser.add_argument('-i',type=str)
            parser.add_argument('-o', type=str)
            args=parser.parse_args()

            innamestr=args.i
            outnamestr=args.o

            with open(innamestr, "w") as f:
                f.write(json.dumps(jsondict))


            rezult=exam_json(innamestr)

            if(rezult['state']==True):
                rezult['error']=' '

            with warnings.catch_warnings(record=True) as w:
                l=len(w)
                w1=list()
                if l>0:
                    rezult['warnings]']=w
                else:
                    rezult['warnings'] = w1


            with open(outnamestr, "w") as outf:
                outf.write(json.dumps(rezult))



        else:

            print('Wrong Command Line ( not exactly two parameters)')

