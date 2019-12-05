# coding: utf8
#import numpy
#import sys
#import cv2
#import skimage

+7-42\
    -5

+7-42

object

True

True and False

Exception

None

pass

...

"""
quotes
"""

from typing import Callable

from numpy import *
import os, sys
os
import argparse
import json
import io
import warnings
from importlib import import_module
import datetime
import inspect
import psutil
# from .auxil import aux
import auxil

from auxil import name_exist
from auxil import create_default_Name
from auxil import readjsonfile

# from pack.auxil import packet

# from ..pack.auxil import packet

#from traindata import data_for_trainModel
#from evaluationdata import data_for_evalModel

# from pack import tools

from numpy import matrix

import pack.tools

from pack.takes import takes
import pack.takes

from ..outcode import out

from numpy.matrixlib import matrix

PASSWORD = '123'

# class dt:
#     t = 1

# captain: str

# complex = lambda x: lambda y: y**2

# def trace(f):
#     def wrap(*args, **kwargs):
#         print(f"[TRACE] func: {f.__name__}, args: {args}, kwargs: {kwargs}")
#         return f(*args, **kwargs)
#
#     return wrap
#
# fu = trace(lambda x: x ** 2)
#
# def kvadrat(x):
#     return x**2
#
# fi = trace(kvadrat)

# verno = lambda x: True

# class y:
#     y=lambda c: c-3

# def nol():
#     class zero:
#         t =0
#     return zero

'st'\
'6'

# limb \
#     = lambda x: \
#     x*\
#     5
#
# lamb = lambda x: x+1
#
# lf: Callable[[int], int] = lambda x: x*x
#
# p: int = 5

# class tee: pass

# class tee: y=[
#
# ]

# class t: object
# class a:
#     class b:
#         class c: y=0
# class tee: y=[1,
#               4,
#               ]


def functor():

  
		
    x = lambda y: y+8

    class one:

        h = lambda h: h**2

        def onefu(self):

            on = lambda x: True

            class onecl:
                f = 0

        class two:
            g = 0

    class odin:

        f = 0

    def h():

        hi = lambda x: x-3

        class h:
            h = 0

class outcode(out):

    j = lambda x: x and True

    def metod(self):

        k = lambda x: isinstance(x, list)

        def inner():

            l = lambda x: False

            exit(0)

        return

    class neste:

        m = lambda x: max(x)

        def stop(self):
            print('stop')

        n = 0

    a = 5

    # +7-42\   # NOT SKIP YET
    # -5

    # NameError

    # object

    # True

    # dict()  # Call

    # "string"

    # "          \
    #            \
    # "
    #
    # """
    # 3
    # 5
    # """

    # 3+5

    # {"d": 'dicto',
    #  "h": 'map'}

    # True and False

    # Exception

    # None

    # pass

    exit

    ...

    ouch = \
        7

    auch = {

    }

    iuch = [

    ]

    # comme


# class matriza(matrix):
#     ma = []

# class unknown(object):
#     unknown=8

class prefix(pack.takes.takes):
    print('.')

# class au(auxil.aux):
#     y = 5
#
# class packet:
#     t = 9
#
# class ai(packet):
#     g = 0

# class auxi(aux):
#      f = '*'

def fun():

    def one():

        def two():
            exit(0)

        return two

    class d:
        f = 0

    t = lambda x: x+1

    __doc__ = "functions"

class dt(takes):
    """
    Note: no __init__
    """
    '''
    stroka
    '''
    """
    comments
    """
    ...
    # comment
    t = 1
    p = 'a'
    k = \
        'po'
    '''
    str
    '''
    """
    strok
    """

# class dt(pack.takes.takes):   # WORK for from pack.takes import takes
#     t = 1

# class dt(pack.tools.tools):
#     t = 1

# class dt(tools.tools):
#     t = 1

class dat(datetime.date):
    t = 0

def f(a, b=1, c=None, d=[], e={}, g=[1], h={'1', 3}, *args, f=42, **kwargs): 
    pass

def split_string(function):
    def wrapper():
        func = function()
        splitted_string = func.split()
        return splitted_string

    return wrapper
	
def uppercase_decorator(function):
    def wrapper():
        func = function()
        make_uppercase = func.upper()
        return make_uppercase

    return wrapper
	

@split_string
@uppercase_decorator
def say_hi():
    return 'hello there'


def stop():
    sys.exit()

def brackets():
    return (1,)

def value(a):
    return a


def text():
    print('txt')


def skip():
    pass


def out(): 
    return

def tes():
    return 0

def memory_usage_psutil():
    # return the memory usage in MB
    #import psutil
    #process = psutil.Process(os.getpid())
   # mem = process.get_memory_info()[0]# / float(2 ** 20)
    process = psutil.Process(os.getpid())
    #mem = process.get_memory_info_ex()
    mem = process.memory_info_ex()
    memused = float(mem.vms) / 1048576.0
    #mem = psutil.virtual_memory()
    #memused=float(mem.used)/1048576.
    memused = round(memused)
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
        with io.open(outFile, "w", encoding='utf-8')  as lf:
            lf.write(logstr)
            if(userstr!=None):
                lf.write(userstr)
        return


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

        #with open(input_file_name, "rU") as inf:
        try:
                #jdata = json.load(inf)
            jdata=readjsonfile(input_file_name)
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
                        #errstr=str(err)
                        #raise ValueError(errstr)
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

        parser = argparse.ArgumentParser(description='Main command to pass json parameters to selected method.')
        parser.add_argument('-i', '--input', type=str, metavar='inputFilePath', required=True, dest='inputFile',
                        help='File with json for input parameters')
        parser.add_argument('-o', '--output', type=str, metavar='outputFilePath', required=True, dest='outFile',
                        help='File with json to save output parameters')
        args = parser.parse_args()

        inputFilePath = args.inputFile
        outFile = args.outFile

        rezult=exam_json(inputFilePath)

        if(rezult['state']==True):
                rezult['error']=' '

        with warnings.catch_warnings(record=True) as w:
            l=len(w)
            w1=list()
            if l>0:
                rezult['warnings]']=w
            else:
                    rezult['warnings'] = w1

        with io.open(outFile, "w", encoding='utf-8') as outf:
            outf.write(json.dumps(rezult))



       # else:

        #    print('Wrong Command Line ( not exactly two parameters)')

