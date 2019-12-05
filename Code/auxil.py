import os
import numpy
import pandas
import numbers
import re
import datetime
import json
import io
from config import CONFIG

class aux(object):
    o = 'n'

def create_default_Name(extenstr):

    pathstr=CONFIG['default_save_path']
    symb=pathstr[-1]
    if((symb!='/')&(symb!='\\')):
        pathstr=pathstr+'/'
    now = datetime.datetime.now()
    fstr = str(now)
    fstr = fstr.replace(' ', '_')
    fstr = fstr.replace(':', '_')
    filenamestr= pathstr+fstr + '.' +  extenstr

    return filenamestr
def name_exist(file_name):
    #import os.path
    if os.path.exists(file_name):
        return True
    else:
        return False

def readjsonfile(jsonnamestr):
    dataFlag=False
    if ( not os.path.exists(jsonnamestr)):
        exceptionstr = ' File ' + jsonnamestr + ' does not exists'
        raise ValueError(exceptionstr)
        return

    with io.open(jsonnamestr,"r", encoding='utf-8') as jf:
        text=jf.read()

    '''
    ll=len(needlist)
    for i in range(0,ll):
        fstr=needlist[i]
        if not (fstr in text):
            exceptionstr = 'Invalid information in input file '
            raise ValueError(exceptionstr)
            return
    '''
    try:
        data = json.loads(text)
        dataFlag=True
    except:
        dataFlag=False

    if( not dataFlag):
        exceptionstr = ' input file is not json-type'
        raise ValueError(exceptionstr)
        return

    return data
def check_algorithmname(algnamestr):
    n=0
    valid_names_list=['logisticRegression','kNN',]
    n=valid_names_list.count(algnamestr)
    if(n>0):
        rezult=True
    else:
        rezult=False

    return rezult


def checkParams(data,paramslist,isneedfile):
    length=len(paramslist)
    okstr='OK'
   # valueslist=list()
    path=CONFIG['default_save_path']
    for i in range(0,length):
        keystr=paramslist[i]
        key=keystr in data
        if(not key):
            errstr='parameter '+keystr+' is absent'
            return [False,errstr,i]#,valueslist]
        else:
            #value=data[keystr]
            #valueslist.append(value)
            if(isneedfile[i]):
               filenamestr=path+data[keystr]
               if(not name_exist(filenamestr)):
                   errstr='File '+path+filenamestr + ' does not exists'
                   return [False, errstr, i]#,valueslist]

    return [True,okstr,length]#,valueslist]

def toQuantitative(input,decComma=False):
    # replace strings with numbers
    result = input.map(toNumber,{"decComma":decComma})
    return result

def toNumber(x,decComma=False):
    # If x is a string-written number, converts it  to numeric
    t = type(x)
    if t is str:
        if (decComma):
            y = re.sub(",", ".", x)
        else:
            y = re.sub(",", "", x)
        try:
            y = numpy.float(y)
        except BaseException:
            y = numpy.nan
    elif issubclass(t, numbers.Number):
        y = x
    else:
        y = numpy.nan
    return y