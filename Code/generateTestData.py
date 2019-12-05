import json
from evaluationdata import data_for_evalModel


if __name__ == "__main__":

#name_exist("123")
        #print 'Number of arguments:', len(sys.argv), 'arguments.'
        #print 'Argument List:', str(sys.argv)

        paramdict={'modelFile':"e:\\jsnt\\123.bin",'predictedColumnName':'new_column','inputData':"e:\\jsnt\\123.csv",'outFile':"e:\\jsnt\\123out.csv",'progressFile':'progress_123.txt'}
        #jsondict={'name':'Jonson','method':'printonscreen','param':'Hello_From_Jonson'}
        #jsondict={'method':'batchPrediction','parameters':paramdict,'logFile':'123.log'}
        #jsondict=data_for_trainModel()
        jsondict=data_for_evalModel()
        innamestr='e:/jsnt/jsontest.txt'
        with open(innamestr, "wU") as f:
               f.write(json.dumps(jsondict))

