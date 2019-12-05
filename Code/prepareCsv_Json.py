import json
import io

from sklearn import datasets


def prepareCsv_Json():
    iris = datasets.load_iris()
    X = iris.data
    y=iris.target
    ly=len(y)
    a00=X[0,0]
    csvname='e://jsnt//tmp//iris.csv'
    headerstr='IrisP1,IrisP2,IrisP3,IrisP4,class\n'
    with io.open( csvname,"w", encoding='utf-8') as cf:
        cf.write(headerstr)
        for i in range (0,ly):
             a0i = X[i, 0]
             a1i = X[i, 1]
             a2i = X[i, 2]
             a3i = X[i, 3]
             y0i=y[i]


             rowstr=str(a0i)+'  ,  '+str(a1i)+' , '+str(a2i)+' ,  '+str(a3i)+'  ,  '+str(y0i)+' \n '
             cf.write(rowstr)



    '''
    colon1dict={'name':'Колон3',"type": "numeric", "uniqueCount": 0, "uniqueValues": []}
    colon2dict = {'name': 'Колон1', "type": "numeric", "uniqueCount": 0, "uniqueValues": []}
    colon3dict = {'name': 'Колон2', "type": "numeric", "uniqueCount": 0, "uniqueValues": []}
    coloncategdict = {'name': 'категКолон', "type": "categorical", "uniqueCount": 5, "uniqueValues": ['a0','b1','c2','d3','e4']}
    colon4dict = {'name': 'Целев', "type": "categorical", "uniqueCount": 3, "uniqueValues": [0,1,2]}
    colonbindict={'name': 'Целев', "type": "categorical", "uniqueCount": 3, "uniqueValues": [0,1,2]}
    colonbindict={'name': 'бинКолонка', "type": "binary", "uniqueCount": 2, "uniqueValues": ['m','f']}

    descriptlist=[colon1dict,colon2dict,colonbindict,coloncategdict,colon3dict,colon4dict]

    jsondict={ 'rowIdentifier':'nId','selectedColumns': ['Колон3', 'Колон1', 'Колон2','бинКолонка','категКолон'],"target": 'Целев',"description":descriptlist}

    jsondata=json.dumps(jsondict)
    jsonname = 'e://jsnt//tmp//j_iris.txt'
    with io.open( jsonname,"w", encoding='utf-8') as f:
        f.write(jsondata)
    '''
    return
'''
def prepare_trainJson(trainname):

    {
        "method": "trainModel",
        "parameters": {
            "algorithm": [
                {
                    "algorithmName": "logistisRegression",
                    "algorithmParameters": {
                        "fit_intercept": false,
                        "solver": "lbfgs"
                                  "max_iter": "10'
    },
    "outfile": "123.bin"
    },
    {
        "algorithmName": "kNN",
        "algorithmParameters": {
        },
        "outfile": "123.bin"
    }
    ],
    "inputData": "tmp/data/file1.csv",
    "dataScheme": "tmp/scheme/scheme1.csv",
    "progressFile": "progress_123.txt"
    },
    "logFile": "123.log"
    }

    paramdict1={"fit_intercept": False,"solver": "lbfgs","max_iter": "10'}
    algdict1={"algorithmName": "logistisRegression",}
'''

if __name__ == "__main__":
        prepareCsv_Json()
