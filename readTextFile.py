import csv

def readTextFile(compareString,filePath):

    with open(filePath, 'r', newline='') as f:
        reader = csv.reader(f)
        data = list(reader)

    #print(data[0])
    returnedValue = False
    for i in range(0,len(data)):
        output = ""
        for j in range(0,len(data[i])):
            output = output + data[i][j]
        savedHyperparameters = output[0:17].rstrip()
        #print("compareString: " + compareString)
        #print("savedHyperparameters: " + savedHyperparameters)
        #print(str(bool(compareString == savedHyperparameters)))

        if (compareString == savedHyperparameters):
            returnedValue = True
            break
    
    return returnedValue