from modelBuildFourCat import modelBuild
from readTextFile import readTextFile
from comb2 import comb
import multiprocessing as mp
import time
import numpy as np
import os

import csv

kernelWindowSizes = [3,5,7]
numKernels = [16,24,32]
numKernelLayers = [1,2,3]
depthMLPNetwork = [1,2,3]
numEpochs = [10, 25, 50]
modelCounter = 0

fn = './resultsAssignment4_Trial5_Four_Classes_With_Comb2.txt'

def listener(q):
    '''listens for messages on the q, writes to file. '''
    with open(fn, 'a') as f:
        while 1:
            m = q.get()
            if m == 'kill':
                f.write('killed')
                break
            f.write(str(m) + '\n')
            f.flush()

# def comb(L):
#     possibleSizes = []
#     for i in range(3):
#         for j in range(3):
#             for k in range(3):
#                 if (i!=j and j!=k and i!=k):
#                     possibleSizes.append([L[i], L[j], L[k]])
#     return possibleSizes

def main():
    
    modelCounter = 0
    manager = mp.Manager()
    q = manager.Queue()
    pool = mp.Pool(2)

    #put listener to work first
    watcher = pool.apply_async(listener, (q,))

    #fire off workers
    jobs = []
    data = []

    kernelSizes = comb(kernelWindowSizes) #17 total combinations

    for kernelSize in kernelSizes:
        for numKernelLayer in numKernelLayers:
            for numKernel in numKernels:
                for depthMLP in depthMLPNetwork:
                    for numEpoch in numEpochs:
                        #Before submitting a job, lets check the text file to see if the model has already been run.

                        currentModelParams = str(numKernel) + " " + str(numKernelLayer) + " " + str(kernelSize) + " " + str(depthMLP) + " " + str(numEpoch)
                        currentModelParamsStripped = currentModelParams.replace(",","")[0:17].rstrip()
                        inTextFile = readTextFile(currentModelParamsStripped,fn)
                        if not inTextFile:
                            data = [kernelSize, numKernelLayer, numKernel, depthMLP, numEpoch]
                            job = pool.apply_async(modelBuild, (data, q))
                            jobs.append(job)
                        else:
                            print(currentModelParamsStripped + ": is already in text file. Skipping model...")
                            continue
                        #data = [kernelSize, numKernelLayer, numKernel, depthMLP, numEpoch]
                        #job = pool.apply_async(modelBuild, (data, q))
                        #jobs.append(job)


    # collect results from the workers through the pool result queue
    for job in jobs: 
        job.get()

        #DISPLAY COMPLETION PERCENTAGE
        modelCounter = modelCounter + 1
        progressPercent = (modelCounter / 1377) * 100
        print("Models complete: " + str(modelCounter) + ", Percent Complete: " + str(progressPercent) + "%")
        #print(job.get())

    #now we are done, kill the listener
    q.put('kill')
    pool.close()
    pool.join()

if __name__ == '__main__':
    main()