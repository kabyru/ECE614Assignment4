def modelBuild(data, q):
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    from tensorflow.python.keras.datasets import cifar10
    from tensorflow.python.keras.layers.core import Dense
    from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout
    from tensorflow.python.keras.layers import Input
    from tensorflow.python.keras.models import Sequential
    from keras.utils import to_categorical
    from tensorflow.python.keras import optimizers
    from tensorflow.python.keras.models import Model
    from tensorflow.python.keras import backend as K

    import csv

    ##DATA LOADING AND RESHAPNG
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    x_train = x_train.reshape(50000, 32, 32, 3)
    x_test = x_test.reshape(10000, 32, 32, 3)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    x_train = x_train/255
    x_test = x_test/255

    num_classes = 10
    print('y data before: ')
    print(y_train[0:5])

    y_train = to_categorical(y_train, num_classes)
    y_test  = to_categorical(y_test, num_classes)
    print('\ny data after:')
    print(y_train[0:5])

    ##END DATA LOADING AND RESHAPING

    #We need to accept a parameter for Conv. Kernel Sizes, # of Kernels
    #Depth of later MLP Network, and # of epochs.
    kernelSizes = data[0]
    numKernelLayers = data[1]
    numKernels = data[2]
    depthMLPNetwork = data[3]
    numEpochs = data[4]

    #For depthMLPNetwork, we'll just add Dense(16, activation='relu') layers in succession before
    #the Softmax(10) at the end.

    try:
        ##MODEL SETUP
        K.clear_session()
        model = Sequential()
        firstLayerAdded = False
        for i in range(0,numKernelLayers):
            if not firstLayerAdded:
                model.add(Conv2D(((i+1)*numKernels), (kernelSizes[i],kernelSizes[i]), activation='relu', input_shape=(32,32,3,)))
            else:
                model.add(Conv2D(((i+1)*numKernels), (kernelSizes[i],kernelSizes[i]), activation='relu'))
            poolName = "pool_" + str(i+1)
            model.add(MaxPooling2D((2,2), name=poolName))
            firstLayerAdded = True
        model.add(Flatten())
        for j in range(0,depthMLPNetwork):
            model.add(Dense(16, activation='relu'))
        model.add(Dense(10, activation='softmax'))

        model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
        model.summary()
        ##END MODEL SETUP

        ##BEGIN MODEL TRAINING
        training_samples = 50000
        testing_samples  = 10000

        batch_size = 128
        epochs     = numEpochs

        history = model.fit(x_train[:training_samples],
                    y_train[:training_samples],
                    epochs=epochs,
                    batch_size=batch_size,
                    verbose=1,
                    validation_data=(x_test[:testing_samples],y_test[:testing_samples]))
        ##END MODEL TRAINING

        ##SEND RESULTS BACK TO MAIN TO PRINT TO CSV
        res = str(numKernels) + " " + str(numKernelLayers) + " " + str(kernelSizes) + " " + str(depthMLPNetwork) + " " + str(numEpochs) + " " + str(history.history['val_accuracy'][numEpochs-1])
        q.put(res)
        return res
    except Exception:
        import traceback
        print(traceback.format_exc())
        res = str(numKernels) + " " + str(numKernelLayers) + " " + str(kernelSizes) + " " + str(depthMLPNetwork) + " " + str(numEpochs) + " " + "EXCEPTION"
        q.put(res)
        return res

