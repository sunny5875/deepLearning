import random
import numpy as np 
import math

m = 5000 # sample training 수
limit = 0.5 # output limit
alpha = 0.01 # learning rate
k = 5000 # iteration 수

def sigmoid(x):
    return 1 / (1 +np.exp(-x))

def generateSample():
    x1_train = []
    x2_train = []
    y_train = []
    count = 0
    for i in range(m):
        x1_train.append(random.uniform(-10,10))
        x2_train.append(random.uniform(-10,10))

        if x1_train[-1] + x1_train[-1] > 0:
            y_train.append(1)
            count += 1
        else:
            y_train.append(0)

    print("training output: ", count)
    return (x1_train, x2_train, y_train)


def executeBinaryClassification():
    # unknown parameter
    w1 = random.uniform(0,1)
    w2 = random.uniform(0,1)
    b = random.uniform(0,1)

    print("initial paramter: ",w1,w2,b)

    # generate sample
    x1_train, x2_train, y_train  = generateSample()



    for i in range(k): 
        z = []
        a = []
        J = np.float64(0.0)
        dw1 = np.float64(0.0)
        dw2 = np.float64(0.0)
        db = np.float64(0.0)
        for j in range(m):
            z.append( w1 * x1_train[j] + w2* x2_train[j]+ b )
            a.append(sigmoid(z[-1]))
            J += - (y_train[j] * math.log(a[-1]) + (1-y_train[j]) * math.log(1-a[-1]))

            dz = a[-1] - y_train[j]
            dw1 += x1_train[j] * dz
            dw1 += x2_train[j] * dz
            db  += dz
           
        J /= np.float64(m)
        dw1 /= np.float64(m)
        dw2 /= np.float64(m)
        db /= np.float64(m)
        w1 -= alpha * dw1
        w2 -= alpha * dw2
        b -= alpha * b
     

    print("result parameter: ", w1 , w2, b)


    output = 0
    for i in range(m):
        resultZ = w1 * x1_train[i] + w2* x2_train[i]+ b 
        finalResult = sigmoid(resultZ)
        if  finalResult > 0.5:
            output += 1
    print("output: ", output)

executeBinaryClassification()