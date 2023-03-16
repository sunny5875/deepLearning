import random
import numpy as np 
import math

m = 1000 # sample training 수
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

def generateSampleMatrix():
    X = np.zeros((2,m), float)   
    Y = np.zeros([1,m], int)
    count = 0
    for i in range(m):
        x_train = np.array([[random.uniform(-10,10), random.uniform(-10,10)]])
        X[0,i] =  x_train[0,0]
        X[1,i] =  x_train[0,1]
      
        if x_train[0,0] + x_train[0,1] > 0:
            Y[0,i] = 1
            count += 1
        else:
            Y[0,i] = 0

    print("training output: ", count)
    return (X,Y)


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
            z.append( w1 * x1_train[j] + w2* x2_train[j]+ b)
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

        if (i+1) % 500 == 0 :
            print(i+1,"th update parameter: ", w1 , w2, b)
     

    print("result parameters: ", w1 , w2, b)


    realScore = 0
    for i in range(m):
        resultZ = w1 * x1_train[i] + w2* x2_train[i]+ b 
        finalResult = sigmoid(resultZ)
        if  finalResult > limit:
            if 1 == y_train[i]:
                realScore += 1
        else:
             if 0 == y_train[i]:
                realScore += 1
    print("accuracy: ",realScore/m * 100)

def executeBinaryClassificationUsingVectorization():
    # unknown parameter
    w1 = random.uniform(0,1)
    w2 = random.uniform(0,1)
    b = random.uniform(0,1)
    W = np.array([w1, w2])

    print("initial paramter: ",w1,w2,b)

    # generate sample
    X, Y  = generateSampleMatrix()

    
    for i in range(k): 
        J = np.float64(0.0)
        dw1 = np.float64(0.0)
        dw2 = np.float64(0.0)
        db = np.float64(0.0)

        Z = np.dot(W.T, X) + b
        A = sigmoid(Z)
        dZ = A - Y
        dw = 1/m * X @ dZ.T
        db = 1/m * np.sum(dZ)

        W = W - alpha * dw
        b = b - alpha * db


        if (i+1) % 500 == 0 :
            print(i+1,"th update parameter: ", w1 , w2, b)
     

    print("result parameters: ", w1 , w2, b)


    realScore = 0
    Z = np.dot(W.T, X) + b
    A = sigmoid(Z)
    for i in range(m):
        if A[0,i] > limit:
            if 1 == Y[0,i]:
                realScore += 1
        else:
             if 0 == Y[0,i]:
                realScore += 1
    
    print("accuracy: ",realScore/m * 100)    

print("--- execute naive binary classification ---")
executeBinaryClassification()
print("--- execute vectorization binary classification ---")
executeBinaryClassificationUsingVectorization()