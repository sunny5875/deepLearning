import random
import numpy as np 
import math

m = 10000 # sample training 수
limit = 0.5 # output limit
alpha = 0.01 # learning rate
k = 5000 # iteration 수

def sigmoid(x):
    return 1 / (1 +np.exp(-x))

def generateSampleMatrix():
    X = np.zeros((2,m), float)   
    Y = np.zeros([1,m], int)

    for i in range(m):
        x_train = np.array([[random.uniform(-10,10), random.uniform(-10,10)]])
        X[0,i] =  x_train[0,0]
        X[1,i] =  x_train[0,1]
      
        if x_train[0,0] < -5 or x_train[0,1] > 5:
            Y[0,i] = 1
        else:
            Y[0,i] = 0

    return (X,Y)


def executeSimpleNeuralNetworkUsingVectorization():
    # initialize weight and bias
    w1 = random.uniform(-0.5,0.5)
    w2 = random.uniform(-0.5,0.5)
    w3 = random.uniform(-0.5,0.5)
    w4 = random.uniform(-0.5,0.5)
    w5 = random.uniform(-0.5,0.5)
    w6 = random.uniform(-0.5,0.5)
    b1 = random.uniform(-0.5,0.5)
    b2  = random.uniform(-0.5,0.5)
    b3 = random.uniform(-0.5,0.5)

    W1 = np.array([[w1, w2],[w3,w4]])
    B1 = np.array([[b1],[b2]])

    W2 = np.array([[w5], [w6]])
    B2 = np.array([b3])

    # generate sample
    X, Y  = generateSampleMatrix()

    
    for i in range(k): 
        # J = np.zeros((1,m))
        # hidden layer forward progpatgation
        Z1 = np.matmul(W1.T, X) + B1
        A1 = sigmoid(Z1)
        # output forward propagation
        Z2 = np.matmul(W2.T, A1) + B2 
        A2 = sigmoid(Z2)
        # J = J -( Y @ math.log(A2) + (1-Y) @ math.log(1 - A2))

        #W2 back propagation
        dZ2 = A2 - Y
        dW2 = 1/m * np.matmul(dZ2, A1.T)
        dB2 = 1/m * np.sum(dZ2, axis = 1, keepdims = True)

        #W1 back propagation
        tempdZ1 = np.matmul(W2, dZ2)
        sig = sigmoid(Z1) * (1- sigmoid(Z1))
        dZ1 = tempdZ1 * sig

        dW1 = 1/m * dZ1 @ X.T
        dB1 = 1/m * np.sum(dZ1, axis = 1, keepdims = True)

        W1 = W1 - alpha * dW1
        B1 = B1 - alpha * dB1
        W2 = W2 - alpha * dW2
        B2 = B2 - alpha * dB2

          
        if (i+1) % 500 == 0 or i == 0:
            print(i+1,"th update parameter: ", W1[0,0], W1[0,1],W1[1,0], W1[1,1], W2[0,0], W1[1,0],B1[0,0], B1[1,0],B2[0,0])
            realScore = 0
            for a in range(m):
                if A2[0,a] > limit:
                    if 1 == Y[0,a]:
                        realScore += 1
                else:
                    if 0 == Y[0,a]:
                        realScore += 1
    
            print("accuracy: ",realScore/m * 100)    
     


    realScore = 0
    Z1 = np.dot(W1.T, X) + B1
    A1 = sigmoid(Z1)
    Z2 = np.dot(W2.T, A1) + B2 
    A2 = sigmoid(Z2)
    for i in range(m):
        if A2[0,i] > limit:
            if 1 == Y[0,i]:
                realScore += 1
        else:
             if 0 == Y[0,i]:
                realScore += 1
    print("==========================")
    print("result parameters: ", W1[0,0], W1[0,1],W1[1,0], W1[1,1], W2[0,0], W1[1,0],B1[0,0], B1[1,0],B2[0,0])
    print("final accuracy: ",realScore/m * 100)    


executeSimpleNeuralNetworkUsingVectorization()