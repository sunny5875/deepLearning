import random
import numpy as np 
import math

m = 10000 # sample training 수
limit = 0.5 # output limit
alpha = 0.01 # learning rate
k = 5000 # iteration 수

def sigmoid(x):
    return 1 / (1 +np.exp(-x))

def cross_entropy_loss(y_hat, y):
    return -(y*np.log10(y_hat + 1e-10) + (1-y)*np.log10(1-y_hat + 1e-10))

def generateSampleMatrix():
    X = np.zeros((2,m), float)   
    Y = np.zeros((1,m), int)

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
    w1 = random.uniform(0,1)
    w2 = random.uniform(0,1)
    w3 = random.uniform(0,1)
    w4 = random.uniform(0,1)
    w5 = random.uniform(0,1)
    w6 = random.uniform(0,1)
    b1 = random.uniform(0,1)
    b2  = random.uniform(0,1)
    b3 = random.uniform(0,1)

    W1 = np.array([[w1, w2], [w3, w4]]) #(2,2)
    B1 = np.array([[b1], [b2]]).reshape(2,1) #(2,1)
    W2 = np.array([[w5, w6]]).reshape(1,2) #(1,2)
    B2 = np.array([b3]) #(1,1)

    # generate sample
    X, Y = generateSampleMatrix()

    
    for i in range(k):
        J = np.zeros((1,m),float)

        # hidden layer forward progpatgation
        Z1 = np.matmul(W1.T, X) + B1 #(2,m)
        A1 = sigmoid(Z1) #(2,m)
        # output forward propagation
        Z2 = np.matmul(W2, A1) + B2 #(1,m)
        A2 = sigmoid(Z2) #(1,m)
        J += cross_entropy_loss(A2, Y)

        #W2 back propagation
        dZ2 = A2 - Y #(1,m)
        dW2 = float(1.0/m) * np.matmul(dZ2, A1.T) #(2,1)
        dB2 = float(1.0/m) * np.sum(dZ2, axis = 1, keepdims = True)#(1,1)

        #W1 back propagation
        dZ1 = np.matmul(W2.T, dZ2) * sigmoid(Z1) * (1-sigmoid(Z1))

        dW1 = float(1.0/m) * dZ1 @ X.T #(2,2)
        dB1 = float(1.0/m) * np.sum(dZ1, axis = 1, keepdims = True)#(2,1)


        W1 = W1 - alpha * dW1
        B1 = B1 - alpha * dB1
        W2 = W2 - alpha * dW2
        B2 = B2 - alpha * dB2


          
        if (i+1) % 500 == 0 or i == 0:
            # print(i+1,"th update parameter: ", W1[0,0], W1[0,1],W1[1,0], W1[1,1], W2[0,0], W1[1,0],B1[0,0], B1[1,0],B2[0,0])
            # realScore = 0
            # for a in range(m):
            #     if A2[0,a] > limit:
            #         if 1 == Y[0,a]:
            #             realScore += 1
            #     else:
            #         if 0 == Y[0,a]:
            #             realScore += 1
            # print("W1 ", W1.shape, dW1.shape)
            # print("W2 ", W2.shape, dW2.shape)
            # print("B1 ", B1.shape, dB1.shape)
            # print("B2 ", B2.shape, dB2.shape)
            # print("accuracy: ",realScore/m * 100)    
            print(i+1,"th cost: ", np.mean(J))
     


    realScore = 0
    Z1 = np.matmul(W1.T, X) + B1 #(2,m)
    A1 = sigmoid(Z1) #(2,m)
    # output forward propagation
    
    Z2 = np.matmul(W2, A1) + B2 #(1,m)
    A2 = sigmoid(Z2) #(1,m)
    J = cross_entropy_loss(A2, Y)
    for i in range(m):
        if A2[0,i] > limit:
            if 1 == Y[0,i]:
                realScore += 1
        else:
             if 0 == Y[0,i]:
                realScore += 1
    print("==========================")
    print("result parameters: ", W1[0,0], W1[0,1],W1[1,0], W1[1,1], W2[0,0], W1[1,0],B1[0,0], B1[1,0],B2[0,0])
    print("final accuracy: ",realScore/m * 100.0)   
    print("final cost: ", np.mean(J)) 


executeSimpleNeuralNetworkUsingVectorization()