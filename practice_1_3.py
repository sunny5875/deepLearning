import random
from tkinter import E
import numpy as np 

m = 10000 # sample training 수
n = 1000 # test sample 수
alpha = 0.01 # learning rate
k = 5000 # iteration 수
check = k /10 # print하는 수


def generateTrainMatrix():
    X = np.zeros((2,m), float)   
    Y = np.zeros((1,m), int)

    for i in range(m):
        x_train = np.array([random.uniform(-10,10), random.uniform(-10,10)])
        X[0,i] =  x_train[0]
        X[1,i] =  x_train[1]
      
        if x_train[0] < -5 or x_train[0] > 5:
            Y[0,i] = 1
        else:
            Y[0,i] = 0

    return (X,Y)

def generateTestMatrix():
    X = np.zeros((2,n), float)   
    Y = np.zeros((1,n), int)

    for i in range(n):
        x_train = np.array([random.uniform(-10,10), random.uniform(-10,10)])
        X[0,i] =  x_train[0]
        X[1,i] =  x_train[1]
      
        if x_train[0] < -5 or x_train[0] > 5:
            Y[0,i] = 1
        else:
            Y[0,i] = 0

    return (X,Y)

def sigmoid(x):
    return 1 / (1 +np.exp(-x))

def cross_entropy_loss(y_hat, y):
    loss = -(y*np.log(y_hat + 1e-10) + (1-y)*np.log(1-y_hat + 1e-10))
    return np.mean(loss)

  
def logistic_regression_model(w, b, x):
    return sigmoid(np.dot(w, x)+b)

def executeSimpleNeuralNetworkUsingVectorization():
    global alpha

    # initialize weight and bias
    W1 = np.array([[random.uniform(-1,1), random.uniform(-1,1)],
                   [random.uniform(-1,1), random.uniform(-1,1)]]) #(2,2)
    B1 = np.array([[random.uniform(-1,1)], 
                    [random.uniform(-1,1)]])#(2,1)
    W2 = np.array([[random.uniform(-1,1), random.uniform(-1,1)]])#(1,2)
    B2 = random.uniform(-1,1)#(1,1)
    print('initial W1 = ',W1)
    print('initial B1 = ',B1)     
    print('initial W2 = ',W2)
    print('initial B2 = ',B2)
    # generate sample
    X, Y = generateTrainMatrix()
    test_X, test_Y = generateTestMatrix()
 
    train_accuracy = 0.0
    
    for i in range(k):
        cost = 0.0

        # forward progpatgation
        A1 = logistic_regression_model(W1,B1,X)
        A2 = logistic_regression_model(W2,B2,A1)
        cost = cross_entropy_loss(A2, Y)

        #W2 back propagation
        dZ2 = A2 - Y #(1,m)
        dW2 = 1/m *  np.dot(dZ2, A1.transpose()) #np.matmul(dZ2, A1.T) #(2,1)
        dB2 = 1/m * np.sum(dZ2, axis = 1, keepdims = True)#(1,1)

        #W1 back propagation
        dZ1 = np.dot(W2.transpose(), dZ2) * (A1 * (1-A1))#np.matmul(W2.T, dZ2) * (A1 * (1-A1)) #(2,m)
        dW1 = 1/m * np.dot(dZ1, X.transpose())#np.matmul(dZ1, X.T)#dZ1 @ X.T #(2,2)
        dB1 = 1/m * np.sum(dZ1, axis = 1, keepdims = True) #(2,1)

          
        if (i+1) % check == 0 :
            print('-------------------', i+1, "th training RESULT -------------------")
            print('current W1 = ',W1)
            print('current B1 = ',B1)     
            print('current W2 = ',W2)
            print('current B2 = ',B2)
            print('current alpha: ', alpha)
            print("current training cost: ", cost)
            A2[A2>0.5] = 1
            A2[A2<0.5] = 0
            train_accuracy = np.sum(Y==A2)/m*100
            print("current train accuracy: %.2f%%" %(train_accuracy))
          

            test_A1 = logistic_regression_model(W1,B1,test_X)
            test_A2 = logistic_regression_model(W2,B2,test_A1)
            test_cost = cross_entropy_loss(test_A2, test_Y)
            test_A2[test_A2>0.5] = 1
            test_A2[test_A2<0.5] = 0
            test_accuracy = np.sum(test_Y == test_A2)/n*100
            
            print("current test cost: ", test_cost)
            print("current test accuracy: %.2f%%" %(test_accuracy))
           
            if train_accuracy < 90:
                alpha = ((100-train_accuracy)/ 100) 
                # alpha += 1
                # alpha = 3
            else:
                alpha = 0.01

        W1 = W1 - alpha * dW1
        B1 = B1 - alpha * dB1
        W2 = W2 - alpha * dW2
        B2 = B2 - alpha * dB2


            
    # after training  
    print('------------------- Final RESULT -------------------')
    print('estimated W1 = ',W1)
    print('estimated B1 = ',B1)     
    print('estimated W2 = ',W2)
    print('estimated B2 = ',B2)
    print('final alpha: ', alpha)

    A1 = logistic_regression_model(W1,B1,X)
    A2 = logistic_regression_model(W2,B2,A1)
    cost = cross_entropy_loss(A2, Y)
    A2[A2>0.5] = 1
    A2[A2<0.5] = 0
    train_accuracy = np.sum(Y == A2)/m*100
    
    print("final training cost: ", cost)
    print("final train accuracy: %.2f%%" %(train_accuracy))
    
    
    test_A1 = logistic_regression_model(W1,B1,test_X)
    test_A2 = logistic_regression_model(W2,B2,test_A1)
    test_A2[test_A2>0.5] = 1
    test_A2[test_A2<0.5] = 0
    test_accuracy = np.sum(test_Y == test_A2)/n*100
    print("final test accuracy: %.2f%%" %(test_accuracy))


if __name__ == '__main__':
    executeSimpleNeuralNetworkUsingVectorization()