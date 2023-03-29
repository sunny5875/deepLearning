
# import os
# import sys
# import time
# import random
# import numpy as np

# def read_dataset(filename):
#     data = np.load(filename)
#     x = data['x']
#     y = data['y']
#     return x, y

# def generate_and_save_dataset(filename, size):
#     x = []
#     y = []
#     for _ in range(size):
#         temp = np.array([random.uniform(-10, 10), random.uniform(-10, 10)])
#         x.append(temp)

#         if temp[0] < -5 or temp[0] > 5:#temp.sum() > 0:
#             y.append(1)
#         else:
#             y.append(0)
    
#     x = np.array(x)
#     y = np.array(y)

#     # np.savez(filename, x=x, y=y)

#     return x, y

# def cross_entropy_loss(y_, y):
#     return -(y*np.log(y_+1e-10)+((1-y)*np.log(1-y_+1e-10)))

# def sigmoid(z):
#     return 1/(1+np.exp(-z))

# def model(w, b, x):
#     return sigmoid(np.dot(w, x)+b)

# def train_and_test(train_x, train_y, test_x, test_y, iteration, alpha, w1, b1, w2, b2, log_step):
#     train_size = train_x.shape[0]
#     test_size = test_x.shape[0]

#     train_x = train_x.T
#     test_x = test_x.T

#     start = time.time()
#     for step in range(iteration):
#         train_cost = 0
#         test_cost = 0
#         train_accuracy = 0
#         test_accuracy = 0
        
#         a1 = model(w1, b1.reshape((2,1)), train_x)
#         y_ = model(w2, b2, a1)
#         train_cost = -cross_entropy_loss(y_, train_y).sum()/train_size

#         dz2 = y_ - train_y
#         dw2 = np.dot(dz2, a1.T)/train_size
#         db2 = dz2.sum()/train_size
        
#         dz1 = np.dot(np.array([w2]).T, np.array([dz2]))*(a1*(1-a1))
#         dw1 = np.dot(dz1, train_x.T)/train_size
#         db1 = dz1.sum()/train_size

#         y_[y_>=0.5] = 1
#         y_[y_<0.5] = 0
#         train_accuracy = np.sum(y_==train_y)/train_size*100
        
#         if log_step:
#             y_ = model(w1, b1, test_x)
#             test_cost = -cross_entropy_loss(y_, test_y).sum()/test_size
            
#             y_[y_>=0.5] = 1
#             y_[y_<0.5] = 0
#             test_accuracy = np.sum(y_==test_y)/test_size*100

#         w1 = w1 - alpha*dw1
#         b1 -= alpha*db1
#         w2 = w2 - alpha*dw2
#         b2 -= alpha*db2
        
#         if (step+1) % 500 == 0:
#             print('Iteration #',(step+1))
#             print('Now W1 = ', w1)
#             print('Now B1 = ', b1)            
#             print('Now W2 = ', w2)
#             print('Now B2 = ', b2)
#             print()
#             print('Cost for Training Dataset = ', train_cost)
#             print('Cost for Testing Dataset  = ', test_cost)
#             print()
#             print('Accuracy for Training Dataset = %.2f%%' % (train_accuracy))
#             print('Accuracy for Testing Dataset  = %.2f%%' % (test_accuracy))
#             print()
#     end = time.time()

#     training_time = end-start
#     a1 = model(w1, b1.reshape((2,1)), train_x)
#     y_ = model(w2, b2, a1)
#     y_[y_>=0.5] = 1
#     y_[y_<0.5] = 0
#     train_accuracy = np.sum(y_==train_y)/train_size*100
    
#     start = time.time()
#     y_ = []
#     test_x = test_x.T

#     for x in test_x:
#         y_.append(model(w2, b2, model(w1, b1, x)))
        
#     y_ = np.array(y_)
#     y_[y_>=0.5] = 1
#     y_[y_<0.5] = 0
#     test_accuracy = np.sum(y_==test_y)/test_size*100
#     end = time.time()
#     test_time = end-start
    
#     return w1, b1, w2, b2, training_time, test_time, train_accuracy, test_accuracy

# def main(argv):
#     m = 10000
#     n = 500
#     k = 5000

#     mode = None
#     if len(argv)>1:
#         mode = argv[1]
    
#     train_filename = 'train_2018008395.npz'
#     test_filename = 'test_2018008395.npz'
    
#     w1 = np.array([[random.uniform(-10,10), random.uniform(-10,10)],
#                    [random.uniform(-10,10), random.uniform(-10,10)]])
#     b1 = np.array([random.uniform(-10,10), random.uniform(-10,10)])
#     w2 = np.array([random.uniform(-10,10), random.uniform(-10,10)])
#     b2 = random.uniform(-10,10)

#     if mode == 'zero':
#         w1 = np.array([[0.0,0.0],
#                        [0.0,0.0]])
#         b1 = np.array([0.0,0.0])
#         w2 = np.array([0.0,0.0])
#         b2 = 0.0

#     train_x = None
#     train_y = None
#     test_x = None
#     test_y = None

#     if os.path.isfile(train_filename) and os.path.isfile(test_filename):
#         train_x, train_y = read_dataset(train_filename)
#         test_x, test_y = read_dataset(test_filename)
#     else:
#         train_x, train_y = generate_and_save_dataset(train_filename, m)
#         test_x, test_y = generate_and_save_dataset(test_filename, n)
    
#     result = train_and_test(train_x, train_y, test_x, test_y, k, 3, w1, b1, w2, b2, mode=='step')
#     print('----------------RESULT----------------')
#     print('Estimated W1 = ',result[0])
#     print('Estimated B1 = ',result[1])     
#     print('Estimated W2 = ',result[2])
#     print('Estimated B2 = ',result[3])
#     print('Training Time = %f sec' % (result[4]))
#     print('Testing Time  = %f sec' % (result[5]))
#     print('Accuracy for Training Dataset = %.2f%%' % (result[6]))
#     print('Accuracy for Testing Dataset  = %.2f%%' % (result[7]))

# if __name__ == '__main__':
#     main(sys.argv)

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
    W1 = np.array([[random.uniform(-10,10), random.uniform(-10,10)],
                   [random.uniform(-10,10), random.uniform(-10,10)]]) #(2,2)
    B1 = np.array([[random.uniform(-10,10)], 
                    [random.uniform(-10,10)]])#(2,1)
    W2 = np.array([[random.uniform(-10,10), random.uniform(-10,10)]])#(1,2)
    B2 = random.uniform(-10,10)#(1,1)
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
                # alpha = ((100-train_accuracy)/ 100) * 3
                # alpha += 1
                alpha = 3
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