import random
import numpy as np 

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
        x_train = np.array([random.uniform(-10,10), random.uniform(-10,10)])
        X[0,i] =  x_train[0]
        X[1,i] =  x_train[1]
      
        if x_train[0] < -5 or x_train[0] > 5:
            Y[0,i] = 1
        else:
            Y[0,i] = 0

    return (X,Y)

def generate_and_save_dataset(size):
    x = []
    y = []
    for _ in range(size):
        temp = np.array([random.uniform(-10, 10), random.uniform(-10, 10)])
        x.append(temp)

        if temp[0] < -5 or temp[0] > 5:#temp.sum() > 0:
            y.append(1)
        else:
            y.append(0)
    
    x = np.array(x)
    y = np.array(y)

    return x, y
    
def logisticRegressionModel(w, b, x):
    return sigmoid(np.dot(w, x)+b)

def executeSimpleNeuralNetworkUsingVectorization():
    global alpha

    # initialize weight and bias
    W1 = np.array([[random.uniform(-10,10), random.uniform(-10,10)],
                   [random.uniform(-10,10), random.uniform(-10,10)]]) #(2,2)
    B1 = np.array([[random.uniform(-10,10)], [random.uniform(-10,10)]])#(2,1)
    W2 = np.array([[random.uniform(-10,10), random.uniform(-10,10)]])#(1,2)
    B2 = random.uniform(-10,10)#(1,1)
    # generate sample
    X, Y = generateSampleMatrix()
 
    
    for i in range(k):
        cost = 0

        # hidden layer forward progpatgation
        # Z1 = np.matmul(W1.T, X) + B1 #(2,m)
        # A1 = sigmoid(Z1) #(2,m)
        # # output forward propagation
        # Z2 = np.matmul(W2, A1) + B2 #(1,m)
        # A2 = sigmoid(Z2) #(1,m)
        # J += cross_entropy_loss(A2, Y)
        A1 = logisticRegressionModel(W1,B1,X)
        A2 = logisticRegressionModel(W2,B2,A1)
        cost = -cross_entropy_loss(A2, Y).sum()/m

        #W2 back propagation
        dZ2 = A2 - Y #(1,m)
        dW2 = 1/m *  np.dot(dZ2, A1.transpose()) #np.matmul(dZ2, A1.T) #(2,1)
        dB2 = 1/m * np.sum(dZ2, axis = 1, keepdims = True)#(1,1)

        #W1 back propagation
        dZ1 = np.dot(W2.transpose(), dZ2) * (A1* (1-A1))#np.matmul(W2.T, dZ2) * (A1 * (1-A1)) #(2,m)
        dW1 = 1/m * np.dot(dZ1, X.transpose())#np.matmul(dZ1, X.T)#dZ1 @ X.T #(2,2)
        dB1 = 1/m * np.sum(dZ1, axis = 1, keepdims = True)#(2,1)


        W1 = W1 - alpha * dW1
        B1 -= alpha * dB1
        W2 = W2 - alpha * dW2
        B2 -= alpha * dB2


        
        if (i+1) >= 4000: 
            alpha = 0.01

          
        if (i+1) % 500 == 0 :
            print('----------------', i+1, "th RESULT----------------")
            print('Estimated W1 = ',W1)
            print('Estimated B1 = ',B1)     
            print('Estimated W2 = ',W2)
            print('Estimated B2 = ',B2)
            print("cost: ", cost)
            A2[A2>=0.5] = 1
            A2[A2<0.5] = 0
            train_accuracy = np.sum(Y==A2)/m*100
            print("accuracy: ", train_accuracy)#np.mean(J))

            if train_accuracy < 50:
                alpha = 3
     


    A1 = logisticRegressionModel(W1,B1,X)
    A2 = logisticRegressionModel(W2,B2,A1)
    A2[A2>=0.5] = 1
    A2[A2<0.5] = 0
    train_accuracy = np.sum(Y==A2)/m*100
    print('----------------RESULT----------------')
    print('Estimated W1 = ',W1)
    print('Estimated B1 = ',B1)     
    print('Estimated W2 = ',W2)
    print('Estimated B2 = ',B2)
    print("final accuracy: %.2f%%" %(train_accuracy))


def test():
    global alpha
    # initialize weight and bias
    w1 = np.array([[random.uniform(-10,10), random.uniform(-10,10)],
                   [random.uniform(-10,10), random.uniform(-10,10)]])
    b1 = np.array([random.uniform(-10,10), random.uniform(-10,10)])
    w2 = np.array([random.uniform(-10,10), random.uniform(-10,10)])
    b2 = random.uniform(-10,10)

    # generate sample
    train_x, train_y  = generate_and_save_dataset(m)
    test_x, test_y  = generate_and_save_dataset(m)
    train_x = train_x.T
    test_x = test_x.T
    
    for i in range(k):
        train_cost = 0
        train_accuracy = 0
        test_accuracy = 0
        
        a1 = logisticRegressionModel(w1, b1.reshape((2,1)), train_x)
        y_ = logisticRegressionModel(w2, b2, a1)
        train_cost = -cross_entropy_loss(y_, train_y).sum()/m

        dz2 = y_ - train_y
        dw2 = np.dot(dz2, a1.T)/m
        db2 = dz2.sum()/m
        
        dz1 = np.dot(np.array([w2]).T, np.array([dz2]))*(a1*(1-a1))
        dw1 = np.dot(dz1, train_x.T)/m
        db1 = dz1.sum()/m

        y_[y_>=0.5] = 1
        y_[y_<0.5] = 0
        train_accuracy = np.sum(y_==train_y)/m*100
        

        w1 = w1 - alpha*dw1
        b1 -= alpha*db1
        w2 = w2 - alpha*dw2
        b2 -= alpha*db2
        
        # if (i+1) % 500 == 0:
        #     print(i+1,"th cost: ", train_cost)
        #     print(i+1,"th accuracy: ", train_accuracy)
   
    a1 = logisticRegressionModel(w1, b1.reshape((2,1)), train_x)
    y_ = logisticRegressionModel(w2, b2, a1)
    y_[y_>=0.5] = 1
    y_[y_<0.5] = 0
    train_accuracy = np.sum(y_==train_y)/m*100
    
   
 
    print('----------------RESULT----------------')
    print('Estimated W1 = ',w1)
    print('Estimated B1 = ',b1)     
    print('Estimated W2 = ',w2)
    print('Estimated B2 = ',b2)
    print('Accuracy for Training Dataset = %.2f%%' % (train_accuracy))



executeSimpleNeuralNetworkUsingVectorization()