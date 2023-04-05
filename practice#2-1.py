import tensorflow as tf
import numpy as np 
import random

m = 10000 # sample training 수
n = 1000 # test 수

def generateData(size):
    X = np.zeros((size,2), float)   
    Y = np.zeros((size,1), int)

    for i in range(size):
        x_train = np.array([random.uniform(-10,10), random.uniform(-10,10)])
        X[i,0] =  x_train[0]
        X[i,1] =  x_train[1]
      
        if x_train[0] < -5 or x_train[0] > 5:
            Y[i,0] = 1
        else:
            Y[i,0] = 0

    return (X,Y)


x_train, y_train = generateData(m)
x_test, y_test = generateData(n)

print(x_train.shape, y_train.shape)
print(x_test.shape, y_train.shape) 


model = tf.keras.models.Sequential([ #using kera 2 layer neural network
  tf.keras.layers.Flatten(input_shape=(2, 1)),
  tf.keras.layers.Dense(2, activation='elu'), # layer1
  # tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(1, activation='relu') # layer2
])

loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)



model.compile(optimizer='SGD', #optimizer
              loss=loss_fn,
              metrics=['accuracy'])

model.summary()
model.fit(x_train, y_train, epochs=50, verbose=2)

model.evaluate(x_test,  y_test, verbose=2) #verbose는 프린트 모드 





