import numpy as np 
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD

#generate synthetic data
np.random.seed(42)
X = np.random.rand(100,1)
y = 2 * X + np.random.rand(100,1)*2

#step-1:define the model
model = Sequential()

#step-2:add the layer
model.add(Dense(1,input_dim=1))

#step-3:compile
model.compile(optimizer = SGD(learning_rate = 0.002),loss = 'mean_squared_error')

#step-4: train the model
model.fit(X,y,epochs = 100,verbose =0)

#step-5: make predictions
output = model.predict(X)

#plot
plt.scatter(X,y,label = 'original data')
plt.plot(X,  output, label = 'predicted output')
plt.xlabel('X-values')
plt.ylabel('Y-values')
plt.legend()
plt.show()