from sklearn.neural_network import MLPRegressor
from sklearn import preprocessing
import numpy as np
import matplotlib.pyplot as plt
import data_processor as dp
from pyESN import ESN

'''clf = MLPRegressor(solver='lbfgs', alpha=1e-5,
hidden_layer_sizes=(5, 2), random_state=1)
X=[[-61, 25, 0.62, 0.64, 2, -35, 0.7, 0.65], [2,-5,0.58,0.7,-3,-15,0.65,0.52]]
y=[ [0.63, 0.64], [0.58,0.61] ]
clf.fit(X,y)'''
#exit(0)
x = dp.input_data.tolist()
#scaler = preprocessing.StandardScaler().fit(x)
#x = scaler.transform(x)
y = dp.output_data.tolist()

clf = MLPRegressor(solver='lbfgs', alpha=1e-3, learning_rate_init=1e-3, batch_size = 100, hidden_layer_sizes=(50, 30), random_state=1, activation='tanh', max_iter = 100, verbose=True)

clf.fit(x,y)
print('test input: ', x[5], '\n', 'test output: ', y[5])
test_x = [x[5]]
test_y = clf.predict(test_x)
print('result output: ', test_y)
