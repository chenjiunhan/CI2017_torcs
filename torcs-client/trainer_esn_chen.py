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
x = dp.input_data
#scaler = preprocessing.StandardScaler().fit(x)
#x = scaler.transform(x)

y = dp.output_data
#y = y/2
#scaler_y = preprocessing.StandardScaler().fit(y)
#print('!!', scaler_y.mean_)
#y = scaler_y.transform(y)
                                     
n_inputs = len(x[0])
n_outputs = len(y[0])
esn = ESN(n_inputs=n_inputs,
    n_outputs=n_outputs,
    n_reservoir=100,
	#sparsity=0.01,
    spectral_radius=1,
    teacher_shift = 0,
    #teacher_scaling = 0.9,    
    #out_activation=np.tanh,
    #inverse_out_activation=np.arctanh,
#    noise=0.001,
    silent=False)

pred_train = esn.fit(x, y)
# print "test error:"
x = dp.input_data
#scaler = preprocessing.StandardScaler().fit(x)
#x = scaler.transform(x)

y = dp.output_data

test_x = np.array(x[4200:4220]).reshape(20, n_inputs)
print('test input: ', test_x)
#test_x = np.array([ -9.68175000e-04,3.31107000e-01,6.41615328e-01,4.01361000e+00
#, 4.16770000e+00,4.66468000e+00,5.74039000e+00,8.18601000e+00
#, 1.21075000e+01,1.61838000e+01,2.46810000e+01,5.28110000e+01 
#, 2.00000000e+02,8.12428000e+01,4.32493000e+01,2.96219000e+01
#, 2.26558000e+01,1.56703000e+01,1.11704000e+01,9.16351000e+00
#, 8.24416000e+00,7.98714000e+00]).reshape(1, n_inputs)

#test_y = esn.predict(scaler.transform(test_x))
test_y = esn.predict(test_x)
#print('output: ', y[4200:4220])
#print('result output: ', test_y)
