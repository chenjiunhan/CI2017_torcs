from sklearn.neural_network import MLPRegressor
import numpy as np
import matplotlib.pyplot as plt
import data_processor as dp

x = dp.input_data
y = dp.output_data
print(len(x), len(y))
x = dp.input_data.reshape(-1, 1)
y = dp.output_data.reshape(-1, 1)
print(len(x), len(y))
print(x, y)

nn = MLPRegressor(
    hidden_layer_sizes=(2,),  activation='tanh', solver='adam', alpha=0.001, batch_size='auto',
    learning_rate='constant', learning_rate_init=0.01, power_t=0.5, max_iter=10, shuffle=True,
    random_state=9, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
    early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

n = nn.fit(x, y)
test_x = x[0]
test_y = nn.predict(test_x)
#test_x = np.arange(0.0, 1, 0.05).reshape(-1, 1)
#test_y = nn.predict(test_x)
#fig = plt.figure()
#ax1 = fig.add_subplot(111)
#ax1.scatter(x, y, s=1, c='b', marker="s", label='real')
#ax1.scatter(test_x,test_y, s=10, c='r', marker="o", label='NN Prediction')
#plt.show()
