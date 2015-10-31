import numpy as np
import matplotlib.pyplot as plt

def tanh(x):
    return np.tanh(x)

def tanh_d(x):
    return 1.0 - np.tanh(x) * np.tanh(x)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_d(x):
    return (1-sigmoid(x))*sigmoid(x)

class NeuralNetwork:
    def __init__(self, layers):
        #   number of layers in this net, visible and output layer included
        self.num_layers = len(layers)
        #   randomly initial weights from -0.5 to 0.5
        self.weights = []
        for idx in range(1, self.num_layers - 1):
            self.weights.append(np.random.random((layers[idx - 1] + 1, layers[idx])) - 0.5)
        self.weights.append((np.random.random((layers[-2] + 1, layers[-1])) - 0.5))
        #   choosing activation function
        self.f_activation = tanh
        self.f_activation_d = tanh_d

    def train(self, x, y, epochs, learning_rate=0.2):
        for k in range(epochs):

            for j in range(self.num_layers-1):
                if j == 0:
                    # activation of the first layer, which is just the input
                    activation = [x[k % len(x)]]
                    # activation of the second layer, computed by adding bias term
                    next_activation = self.f_activation(np.dot(np.append(activation, 1.0), self.weights[j]))
                else:
                    # feed forward, compute the activation of each layer
                    next_activation = self.f_activation(np.dot(np.append(activation[-1], 1.0), self.weights[j]))
                activation.append(next_activation)  # actiation[0]: first layer

            # gradient of error function between the last two layers
            absolute_er = y[i] - activation[-1]
            er_d = self.f_activation_d(activation[-1])
            er_func_grad = [absolute_er * er_d]   # er_func_grad[0]: last layer

            # gradient of error function between other layers
            for j in range(self.num_layers - 2, 0, -1):
                #  back propagate error gradient through weight matrix, for BOTH weight matrix and bias
                er_grad_back = np.dot(self.weights[j], er_func_grad[-1])
                er_d = self.f_activation_d(activation[j])
                next_er_func_grad = er_grad_back[:-1] * er_d  # error of bias DOESN'T propagate
                er_func_grad.append(next_er_func_grad)

            # gradient descent of weight matrix
            for idx in range(1, self.num_layers, 1):
                cnt = -idx  # counting begins from the top layer
                aa = np.array(activation[cnt-1])  # MUST change into a two dimensional array
                gg = np.array(er_func_grad[idx-1])
                step = learning_rate * np.dot(np.array([aa]).T, np.array([gg]))
                self.weights[cnt][0:-1] += step
            # print np.sum(self.weights[0])
            # how about bias terms???

    def predict(self, x):
        for idx in range(self.num_layers-1):
            if idx == 0:
                activation = self.f_activation(np.dot(np.concatenate((x, [1])), self.weights[idx]))
            else:
                activation = self.f_activation(np.dot(np.concatenate((activation, [1])), self.weights[idx]))
        return activation



# Time series prediction
# input_size = 17
# nn = NeuralNetwork([input_size, 9, 5, 3, 1])
input_size = 4   # number of data used as input of visible layer
nn = NeuralNetwork([input_size, 3, 1])  # bias are not counted

data = np.loadtxt("monthly_price.txt")
test_num = 4     # hold some most recent data for test purpose, say four
input_num = len(data)-test_num-input_size

# train neural net
visible_data = np.array([])
output_data = np.array([])
for i in range(input_num):
    visible_data = np.concatenate((visible_data, data[i:input_size+i]))
    print i
    output_data = np.concatenate((output_data, [data[input_size+i]]))
visible_data = np.resize(visible_data, (input_num, input_size))
epochs = 4000
nn.train(np.array(visible_data), np.array(output_data), epochs)

# test data set
test_output_data = []
test_visible_data = []
for i in range(test_num):
    test_visible_data.append(data[-i-input_size-1:-i-1])
    if i == 0:
        test_output_data.append([data[-1]])
    else:
        test_output_data.append([data[-i-1]])
test_output_data = np.array(test_output_data)
test_visible_data = np.array(test_visible_data)

# test neural net
er = 0
mag = 0
for i in range(test_num):
    nn_predict = nn.predict(test_visible_data[i])
    er += pow((test_output_data[i]-nn_predict), 2)
    mag += pow(test_output_data[i], 2)
print np.sqrt(er/mag)

#
# y = np.array([0, 1, 1, 0])
# epochs = np.logspace(2, 5, 100)
# for k in range(len(epochs)):
#     nitr = int(epochs[k])
#     nn.train(X, y, nitr)
#     cnt = 0
#     print(epochs[k], er[0][k])
# # fig = pyplot.figure()
#
# plt.plot(epochs, er[0])
# plt.xscale('log')
# plt.yscale('log')
# plt.show()
