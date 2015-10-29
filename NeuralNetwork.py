import numpy as np
import matplotlib.pyplot as plt

def tanh(x):
    return np.tanh(x)

def tanh_deriv(x):
    return 1.0 - np.tanh(x) * np.tanh(x)

def sigmoid(x):
    return 1/(1+np.exp(x))

def sigmoid_deriv(x):
    return (1-sigmoid(x))*sigmoid(x)

class NeuralNetwork:
    def __init__(self, layers):

        self.activation = tanh
        self.activation_deriv = tanh_deriv

        self.weights = []
        #   for i in range(1, len(layers) - 1):
        for i in range(1, len(layers) - 1):
            #   the weights is from -0.5 to 0.5
            self.weights.append(np.random.random((layers[i - 1] + 1, layers[i] + 1)) - 0.5)
        self.weights.append((np.random.random((layers[-2] + 1, layers[-1])) - 0.5))

    def train(self, X, y, epochs, learning_rate=0.2):

        # bias setting
        temp = np.ones([X.shape[0], X.shape[1] + 1])
        temp[:, 0:-1] = X
        X = temp
        y = np.array(y)

        for k in range(epochs):
            i = k % len(X)
            output = [X[i]]

            # feed forward
            for l in range(len(self.weights)):
                output.append(self.activation(np.dot(output[l], self.weights[l])))  # output of each layer

            # last layer error gradient
            error = y[i] - output[-1]
            deltas = [error * self.activation_deriv(output[-1])]

            # hidden layer error gradient
            for l in range(len(output) - 2, 0, -1):
                #   print self.weights
                deltas.append(np.dot(self.weights[l], deltas[-1]) * self.activation_deriv(output[1]))

            # back propagation
            for i in range(len(self.weights)):
                layer = np.array([output[len(self.weights) - i - 1]])
                delta = np.array([deltas[i]])
                self.weights[len(self.weights) - i - 1] += learning_rate * np.dot(layer.T, delta)
                print np.dot(layer.T, delta)
        print np.sum(self.weights[0])

    def predict(self, x):
        x = np.array(x)
        temp = np.ones([len(x) + 1])
        temp[0:-1] = x
        output = temp

        for l in range(len(self.weights)):
            output = self.activation(np.dot(output, self.weights[l]))
        return output



# Time series prediction
# input_size = 17
# nn = NeuralNetwork([input_size, 9, 5, 3, 1])
input_size = 4   # number of data used as input of visible layer
nn = NeuralNetwork([input_size, 5, 1])

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
