import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg

# load data
data = np.loadtxt('Mackey-Glass.txt')
data_size = data.shape[0]
train_size = int(0.8*data_size)
test_size = data_size - train_size-1

# reservoir
input_size = 1
output_size = 1
# hidden_size = 1000
leaking_rate = 0.3
np.random.seed(3)

for index, hidden_size in enumerate([10, 100, 1000]):
    W_in = np.random.randn(hidden_size, 1+input_size)
    W_h = np.random.randn(hidden_size, hidden_size)

    rhoW = max(abs(linalg.eig(W_h)[0]))
    W_h = W_h / rhoW

    X = np.zeros((1+input_size+hidden_size, train_size))
    Yt = data[1:train_size+1]
    # print(X.shape)
    # print(Yt.shape)

    # run the reservoir with the data and collect X
    x = np.zeros((hidden_size, 1))
    for t in range(train_size):
        u = data[t]
        x = (1-leaking_rate)*x + leaking_rate * \
            np.tanh(np.matmul(W_in, np.vstack((1, u))) + np.matmul(W_h, x))
        X[:, t] = np.vstack((1, u, x))[:, 0]

    # ridge regression
    lambda_reg = 1e-8
    W_out = linalg.solve(np.matmul(X, X.T) + lambda_reg*np.eye(1+input_size+hidden_size),
                         np.matmul(X, Yt.T)).T

    # eval
    Y = np.zeros((output_size, test_size))
    u = data[train_size]
    for t in range(test_size):
        x = (1-leaking_rate)*x + leaking_rate * \
            np.tanh(np.matmul(W_in, np.vstack((1, u))) + np.matmul(W_h, x))
        y = np.matmul(W_out, np.vstack((1, u, x)))
        Y[:, t] = y
        u = y

    mse = sum(np.square(data[train_size+1:train_size+test_size+1] -
                        Y[0, 0:test_size])) / test_size
    # print('MSE = ' + str(mse))

    # plot
    plt.subplot(3, 1, index+1)
    plt.plot(data[train_size+1:train_size+test_size+1])
    plt.plot(Y.T)
    plt.title("Target and Predicted Signals, (hidden size = {}, MSE = {:.2f})".format(
        hidden_size, mse))
    plt.legend(['Target signal', 'Predicted Signals'])

plt.show()
