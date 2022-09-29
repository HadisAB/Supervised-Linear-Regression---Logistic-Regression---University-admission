import numpy as np
import matplotlib.pyplot as plt
from utils import *
import copy
import math

#%matplotlib inline

# load dataset
X_train, y_train = load_data("data/ex2data1.txt")

# Plot examples
plot_data(X_train, y_train[:], pos_label="Admitted", neg_label="Not admitted")

# Set the y-axis label
plt.ylabel('Exam 2 score')
# Set the x-axis label
plt.xlabel('Exam 1 score')
plt.legend(loc="upper right")
plt.show()

#Compute sigmoid
def sigmoid(z):
    g = 1 / (1 + np.exp(-z))
    return g


# GRADED FUNCTION: compute_cost
def compute_cost(X, y, w, b, lambda_=1):
    m, n = X.shape
    Loss = 0
    for i in range(m):
        z = np.dot(X[i], w) + b
        f = sigmoid(z)
        Loss = Loss - (y[i] * np.log(f) + (1 - y[i]) * np.log(1 - f))
    total_cost = Loss / m
    return total_cost


def compute_gradient(X, y, w, b, lambda_=None):
    m, n = X.shape
    dj_dw = np.zeros(w.shape)
    dj_db = 0.

    for i in range(m):
        z_wb = sigmoid(np.dot(X[i], w) + b) - y[i]
        dj_db = dj_db + sigmoid(np.dot(X[i], w) + b) - y[i]
        for j in range(n):
            dj_dw[j] = dj_dw[j] + z_wb * X[i, j]

    dj_dw = dj_dw / m
    dj_db = dj_db / m
    return dj_db, dj_dw


def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters, lambda_):
    m = len(X)
    # An array to store cost J and w's at each iteration primarily for graphing later
    J_history = []
    w_history = []

    for i in range(num_iters):

        # Calculate the gradient and update the parameters
        dj_db, dj_dw = gradient_function(X, y, w_in, b_in, lambda_)

        # Update Parameters using w, b, alpha and gradient
        w_in = w_in - alpha * dj_dw
        b_in = b_in - alpha * dj_db

        # Save cost J at each iteration
        if i < 100000:  # prevent resource exhaustion
            cost = cost_function(X, y, w_in, b_in, lambda_)
            J_history.append(cost)

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i % math.ceil(num_iters / 10) == 0 or i == (num_iters - 1):
            w_history.append(w_in)
            print(f"Iteration {i:4}: Cost {float(J_history[-1]):8.2f}   ")

    return w_in, b_in, J_history, w_history  # return w and J,w history for graphing


np.random.seed(1)
intial_w = 0.01 * (np.random.rand(2).reshape(-1,1) - 0.5)
initial_b = -8


# Some gradient descent settings
iterations = 10000
alpha = 0.001

w,b, J_history,_ = gradient_descent(X_train ,y_train, initial_w, initial_b,
                                   compute_cost, compute_gradient, alpha, iterations, 0)


plot_decision_boundary(w, b, X_train, y_train)


#Prediction
def predict(X, w, b):
    # number of predicted samples
    m, n = X.shape
    p = np.zeros(m)
    for i in range(m):
        if sigmoid(np.dot(X[i], w) + b) >= 0.5:
            p[i] = 1
    return p

#Compute accuracy on our training set
p = predict(X_train, w,b)
print('Train Accuracy: %f'%(np.mean(p == y_train) * 100))