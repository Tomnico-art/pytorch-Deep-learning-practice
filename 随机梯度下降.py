# Nico
# 时间：2021/9/5 11:06

import numpy as np
import matplotlib.pyplot as plt

# Prepare the train set
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

# initial guess of weight
w = 1.0

# Define the model
def forward(x):
    return x * w

# Define the loss function
def loss (x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2

# Define the gradient function
def gradient(x, y):
    return 2 * x * (x * w - y)

epoch_list = []
cost_list = []
# Do the update
print('Predict (before training)', 4, forward(4))
for epoch in range(100):
    for x, y in zip(x_data, y_data):
        grad = gradient(x, y)
        w = w - 0.01 * grad
        print('\tgrad: ', x, y, grad)
        l = loss(x, y)

    print('Epoch:', epoch, 'w=', w, 'loss', l)
    epoch_list.append(epoch)
    cost_list.append(l)
print('Predict (after training)', 4, forward(4))

# draw the graph
plt.plot(epoch_list, cost_list)
plt.ylabel('Cost')
plt.xlabel('Epoch')
plt.show()