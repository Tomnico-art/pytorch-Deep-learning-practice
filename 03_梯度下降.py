# Nico
# 时间：2021/9/4 18:32

# Nico
# 时间：2021/9/4 11:46

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

# Define the cost function
def cost(xs, ys):
    cost = 0
    for x, y in zip(xs, ys):
        y_pred = forward(x)
        cost += (y_pred - y) ** 2
    return cost / len(xs)

# Define the gradient function
def gradient(xs, ys):
    grad = 0
    for x, y in zip(xs, ys):
        grad += 2 * x * (x * w - y)
    return grad / len(xs)

epoch_list = []
cost_list = []
# Do the update
print('Predict (before training)', 4, forward(4))
for epoch in range(100):
    cost_val = cost(x_data, y_data)
    grad_val = gradient(x_data, y_data)
    w -= 0.01 * grad_val
    print('Epoch:', epoch, 'w=', w, 'loss', cost_val)
    epoch_list.append(epoch)
    cost_list.append(cost_val)
print('Predict (after training)', 4, forward(4))

# draw the graph
plt.plot(epoch_list, cost_list)
plt.ylabel('Cost')
plt.xlabel('Epoch')
plt.show()