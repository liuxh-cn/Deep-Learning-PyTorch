import numpy as np
import matplotlib.pyplot as plt
import operator
from functools import reduce
from Demo.tool import *
# %matplotlib inline

r = np.random.randn(200)*0.8    # 高斯分布随机数 ~(0,1)
x1 = np.linspace(-3, 1, 200)    # [-3,1]等距生成200个数
x2 = np.linspace(-1, 3, 200)
y1 = x1*x1 + 2*x1 - 2 + r       # 数组对应位置相乘
y2 = -x2*x2 + 2*x2 + 2 + r
X = np.hstack(([x1, y1], [x2, y2]))
Y = np.hstack((np.zeros((1, 200)), np.ones((1, 200))))

# print(X.shape)
# print(Y.shape)
# print(X)
# print(Y)
plt.scatter(X[0, :], X[1, :], c=Y[0, :], s=40, cmap=plt.cm.Spectral)
# plt.show()

# 定义神经网络输入层、隐藏层、输出层的神经元个数
m = X.shape[1]      # 样本个数
n_x = X.shape[0]    # 输入层神经元个数 - 样本维度
n_h = 4             # 隐藏层神经元个数
n_y = Y.shape[0]    # 输出层神经元个数

# 网络参数 W 和 b 初始化
W1 = np.random.randn(n_h, n_x) * 0.01   # 隐藏层
b1 = np.zeros((n_h, 1))
W2 = np.random.randn(n_y, n_h) * 0.01   # 输出层
b2 = np.zeros((n_y, 1))

assert(W1.shape == (n_h, n_x))
assert(b1.shape == (n_h, 1))
assert(W2.shape == (n_y, n_h))
assert(b2.shape == (n_y, 1))
parameters = {"W1": W1,
              "b1": b1,
              "W2": W2,
              "b2": b2}

# 正向传播 - 计算Z1，A1，Z2，A2

def sigmoid(x):
    s = 1/(1+np.exp(-x))
    return s

def forward_propagation(X, parameters):

    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)

    assert(A2.shape == (1, X.shape[1]))

    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}

    return A2, cache

def compute_cost(A2, Y, parameters):
    m = Y.shape[1]

    # 计算交叉熵损失函数
    logprobs = np.multiply(np.log(A2), Y) + np.multiply(np.log(1-A2), (1-Y))
    cost = - 1/m * np.sum(logprobs)

    cost = np.squeeze(cost)

    return cost

# 反向传播
def backward_propagation(parameters, cache, X, Y):

    m = X.shape[1]

    W1 = parameters["W1"]
    W2 = parameters["W2"]

    A1 = cache["A1"]
    A2 = cache["A2"]

    # 反向求导
    dZ2 = A2 - Y
    dW2 = 1/m*np.dot(dZ2,A1.T)
    db2 = 1/m*np.sum(dZ2,axis=1,keepdims=True)
    dZ1 = np.dot(W2.T,dZ2)*(1 - np.power(A1, 2))
    dW1 = 1/m*np.dot(dZ1,X.T)
    db1 = 1/m*np.sum(dZ1,axis=1,keepdims=True)

    # 存储各个梯度值
    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}

    return grads

# 网络参数更新
def update_parameters(parameters, grads, learning_rate = 0.1):

    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]

    W1 = W1 - learning_rate*dW1
    b1 = b1 - learning_rate*db1
    W2 = W2 - learning_rate*dW2
    b2 = b2 - learning_rate*db2

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters

# 搭建整个网络模型
def nn_model(X, Y, n_h = 3, num_iterations = 10000, print_cost=False):

    m = X.shape[1]      # 样本个数
    n_x = X.shape[0]    # 输入层神经元个数
    n_y = Y.shape[0]    # 输出层神经元个数

    W1 = np.random.randn(n_h,n_x)*0.01
    b1 = np.zeros((n_h,1))
    W2 = np.random.randn(n_y,n_h)*0.01
    b2 = np.zeros((n_y,1))

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    # 迭代训练
    J = []     # 存储损失函数
    for i in range(0, num_iterations):

        A2, cache = forward_propagation(X, parameters)            # 正向传播
        cost = compute_cost(A2, Y, parameters)                    # 计算损失函数
        grads = backward_propagation(parameters, cache, X, Y)     # 反向传播
        parameters = update_parameters(parameters, grads)         # 更新权重
        J.append(cost)

        # 每隔 1000 次训练，打印 cost
        if print_cost and i % 1000 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))

    return parameters

parameters = nn_model(X, Y, n_h = 3, num_iterations = 10000, print_cost=True)

def predict(parameters, X):

    A2, cache = forward_propagation(X, parameters)
    predictions = A2 > 0.5

    return predictions

# 使用训练好的模型对新样本进行预测
y_pred = predict(parameters,X)
accuracy = np.mean(y_pred == Y)
print(accuracy)

# 绘制分类界限
def plot_decision_boundary(model, X, y):

    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[0, :], X[1, :], c=reduce(operator.add, Y), cmap=plt.cm.Spectral)
plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
plt.title("Decision Boundary, hidden layers = 5")
plt.show()










