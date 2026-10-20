from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

# сигмоидная функция активации
def sigmoid(z):
    return 1. / (1. + np.exp(-z))

# строим метки классов с унитарным кодированием
def int_to_onehot(y, num_labels):
    ary = np.zeros((y.shape[0], num_labels))
    for i, val in enumerate(y):
        ary[i, val] = 1

    return ary

class NeuralNetMLP:
    def __init__(self, num_features, num_hidden, num_classes, random_seed=123):
        super().__init__()

        self.num_classes = num_classes

        # скрытый слой
        rng = np.random.RandomState(random_seed)

        self.weight_h = rng.normal(loc=0.0, scale=0.1, size=(num_hidden, num_features))
        self.bias_h = np.zeros(num_hidden)

        # выход
        self.weight_out = rng.normal(loc=0.0, scale=0.1,
                                     size=(num_classes, num_hidden))
        self.bias_out = np.zeros(num_classes)

    def forward(self, x):
        # скрытый слой
        z_h = np.dot(x, self.weight_h.T) + self.bias_h
        a_h = sigmoid(z_h)

        # выходной слой
        z_out = np.dot(a_h, self.weight_out.T) + self.bias_out
        a_out = sigmoid(z_out)

        return a_h, a_out

    def backward(self, x, a_h, a_out, y):
        # унитарное кодирование
        y_onehot = int_to_onehot(y, self.num_classes)

        d_loss__d_a_out = 2. * (a_out - y_onehot) / y.shape[0]

        d_a_out__d_z_out = a_out * (1. - a_out)  # производная сигмоиды

        delta_out = d_loss__d_a_out * d_a_out__d_z_out

        # градиент для выходных весов
        d_z_out__dw_out = a_h

        d_loss__dw_out = np.dot(delta_out.T, d_z_out__dw_out)
        d_loss__db_out = np.sum(delta_out, axis=0)

        # вторая часть
        d_z_out__a_h = self.weight_out

        d_loss__a_h = np.dot(delta_out, d_z_out__a_h)

        d_a_h__d_z_h = a_h * (1. - a_h)

        d_z_h__d_w_h = x

        d_loss__d_w_h = np.dot((d_loss__a_h * d_a_h__d_z_h).T, d_z_h__d_w_h)
        d_loss__d_b_h = np.sum((d_loss__a_h * d_a_h__d_z_h), axis=0)

        return (d_loss__dw_out, d_loss__db_out,
                d_loss__d_w_h, d_loss__d_b_h)

X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
y = y.astype(int)

print(X.shape)
print(y.shape)

# приведём пикселы к диапазону от -1 до 1
X = ((X / 255.) - 0.5) * 2

fig, ax = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True)
ax = ax.flatten()
for i in range(10):
    img = X[y == i].iloc[0].to_numpy().reshape(28, 28)
    ax[i].imshow(img, cmap='Greys')
ax[0].set_xticks([])
ax[0].set_yticks([])
plt.show()

fig, ax = plt.subplots(nrows=5, ncols=5, sharex=True, sharey=True)
ax = ax.flatten()
for i in range(25):
    img = X[y == 5].iloc[i].to_numpy().reshape(28, 28)
    ax[i].imshow(img, cmap='Greys')
ax[0].set_xticks([])
ax[0].set_yticks([])
plt.show()

X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=10000,
                                                  random_state=123, stratify=y)
X_train, X_valid, y_train, y_valid = train_test_split(X_temp, y_temp,
                                                      test_size=5000,
                                                      random_state=123,
                                                      stratify=y_temp)

model = NeuralNetMLP(num_features=28*28,
                     num_hidden=50,
                     num_classes=10)
