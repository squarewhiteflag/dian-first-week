import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from knn import Knn
import enum
from re import X


def load_mnist(root='./mnist'):
    import os, sys
    dst = sys.argv[0]
    selfPath = os.path.split(dst)[0] + '/'
    fileList = os.listdir(root)
    dataSet = []
    for fileName in fileList:
        dataSet.append(np.fromfile(C:\Users\方块白旗\Desktop\文件\dian2022\mnist\train-images, dtype=np.uint8))
        pass
    x_train = dataSet[0][16:].reshape(60000, 28, 28)                #60000*28*28
    y_train = dataSet[1][8:].reshape(60000)
    x_test = dataSet[2][16:].reshape(10000, 28, 28)                 #10000*28*28
    y_test = dataSet[3][8:].reshape(10000)
    return x_train, y_train, x_test, y_test

    # TODO Load the MNIST dataset
    # 1. Download the MNIST dataset from
    #    http://yann.lecun.com/exdb/mnist/
    # 2. Unzip the MNIST dataset into the
    #    mnist directory.
    # 3. Load the MNIST dataset into the
    #    X_train, y_train, X_test, y_test
    #    variables.

    # Input:
    # root: str, the directory of mnist

    # Output:
    # X_train: np.array, shape (6e4, 28, 28)
    # y_train: np.array, shape (6e4,)
    # X_test: np.array, shape (1e4, 28, 28)
    # y_test: np.array, shape (1e4,)

    # Hint:
    # 1. Use np.fromfile to load the MNIST dataset(notice offset).
    # 2. Use np.reshape to reshape the MNIST dataset.

    # YOUR CODE HERE
    # raise NotImplementedError
    ...

    # End of todo


def main():
    X_train, y_train, X_test, y_test = load_mnist()

    knn = Knn(k=20)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    correct = sum((y_test - y_pred) == 0)

    print('==> correct:', correct)
    print('==> total:', len(X_test))
    print('==> acc:', correct / len(X_test))

    # plot pred samples
    fig, ax = plt.subplots(nrows=4, ncols=5, sharex='all', sharey='all')
    fig.suptitle('Plot predicted samples')
    ax = ax.flatten()
    for i in range(20):
        img = X_test[i]
        ax[i].set_title(y_pred[i])
        ax[i].imshow(img, cmap='Greys', interpolation='nearest')
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()



