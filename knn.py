

import numpy as np
from tqdm import tqdm
import operator

class Knn(object):

    def __init__(self, k=10):
        self.k = k

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.Size = 60000

    def predict_internet(self, X):
        size = self.Size
        pbar = tqdm(
            total= size, initial= 0,
            unit= 'B', unit_scale= True, leave= True)

        dataSet = self.X
        labels = self.y
        k = self.k
        results = []
        for inX in X:
            dataSetSize = dataSet.shape[0]#查看矩阵的维度
            diffMat = np.tile(inX,(dataSetSize,1)) - dataSet
            sqDiffMat = diffMat**2
            sqDistances = sqDiffMat.sum(axis=1)
            distances = sqDistances**0.5
            sortedDistIndicies = distances.argsort()
            classCount={}
            for i in range(k):
                voteIlabel = labels[sortedDistIndicies[i]]
                classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
            sortedClassCount = sorted(classCount.items(),
                key= operator.itemgetter(1),reverse=True)
            results.append(sortedClassCount[0][0])
            pbar.update(1)
        return results


    def predict(self, X):

        # TODO Predict the label of X by
        # the k nearest neighbors.

        # Input:
        # X: np.array, shape (n_samples, n_features)

        # Output:
        # y: np.array, shape (n_samples,)

        # Hint:
        # 1. Use self.X and self.y to get the training data.
        # 2. Use self.k to get the number of neighbors.
        # 3. Use np.argsort to find the nearest neighbors.

        # YOUR CODE HERE
        # Totally 60000 groups of x_data, each data group is a 28x28 matrix and has a int label between 0-9
        trainingSize = self.Size
        testSize = X.shape[0]

        results = np.empty((trainingSize,))         #创建数组
        pbar = tqdm(total= testSize, initial= 0,unit_scale= True, leave= True)          #进度条
        for index_test in range(0, testSize):
            distance = np.empty((trainingSize,))       #为存放距离创建数组
            labelCounter = np.zeros((10,))              #标签
            for index_dist, Matrix_train in enumerate(self.X):          #计数
                distance[index_dist] = np.sum((Matrix_train - X[index_test]))                   #行求和
                pass
            distIndicies = np.argsort(distance)             #排序
            for i in range(self.k):
                label = self.y[distIndicies[i]]
                labelCounter[label] += 1                    #计数
            predictLabel = labelCounter.argmax()
            results[index_test] = predictLabel
            pbar.update(1)

        return results
        #A single group of sum of distance: np.sum((X_train[0]-X_test[0])**0.5)

        pass
        # End of todo

if __name__ == '__main__':

    pass