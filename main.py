from time import time

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import sklearn.svm as svm
from sklearn.metrics import accuracy_score


def startjob():
    print('Load Training File Start...')
    data = pd.read_csv('./data/optdigits.tra', header=None)

    x, y = data.iloc[:,range(64)].values, data.iloc[:, 64].values
    images = x.reshape(-1, 8, 8)
    print(images.shape)
    y = y.ravel().astype(np.int)
    print(x.shape, y.shape)
    print('Load Test Data Start...')
    data = pd.read_csv('./data/optdigits.tes', header=None)
    x_test, y_test = data.iloc[:,range(64)].values, data.iloc[:, 64].values

    images_test = x_test.reshape(-1, 8, 8)
    y_test = y_test.ravel().astype(np.int)
    # data = np.loadtxt('./data/optdigits.tes', dtype=np.float, delimiter=',')
    # x_test, y_test = np.split(data, (-1, ), axis=1)
    print(x_test.shape, y_test.shape)
    print('Load Data OK')

    mpl.rcParams['font.sans-serif'] = ['SimHei']
    mpl.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(15, 9), facecolor='w')
    #训练图片与测试图片各取16个
    #For the Agg, ps and pdf backends,
    # interpolation = ‘none’ works well when a big image is scaled down,
    # while interpolation = ‘nearest’ works well when a small image is scaled up.
    for index, image in enumerate(images[:16]):
        plt.subplot(4, 8, index + 1)
        plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
        plt.title('训练图片: %i' % y[index])
        print(index)
    for index, image in enumerate(images_test[:16]):
        plt.subplot(4, 8, index + 17)
        plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
        # save_image(image.copy(), index)
        plt.title('测试图片: %i' % y_test[index])
    plt.tight_layout()
    # plt.show()
    model = svm.SVC(C=10, kernel='rbf', gamma=0.001)
    print('Start Learning...')
    t0 = time()
    model.fit(x, y)
    t1 = time()
    t = t1 - t0
    print('训练+CV耗时: %d分钟%.3f秒' % (int(t/60), t - 60 * int(t/60)))
    print('Learning is OK')
    print('训练集准确率: ', accuracy_score(y, model.predict(x)))

    y_hat = model.predict(x_test)
    print('测试集准确率: ', accuracy_score(y_test, y_hat))
    print(y_hat)
    print(y_test)

    err_images = images_test[y_test != y_hat]
    err_y_hat = y_hat[y_test != y_hat]
    err_y = y_test[y_test != y_hat]
    print(err_y_hat)
    print(err_y)
    plt.figure(figsize=(10, 8), facecolor='w')
    for index, image in enumerate(err_images):
        if index >= 16:
            break
        plt.subplot(4, 4, index + 1)
        plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
        plt.title('错分为: %i, 真实值: %i' % (err_y_hat[index], err_y[index]))
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    startjob()