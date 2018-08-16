# coding=utf-8
import numpy as np
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.datasets.samples_generator import make_classification
from sklearn.svm import SVC
from sklearn.cross_validation import cross_val_score
from sklearn.learning_curve import learning_curve
from sklearn.learning_curve import validation_curve
import pickle
from app import approot
from sklearn.externals import joblib


# sklearn methods

##classification(supervision)


##regression(supervision)

##clustering(unsupervision)

##dimensionality reduction

##sklearn normal model


def iris_show():
    iris = datasets.load_iris()
    iris_X = iris.data
    iris_y = iris.target
    print(iris_X[:2, :])
    print(iris_y)

    X_train, X_test, Y_train, Y_test = train_test_split(iris_X, iris_y, test_size=0.3)

    # print(Y_train)
    knn = KNeighborsClassifier()

    knn.fit(X_train, Y_train)
    print(knn.predict(X_test))
    print(Y_test)


def dataset_show():
    loaded_data = datasets.load_boston()
    data_X = loaded_data.data
    data_y = loaded_data.target
    model = LinearRegression()
    model.fit(data_X, data_y)
    print(model.predict(data_X[:4, :]))
    print(data_y[:4])


def make_data():
    X, y = datasets.make_regression(n_samples=100, n_features=1, n_targets=1, noise=1)
    plt.scatter(X, y)
    plt.show()


def sklearn_mode():
    loaded_data = datasets.load_boston()
    data_X = loaded_data.data
    data_y = loaded_data.target
    model = LinearRegression()
    model.fit(data_X, data_y)
    print(model.coef_)  # y=0.1x+0.3
    print(model.intercept_)
    print(model.get_params())
    print(model.score(data_X, data_y))  # R^2 coefficient of determination


def normalization():
    # feature scaling
    a = np.array([[10, 2.7, 3.6],
                  [-100, 5, -2],
                  [120, 20, 40]], dtype=np.float64)
    print(a)
    print(preprocessing.scale(a))

    X, y = make_classification(n_samples=300, n_features=2, n_redundant=0, n_informative=2,
                               random_state=22, n_clusters_per_class=1, scale=100)
    # plt.scatter(X[:, 0], X[:, 1], c=y)
    # plt.show()

    # X = preprocessing.minmax_scale(X, feature_range=(-1, 1))
    # X = preprocessing.scale(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)
    clf = SVC()
    clf.fit(X_train, y_train)
    print(clf.score(X_test, y_test))


def neural_network():
    # accuracy R2 score F1 score. overfit.
    pass


def cross_validation():
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=4)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    # y_pred = knn.predict(X_test)
    print(knn.score(X_test, y_test))

    knn = KNeighborsClassifier(n_neighbors=5)
    scores = cross_val_score(knn, X, y, cv=5, scoring='accuracy')
    print(scores.mean())

    k_range = range(1, 31)
    k_scores = []
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        loss = -cross_val_score(knn, X, y, cv=10, scoring='mean_squared_error')  # for regression
        # scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')  # for classification

        k_scores.append(loss.mean())

    plt.plot(k_range, k_scores)
    plt.xlabel('Value of K for KNN')
    plt.ylabel('Cross-Validated Accuracy')
    plt.show()


def cross_validation2():  ##overfit
    digits = datasets.load_digits()
    X = digits.data
    y = digits.target
    train_sizes, train_loss, test_loss = learning_curve(SVC(gamma=0.001), X, y, cv=10, scoring='mean_squared_error',
                                                        train_sizes=[0.1, 0.25, 0.5, 0.75, 1])

    train_loss_mean = -np.mean(train_loss, axis=1)
    test_loss_mean = -np.mean(test_loss, axis=1)

    plt.plot(train_sizes, train_loss_mean, 'o-', color='r', label="Training")
    plt.plot(train_sizes, test_loss_mean, 'o-', color='g', label="Cross-validation  ")

    plt.xlabel("Training examples")
    plt.ylabel("Training examples")
    plt.legend(loc='best')
    plt.show()


def cross_validation3():
    digits = datasets.load_digits()
    X = digits.data
    y = digits.target
    param_range = np.logspace(-6, -2.3, 5)
    train_loss, test_loss = validation_curve(
        SVC(), X, y, param_name='gamma', param_range=param_range, cv=10,
        scoring='mean_squared_error')
    train_loss_mean = -np.mean(train_loss, axis=1)
    test_loss_mean = -np.mean(test_loss, axis=1)

    plt.plot(param_range, train_loss_mean, 'o-', color="r",
             label="Training")
    plt.plot(param_range, test_loss_mean, 'o-', color="g",
             label="Cross-validation")

    plt.xlabel("gamma")
    plt.ylabel("Loss")
    plt.legend(loc="best")
    plt.show()


def save_model():
    clf = SVC()
    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    clf.fit(X, y)
    # method pickle 存放数据
    # pickle_file = approot.get_root('clf.pickle')

    # with open(pickle_file, 'wb') as f:
    #     pickle.dump(clf, f)

    # with open(pickle_file, 'rb') as f:
    #     clf2 = pickle.load(f)
    #     print(clf2.predict(X[0:1]))

    # method 2:joblib
    pickle_file = approot.get_root('joblib.pickle')
    # joblib.dump(clf,pickle_file)

    clf3 = joblib.load(pickle_file)
    print(clf3.predict(X[0:1]))


if __name__ == '__main__':
    save_model()
