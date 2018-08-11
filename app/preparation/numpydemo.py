import numpy as np

'''
之前学习过python，在跟着github上面的大神学习的时候，发现自己在numpy、pandas、matlabplot上面还有很多的不足。
顺便复习和学习一下。
'''


def first():
    array = np.array([[1, 2, 3], [4, 5, 6]])
    print(array)
    print("number of dim:", array.ndim)
    print("shape:", array.shape)
    print("size", array.size)


def second():
    a = np.array([2, 23, 4], dtype=int)
    print(a.dtype)
    b = np.zeros((3, 2))
    print(b)
    c = np.ones((3, 2))
    print(c)
    d = np.arange(10, 20, 2)
    print(d)
    e = np.arange(12).reshape((3, 4))
    print(e)
    f = np.linspace(1, 10, 6).reshape((2, 3))
    print(f)


def compute():
    a = np.array([10, 20, 30, 40])
    b = np.arange(4)
    temp = a ** 2
    print(temp)
    temp = np.sin(a)
    temp = np.cos(a)
    temp = np.tan(a)
    c = a - b
    print(c)
    print(b < 3)
    c_dot = np.dot(a, b)
    print(c_dot)
    c_dot_2 = a.dot(b)
    print(c_dot_2)

    a = np.random.random((2, 3))
    print(a)
    print(a.max(a, axis=0))
    print(a.min(a, axis=1))
    print(a.max())


def compute2():
    A = np.arange(2, 14).reshape((3, 4));

    print(np.argmin(A))
    print(np.argmax(A))
    print(np.mean(A))
    print(np.average(A))
    print(np.median(A))
    print(np.cumsum(A))
    print(np.diff(A))
    print(np.nonzero(A))
    print(np.sort(A))
    print(np.transpose(A))
    print(A.T.dot(A))
    print(np.clip(A, 5, 9))
    print(A)


def index():
    A = np.arange(3, 15).reshape((3, 4))
    print(A)
    print(A[2, 1])
    print(A[2, :])
    print(A[2, 0:2])

    for row in A:
        print(row)

    for row in A.T:
        print(row)

    print(A.flatten())


def combine():
    A = np.array([1, 1, 1])
    B = np.array([2, 2, 2])
    C = np.vstack((A, B))
    print(C)  # vertiacl stack
    D = np.hstack((A, B))  # horizontal stack
    print(D)
    print(D.size)
    E = D.reshape(D.size, 1)
    print(E)

    A = np.array([1, 1, 1, ]).reshape(-1, 1)
    B = np.array([2, 2, 2]).reshape(-1, 1)
    C = np.concatenate((A, B), axis=1)
    print(C)


def division():
    A = np.arange(12).reshape((3, 4))
    print(A)
    print(np.split(A, 2, axis=1))

    # 不等量的分割
    print(np.array_split(A, 3, axis=1))
    print(np.vsplit(A, 3))
    print(np.hsplit(A, 2))


def copy():
    # 赋值
    a = np.arange(12).reshape((3, 4))
    b = a
    c = a
    # b 就是a
    print(b is a)  # True

    b = a.copy()
    print(b)


if __name__ == '__main__':
    division()
