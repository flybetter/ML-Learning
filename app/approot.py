import os


def get_root(filename):
    return os.path.dirname(os.path.abspath(__file__)) + os.sep + 'dataset' + os.sep + filename


if __name__ == '__main__':
    print(get_root())
    print(__file__)
