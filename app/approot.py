import os


def get_dataset(filename):
    return os.path.dirname(os.path.abspath(__file__)) + os.sep + 'dataset' + os.sep + filename


def get_picture(filename):
    return os.path.dirname(os.path.abspath(__file__)) + os.sep + 'picture' + os.sep + filename


if __name__ == '__main__':
    print(get_dataset())
    print(__file__)
