from KeyMatrix import create_matrix
from FileManager import FileManager


def nine_hex(x):
    y = 0
    i = 0
    while x > 0:
        b = x % 9
        for _ in range(i):
            b *= 10
        y = y + b
        x = int(x / 9)
        i += 1
    return y


if __name__ == '__main__':
    # matrix = create_matrix()
    f = FileManager()
    f.read_file('1.csv')
    f.read_message('message.txt')
    list = f.message
    print(list)
    list_2 = [nine_hex(x) for x in list]
    print(list_2)
