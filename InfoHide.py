def ten_2_nine(x):
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


def nine_2_ten(x):
    a = x % 10
    b = int(x / 10) % 10 * 9
    c = int(x / 100) % 10 * 9 * 9
    y = a + b + c
    return y


def message_2_stream(message):
    stream = []
    message_nine = [ten_2_nine(x) for x in message]
    for x in message_nine:
        stream.append(int(x / 100))
        x = x % 100
        stream.append(int(x / 10))
        x = x % 10
        stream.append(x)
    # print('stream')
    # print(stream)
    return stream


def ceh(a, x, y, matrix):
    l = max(y - 4, 0)
    for i in range(l, l + 9):
        if matrix[x][i] == a:
            return x, i
    return None


def cev(a, x, y, matrix):
    l = max(x - 4, 0)
    for i in range(l, l + 9):
        if matrix[i][y] == a:
            return i, y
    return None


def ceb(a, x, y, matrix):
    x_1 = int(x / 3) * 3
    y_1 = int(y / 3) * 3
    for i in range(0, 3):
        for j in range(0, 3):
            if matrix[x_1 + i][y_1 + j] == a:
                return x_1 + i, y_1 + j
    return None


def min_x_y(x, y, h, v, b):
    h_d = abs(x - h[0]) + abs(y - h[1])
    v_d = abs(x - v[0]) + abs(y - v[1])
    b_d = abs(x - b[0]) + abs(y - b[1])
    min_d = min(h_d, v_d, b_d)
    if h_d == min_d:
        return h[0], h[1]
    if v_d == min_d:
        return v[0], v[1]
    if b_d == min_d:
        return b[0], b[1]


def hide_info(data, message, matrix):
    stream = message_2_stream(message)
    i = 0
    if len(stream) * 2 > len(data):
        exit('not enough data')
    for a in stream:
        x = data[i]
        y = data[i + 1]
        h = ceh(a, x, y, matrix)
        v = cev(a, x, y, matrix)
        b = ceb(a, x, y, matrix)
        new_x, new_y = min_x_y(x, y, h, v, b)
        data[i] = new_x
        data[i + 1] = new_y
        i += 2
    # print('new_data')
    # print(data[:30])
    return data


def decode_message(stream):
    x = stream[0] * 100 + stream[1] * 10 + stream[2]
    y = nine_2_ten(x)
    return y


def decode_info(data, matrix):
    stream = []
    message = []
    i = 0
    k = 0
    while True:
        x = data[i]
        y = data[i + 1]
        stream.append(matrix[x][y])
        k += 1
        if k == 3:
            a = decode_message(stream)
            if a == 127:
                break
            message.append(a)
            stream = []
            k = 0
        i += 2
    print('message')
    print(message)


if __name__ == '__main__':
    message = [104, 101, 108, 108, 111, 32, 119, 111, 114, 108, 100]
    message_2_stream(message)
