import numpy as np
import math
import random

matrix = np.zeros([9, 9])


# 生成一个随机的数组
def get_random_unit():
    _num_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    random.shuffle(_num_list)
    return _num_list


def get_row(row):
    row_arr = []
    for v in matrix[row]:
        if v == 0:
            continue
        row_arr.append(v)
    return row_arr


def get_col(col):
    col_arr = []
    for i in range(9):
        val = matrix[i][col]
        if val == 0:
            continue
        col_arr.append(matrix[i][col])
    return col_arr


def get_block(num):
    col_arr = []
    seq = num % 3
    col_end = 9 if seq == 0 else seq * 3
    row_end = int(math.ceil(num / 3) * 3)
    for i in range(row_end - 3, row_end):
        for j in range(col_end - 3, col_end):
            val = matrix[i][j]
            if val != 0:
                col_arr.append(matrix[i][j])
    return col_arr


def get_block_seq(row, col):
    col_seq = int(math.ceil((col + 0.1) / 3))
    row_seq = int(math.ceil((row + 0.1) / 3))
    return 3 * (row_seq - 1) + col_seq


def get_enable_arr(row, col):
    avail_arr = get_random_unit()
    seq = get_block_seq(row, col)
    block = get_block(seq)
    row = get_row(row)
    col = get_col(col)
    unable_arr = list(set(block + row + col))
    for v in unable_arr:
        if v in avail_arr:
            avail_arr.remove(v)
    return avail_arr


def create_base_matrix():
    num_list = get_random_unit()
    for row in range(3):
        for col in range(3):
            matrix[row][col] = num_list.pop(0)

    num_list = get_random_unit()
    for row in range(3, 6):
        for col in range(3, 6):
            matrix[row][col] = num_list.pop(0)

    num_list = get_random_unit()
    for row in range(6, 9):
        for col in range(6, 9):
            matrix[row][col] = num_list.pop(0)

    box_list = []
    for row in range(9):
        for col in range(9):
            if matrix[row][col] == 0:
                box_list.append({'row': row, 'col': col})

    can_num = {}
    count = 0
    i = 0
    while i < len(box_list):
        count += 1
        position = box_list[i]
        row = position['row']
        col = position['col']
        key = '%dx%d' % (row, col)
        if key in can_num:
            enable_arr = can_num[key]
        else:
            enable_arr = get_enable_arr(row, col)
            can_num[key] = enable_arr

        if len(enable_arr) <= 0:
            i -= 1
            if key in can_num:
                del (can_num[key])
            matrix[row][col] = 0
            continue
        else:
            matrix[row][col] = enable_arr.pop()
            i += 1

    return matrix - 1


def create_matrix():
    matrix = create_base_matrix()
    for _ in range(7):
        matrix = np.vstack((matrix, matrix))
        matrix = np.hstack((matrix, matrix))
    return matrix


if __name__ == '__main__':
    matrix = create_base_matrix()
    print(matrix)
