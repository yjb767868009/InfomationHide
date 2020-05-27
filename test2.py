import random
import math
import numpy as np
import re
import time

start_time = time.clock()
matrix = []


# generate a random matrix （9x9）
def get_random_unit():
    _num_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    random.shuffle(_num_list)
    return _num_list


def print_grid(arr):
    for i in range(9):
        print(arr[i])


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


start_time1 = time.clock()
can_num = {}
count = 0
for i in range(9):
    matrix.append([0] * 9)

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

for row in range(0, 9):
    for col in range(0, 9):
        matrix[row][col] = matrix[row][col] - 1

print(matrix)
end_time1 = time.clock()
time1 = end_time1 - start_time1
print(time1)

start_time2 = time.clock()
with open('ascii.txt', 'r') as file:
    lines = file.readlines()
    list_arr = []
    for line in lines:
        line = line.strip('\n')
        items = line.split(';')
        if len(items) == 1 and not len(items[0]):
            continue
        list_arr.extend(items)

list_arr2 = [0] * len(list_arr)

for i in range(len(list_arr)):
    list_arr2[i] = int(list_arr[i]) % 9 \
                   + int(int(list_arr[i]) / 9) % 9 * 10 \
                   + int(int(int(list_arr[i]) / 9) / 9) % 9 * 100

list_arr3 = [0] * len(list_arr)

for i in range(0, int(len(list_arr2) / 2)):
    row = list_arr2[2 * i]
    col = list_arr2[2 * i + 1]
    x = y = 0
    if row <= 142:
        for r in range(row, row + 9):
            if matrix[r][col] == row % 10:
                x = r
    elif row > 142:
        for r in range(142, 151):
            if matrix[r][col] == row % 10:
                x = r

    if col <= 142:
        for c in range(col, col + 9):
            if matrix[row][c] == col % 10:
                y = c
    elif col > 142:
        for c in range(142, 151):
            if matrix[row][c] == col % 10:
                y = c

    list_arr3[2 * i] = int(x)
    list_arr3[2 * i + 1] = int(y)
sum = 0
for i in range(len(list_arr3)):
    sum += math.sqrt((list_arr3[i] - list_arr2[i]) ** 2)

for i in range(len(list_arr3)):
    list_arr3[i] = round(round(list_arr3[i], 4) / 10000, 4)
    list_arr2[i] = round(round(list_arr2[i], 4) / 10000, 4)

sum = 0
for i in range(len(list_arr3)):
    sum += (list_arr2[i] - list_arr3[i]) ** 2

a = np.mat(list_arr3)
lines = []
test_freebvh = open('test_freebvh.bvh', 'r')
lines = test_freebvh.readlines()

with open("bvh_data.rtf", "w") as fun:
    for i in range(68):
        fun.write(lines[340 + i])

with open('bvh_data.rtf', 'r') as file:
    list_arr4 = file.readlines()
    l = len(list_arr4)
    for i in range(l):
        list_arr4[i] = list_arr4[i].strip()
        list_arr4[i] = list_arr4[i].strip('[]')
        list_arr4[i] = list_arr4[i].split(' ')
        list_arr4[i] = list(map(float, list_arr4[i]))


def update_decimal(a, b):
    sign = True if '-' in str(a) else False
    a = [int(num) for num in str(a).split('.')]
    b = [int(num) for num in str(b).split('.')]
    return '-%d.%d' % (a[0], b[1]) if sign is True and not a[0] else '%d.%d' % (a[0], b[1])


def generate_result(input, output_filepath):
    index = [index for index, item in enumerate(input) if index % 13 == 0]
    splited_list = [input[i:i + 13] for i in index]
    for line in range(len(splited_list)):
        for index, num in enumerate(splited_list[line][0:13]):
            list_arr4[line][index] = update_decimal(list_arr4[line][index], splited_list[line][index])

    content = ''.join(lines)
    head = re.findall(r'RARCHY[\s\S]*MOTION\nFrames:.+\nFrame Time:.+\n', content)
    if not len(head):
        raise Exception('file format changed')
    head = head[0]
    replaced_content = '\n'.join([' '.join(line) for line in list_arr4])
    test_freebvh.close()
    with open(output_filepath, 'w') as file:
        file.write(head + replaced_content)


print(list_arr2)
print(list_arr3)
for i in range(len(list_arr2)):
    if list_arr2 == list_arr3:
        pass
    else:
        print(10000 * abs(list_arr2[i] - list_arr3[i]))
