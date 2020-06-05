from KeyMatrix import create_matrix
from FileManager import FileManager
from InfoHide import hide_info, decode_info
import time

if __name__ == '__main__':
    matrix_start_time = time.clock()
    matrix = create_matrix()
    matrix_end_time = time.clock()
    print('matrix time')
    print(matrix_end_time - matrix_start_time)

    start_time = time.clock()
    f = FileManager()
    f.read_file('data2/a.csv')
    f.read_message('message.txt')
    end_time = time.clock()
    print('read time')
    print(end_time - start_time)

    hide_start_time = time.clock()
    data = hide_info(f.change_data, f.message, matrix)

    hide_end_time = time.clock()
    print('hide time')
    print(hide_end_time - hide_start_time)
    f.hide_data(data)
    f.out_file('data2/a-1.csv')

    start_time = time.clock()
    decode_info(data, matrix)
    end_time = time.clock()
    print('decode time')
    print(end_time - start_time)
