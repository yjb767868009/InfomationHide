import csv
import random
import numpy as np
import time


class FileManager(object):

    def __init__(self):
        self.len = 0
        self.all_data = []
        self.change_data = []
        self.message = None
        self.change_data_index = None

    def find_change_columns(self):
        l = []
        for _ in range(int(self.len / 10)):
            x = self.all_data[random.randint(0, self.len - 1)][2:]
            l.append([int(a) for a in x])
        l = np.array(l)
        v = np.var(l, axis=0)
        t = np.argmax(v)
        return t + 2

    def read_message(self, file_name):
        message = []
        with open(file_name) as file:
            lines = file.readlines()
            for line in lines:
                line = line.strip('\n')
                for x in line:
                    message.append(ord(x))
        message.append(127)
        self.message = message
        # print('message')
        # print(message)
        return message

    def read_file(self, file_name):
        data = []
        with open(file_name) as file:
            file_csv = csv.reader(file)
            for row in file_csv:
                data.append(row)
                self.len += 1
        self.all_data = np.array(data)

        start_time = time.clock()
        self.change_data_index = self.find_change_columns()
        end_time = time.clock()
        print('find time')
        print(end_time - start_time)

        self.change_data = [int(x) for x in self.all_data[:, self.change_data_index]]
        # print('data')
        # print(self.all_data[:10])
        # print('change_data')
        # print(self.change_data[:30])

    def hide_data(self, new_data):
        for i in range(self.len):
            self.all_data[i][self.change_data_index] = new_data[i]
        # print('hide_data')
        # print(self.all_data[:10])

    def out_file(self, file_name):
        with open(file_name, 'w') as file:
            csv_writer = csv.writer(file)
            csv_writer.writerows(self.all_data)


if __name__ == '__main__':
    f = FileManager()
    f.read_file('data1/1.csv')
    f.read_message('message.txt')
    print(f.message)
