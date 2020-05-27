import os
import csv
import random

file_name = 'data2/c.csv'

change_cols = [2, 3, 4, 5, 6, 7]

dir_path = 'data2'

for col in change_cols:
    for i in range(5):
        out_file = 'data3/c/1/' + str(col) + '-' + str(i) + '.csv'
        data = []
        with open(file_name) as file:
            file_csv = csv.reader(file)
            for row in file_csv:
                print(row)
                row[col] = int(row[col]) + random.randint(-5, 5)
                print(row)
                data.append(row)

        with open(out_file, 'w') as file:
            csv_writer = csv.writer(file)
            csv_writer.writerows(data)
