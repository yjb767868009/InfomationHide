import csv
import os

dir_path = 'data2'
file_name = os.path.join(dir_path, 'Data0315Mark.csv')
out_file = os.path.join(dir_path, 'c.csv')
data = []
with open(file_name) as file:
    file_csv = csv.reader(file)
    for row in file_csv:
        if row[8] == '20' and row[9] == '9':
            print(row)
            data.append(row[:8])

    with open(out_file, 'w') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerows(data)
