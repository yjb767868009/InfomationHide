import csv
import random

for s in ['x', 'y']:
    file_name = 'data2/' + s + '.csv'
    for x_1 in range(2, 3):
        for x_2 in range(x_1 + 1, 8):
            for x_3 in range(x_2 + 1, 8):
                change_cols = [x_1, x_2, x_3]
                for i in range(5):

                    out_file = 'data3/' + s + '/3/' + ''.join(str(x) for x in change_cols) + '-' + str(i) + '.csv'
                    data = []
                    with open(file_name) as file:
                        file_csv = csv.reader(file)
                        for row in file_csv:
                            # print(row)

                            for col in change_cols:
                                row[col] = int(row[col]) + random.randint(-5, 5)
                            # print(row)
                            data.append(row)

                    with open(out_file, 'w') as file:
                        csv_writer = csv.writer(file)
                        csv_writer.writerows(data)
