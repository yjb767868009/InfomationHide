import os

file_path = os.path.join('data3', 'a', '2.txt')

with open(file_path)as file:
    lines = file.readlines()
    for line in lines:
        s = line.strip().split(' ')
        label = s[0][:2]
        data = s[1]
        print(label, data)
