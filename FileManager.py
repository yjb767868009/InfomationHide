import csv


class FileManager(object):

    def __init__(self):
        self.all_data = []
        self.change_data = []
        self.message = None

    def read_message(self, file_name):
        message = []
        with open(file_name) as file:
            lines = file.readlines()
            # for line in lines:
            #     line = line.strip('\n')
            #     items = line.split(';')
            #     if len(items) == 1 and not len(items[0]):
            #         continue
            #     message.extend(items)
            for line in lines:
                line = line.strip('\n')
                for x in line:
                    message.append(ord(x))
        self.message = message
        return message

    def read_file(self, file_name):
        self.all_data = []
        self.change_data = []
        with open(file_name) as file:
            file_csv = csv.reader(file)
            for row in file_csv:
                self.all_data.append(row)
                self.change_data.append(row[1])

    def hide_data(self, new_data):
        pass

    def out_file(self, file_name):
        pass


# todo


if __name__ == '__main__':
    f = FileManager()
    f.read_message('message.txt')
    print(f.message)
