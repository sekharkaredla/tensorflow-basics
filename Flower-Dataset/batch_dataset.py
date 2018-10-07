from read_data import split_dataset_with_ratio

class BatchDataset:
    def __init__(self,ratio):
        data = split_dataset_with_ratio(ratio)
        self.data_train = data[0]
        self.labels_train = data[1]
        self.data_test = data[2]
        self.labels_test = data[3]
        self.counter = 0
        self.data_present = True

    def get_next(self):
        if self.counter+20<len(self.data_train):
            return (self.data_train[self.counter:self.counter+20],self.labels_train[self.counter:self.counter+20])
            self.counter += 20
        else:
            self.data_present = False
            return (self.data_train[self.counter:],self.labels_train[self.counter:])

    def get_test_data(self):
        return (self.data_test,self.labels_test)

    def get_details(self):
        print len(self.data_train), len(self.labels_train), len(self.data_test), len(self.labels_test)

