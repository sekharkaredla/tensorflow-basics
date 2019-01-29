import cv2
import numpy as np
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle


def one_hot_encode(labels):
    # if it is 'label1' it will be represented as [1, 0, 0] in one_hot_encoder. At a time only one output is hot
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encode = np.zeros((n_labels,n_unique_labels))
    one_hot_encode[range(n_labels),labels] = 1
    one_hot_encode.astype(float)
    return one_hot_encode

def split_dataset_with_ratio(ratio):
    data = []
    labels = []
    flower_data_folder = "/Users/roshni/Documents/CODE/flowers_dataset/reshaped/"
    for each_flower_image in glob.glob(flower_data_folder+"*.jpg"):
        image_data = cv2.imread(each_flower_image)
        image_data = image_data.astype(float)
        image_label = each_flower_image.split('/')[-1].split('_')[0]

        image_data = (image_data - image_data.min())/(image_data.max() - image_data.min())
        image_data = image_data.reshape(-1,1)
        data.append(image_data)
        labels.append(image_label)
        print image_label
        print image_data

    data, labels = shuffle(data, labels, random_state=1)
    labelEncoder = LabelEncoder()
    labelEncoder.fit(labels)
    labels = labelEncoder.transform(labels)
    labels = one_hot_encode(labels=labels)
    data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size=ratio, random_state=42)
    return data_train, labels_train, data_test, labels_test
