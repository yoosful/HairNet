import os
import random

def train_test_split():
    files = os.listdir("convdata")
    length = len(files)
    list_txt = []
    train_txt = []
    test_txt = []

    for file in files:
        # print(file.split('.')[0]+'_v0')
        data = file.split('.')[0]+'_v0'
        list_txt.append(data)
        if random.random() > 0.7:
            test_txt.append(data)
        else:
            train_txt.append(data)
        data = file.split('.')[0]+'_v1'
        list_txt.append(data)
        if random.random() > 0.7:
            test_txt.append(data)
        else:
            train_txt.append(data)
