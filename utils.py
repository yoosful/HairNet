import os
import random


def train_test_split():
    files = os.listdir("convdata")
    list_txt = []
    train_txt = []
    test_txt = []

    for file in files:
        data = file.split(".")[0] + "_v0"
        list_txt.append(data)
        if random.random() > 0.7:
            test_txt.append(data)
        else:
            train_txt.append(data)
        data = file.split(".")[0] + "_v1"
        list_txt.append(data)
        if random.random() > 0.7:
            test_txt.append(data)
        else:
            train_txt.append(data)


def get_list_name():
    # Append-adds at last
    list = open("list.txt", "a")  # append mode
    files = os.listdir("data")
    for file in files:
        if "txt" in file:
            list.write(file.split(".")[0] + "\n")

    list.close()
