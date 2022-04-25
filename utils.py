import os
import random


def train_test_split():
    list = open("train.txt", "a")
    files = os.listdir("convdata")
    list_txt = []
    train_txt = []
    # test_txt = []

    for file in files:
        data = file.split(".")[0]
        # list_txt.append(data)
        if random.random() > 0.5 and "strands" in data:
            if len(train_txt) >= 1000:
                break
            # train_txt.append(data)
            if random.random() >= 0.5:
                list.write(data + "_v0\n")
            else:
                list.write(data + "_v1\n")

        # if random.random() > 0.7:
        #     test_txt.append(data)
        # else:
        #     train_txt.append(data)

        # data = file.split(".")[0] + "_v1"
        # list_txt.append(data)
        # if random.random() > 0.5:
        #     if len(train_txt) >= 500:
        #         continue
        #     # train_txt.append(data)
        #     list.write(data + "\n")

        # list_txt.append(data)
        # if random.random() > 0.7:
        #     test_txt.append(data)
        # else:
        #     train_txt.append(data)


def get_list_name():
    # Append-adds at last
    list = open("list.txt", "a")  # append mode
    files = os.listdir("data")
    for file in files:
        if "txt" in file:
            list.write(file.split(".")[0] + "\n")

    list.close()


train_test_split()
