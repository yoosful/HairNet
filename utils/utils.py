import os
import random

path = "./data/index/"


def train_test_split():
    get_list_name()

    train = open(path + "train.txt", "a")
    test = open(path + "test.txt", "a")

    with open(path + "list.txt") as f:
        lines = f.readlines()

    for line in lines:
        if random.random() <= 0.7 and "strands" in line:
            train.write(line)
        else:
            test.write(line)

    train.close()
    test.close()


def get_list_name():
    # Append-adds at last
    list = open(path + "list.txt", "a")  # append mode
    files = os.listdir("data")
    for file in files:
        if "txt" in file:
            list.write(file.split(".")[0] + "\n")

    list.close()


train_test_split()
