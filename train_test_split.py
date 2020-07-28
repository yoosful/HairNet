import os
import random
# strands00002_00134_10000_v1
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
# with open('data/index/list.txt', 'w') as wf:
#     wf.write('\n'.join(list_txt) + '\n')
# with open('data/index/train.txt', 'w') as wf:
#     wf.write('\n'.join(train_txt) + '\n')
# with open('data/index/test.txt', 'w') as wf:
#     wf.write('\n'.join(test_txt) + '\n')