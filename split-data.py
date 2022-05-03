import os
import random
import shutil


data_path = "big_data/data"
conv_path = "big_data/convdata"

save_data_path = "128/data"
save_conv_path = "128/convdata"

count = 0

for file in os.listdir(conv_path):
    if "strands" in file and random.random() <= 0.4:
        shutil.copyfile(
            os.path.join(conv_path, file), os.path.join(save_conv_path, file)
        )
        name = file.split(".")[0]
        for data in os.listdir(data_path):
            if name in data:
                shutil.copyfile(
                    os.path.join(data_path, data), os.path.join(save_data_path, data)
                )
        if count == 1000:
            break
        count += 1
