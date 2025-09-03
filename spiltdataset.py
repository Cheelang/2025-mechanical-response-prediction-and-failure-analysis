import os
import random
import shutil

def split_dataset(data_dir, label_dir, val_dir, val_label_dir, test_dir, test_label_dir, train_dir, train_label_dir):

    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(val_label_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(test_label_dir, exist_ok=True)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(train_label_dir, exist_ok=True)


    image_files = os.listdir(data_dir)


    random.shuffle(image_files)


    val_size = int(len(image_files) * 0.1)
    test_size = int(len(image_files) * 0.1)


    for i, image_file in enumerate(image_files):
        label_file = image_file


        if i < val_size:
            shutil.copy(os.path.join(data_dir, image_file), val_dir)
            shutil.copy(os.path.join(label_dir, label_file), val_label_dir)
        elif i < val_size + test_size:
            shutil.copy(os.path.join(data_dir, image_file), test_dir)
            shutil.copy(os.path.join(label_dir, label_file), test_label_dir)
        else:
            shutil.copy(os.path.join(data_dir, image_file), train_dir)
            shutil.copy(os.path.join(label_dir, label_file), train_label_dir)


data_dir = r""
label_dir = r""
val_dir = r""
val_label_dir = r""
test_dir = r""
test_label_dir = r""
train_dir = r""
train_label_dir = r""


split_dataset(data_dir, label_dir, val_dir, val_label_dir, test_dir, test_label_dir, train_dir, train_label_dir)