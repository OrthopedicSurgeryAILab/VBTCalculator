import glob
import os
import numpy as np
import logging

'''
This is a set of functions to load the data and split into
train, val, and test sets.
'''

def build_dataset(data_dir):
    image_dirs = glob.glob(os.path.join(data_dir, "*"), recursive=True)
    dataset = []
    
    for image in image_dirs:
        
        image_dict = {}
        basename = os.path.basename(image)
        
        ## Get image file and path
        image_file = os.path.join(image, basename + ".dcm")        
        if os.path.isfile(image_file):
            image_dict["image"] = image_file
            image_dict["image_path"] = image_file
        else:
            logging.info(f"Base DICOM not found in {image}")
            continue
        
        ## Get label file and path
        label_file = glob.glob(os.path.join(image, "*-label.nrrd"))
        try:
            if os.path.isfile(label_file[0]):
                image_dict["label"] = label_file[0]
            else:
                logging.info(f"Screw label file not found in {image}")
                continue
        except IndexError as e:
            logging.info(f"Screw label file not found in {image}")
            continue
        
        if os.path.isfile(os.path.join(image, basename + ".xlsx")):
            image_dict["angle_file"] = os.path.join(image, basename + ".xlsx")
        
        dataset.append(image_dict)
        
    return dataset

def get_train_val_test_splits(data_dict):
    # length = len(data_dict)
    # indices = np.arange(length)
    # np.random.shuffle(indices)

    # test_split = int(test_frac * length)
    # val_split = int(val_frac * length) + test_split
    # test_indices = indices[:test_split]
    # val_indices = indices[test_split:val_split]
    # train_indices = indices[val_split:]

    # train = [data_dict[i] for i in train_indices]
    # val = [data_dict[i] for i in val_indices]
    # test = [data_dict[i] for i in test_indices]
    val_test = []
    train = []
    for case in data_dict:
        if "angle_file" in case:
            val_test.append(case)
        else:
            train.append(case)
    
    length = len(val_test)
    indices = np.arange(length)
    np.random.shuffle(indices)
           
    split = int(0.5 * length)
    test_indices = indices[:split]   
    val_indices = indices[split:]
    val = [val_test[i] for i in val_indices]
    test = [val_test[i] for i in test_indices]

    return train, val, test
