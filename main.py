import datetime
import torch
import numpy as np
import monai
import os
import logging
from monai.metrics import DiceMetric
import timm.scheduler
from load_data import build_dataset, get_train_val_test_splits
from transforms import build_train_transforms, build_val_transforms
from train import (
    train, 
    test_model,
)
import matplotlib.pyplot as plt
import pickle

###########################################################
### PARAMS                                              ###
###########################################################

EXPERIMENT_RUN_NAME = "Tether_Screw_Segmentation_SquareNormalize"
COMMENTS = """
Adding several random transforms. More frequent LR restarts.
Whole dataset. Final training set. Added figure. Normalization
"""

DATA_DIR = "/scratch/Datasets/Tether_Angle_Calculator/segmented_screws/"
CACHE_DIR = "/scratch/Datasets/Tether_Angle_Calculator/cache/"

RANDOM_SEED = 3222023

#DATASET = "/research/projects/m274639_Kellen/tether_angle_calculator/output/dataset.pkl"
DATASET = None

IMG_HEIGHT = 1024
IMG_WIDTH = 1024

TRAIN_BATCH_SIZE = 2
VAL_BATCH_SIZE = 1
TEST_BATCH_SIZE = 1

TRAIN_WORKERS = 16
VAL_WORKERS = 16
TEST_WORKERS = 16

TRAIN_SHUFFLE = True

MAX_EPOCHS = 2000
INITIAL_LR = 4.5e-5

TRAIN = True
TEST = True
TEST_MODEL = None

def main():

    ###########################################################
    ### INITIALIZE LOGGING                                  ###
    ###########################################################

    ## Initialize logging
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        filename="logs/" + EXPERIMENT_RUN_NAME + ".log",
        encoding="utf-8",
        format="%(message)s",
        level=logging.INFO,
        filemode="w",
    )
    logging.info("\n" + "".join(["="] * 80) + "\nCONFIGURATION\n")
    logging.info(f"Data Directory: {DATA_DIR}")
    logging.info(f"Random Seed: {RANDOM_SEED}")
    logging.info(f"Image Dimensions: {IMG_HEIGHT}x{IMG_WIDTH}")
    logging.info(
        f"Batch Sizes:\n\tTrain: {TRAIN_BATCH_SIZE}\n\tValidation: {VAL_BATCH_SIZE}\n\tTest: {TEST_BATCH_SIZE}"
    )
    logging.info(
        f"Number of Workers:\n\tTrain: {TRAIN_WORKERS}\n\tValidation: {VAL_WORKERS}\n\tTest: {TEST_WORKERS}"
    )
    logging.info(f"Experiment Name: {EXPERIMENT_RUN_NAME}")
    logging.info(f"Epochs: {MAX_EPOCHS}")
    logging.info(f"Initial LR: {INITIAL_LR}")
    logging.info(f"Experiment Comments:\n\n{COMMENTS}")
    np.random.seed(RANDOM_SEED)
    
    logging.info("\n" + "".join(["="] * 80) + "\nLOADING DATA\n")    

    ###########################################################
    ### LOAD DATA AND TRANSFORMS                            ###
    ###########################################################

    if DATASET:
        with open(DATASET, 'rb') as handle:
            (train_list, val_list, test_list) = pickle.load(handle)
            
    ## Get and organize data
    else:
        data_dictionary = build_dataset(data_dir=DATA_DIR)
        train_list, val_list, test_list = get_train_val_test_splits(data_dict=data_dictionary)
        dataset = (train_list, val_list, test_list)
        with open('output/dataset.pkl', 'wb') as handle:
            pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    logging.info(f"Training count: {len(train_list)}")
    logging.info(f"Validation count: {len(val_list)}")
    logging.info(f"Test count: {len(test_list)}")
    
    ## Get transforms
    train_transforms = build_train_transforms(
        img_height=IMG_HEIGHT, img_width=IMG_WIDTH
    )
    val_transforms = build_val_transforms(img_height=IMG_HEIGHT, img_width=IMG_WIDTH)

    ## Set up datasets and loaders
    train_ds = monai.data.PersistentDataset(
        data=train_list, transform=train_transforms, cache_dir=CACHE_DIR
    )
    train_loader = monai.data.DataLoader(
        train_ds,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=TRAIN_SHUFFLE,
        num_workers=TRAIN_WORKERS,
    )
    val_ds = monai.data.PersistentDataset(
        data=val_list, transform=val_transforms, cache_dir=CACHE_DIR
    )
    val_loader = monai.data.DataLoader(
        val_ds,
        batch_size=VAL_BATCH_SIZE,
        num_workers=VAL_WORKERS,
    )
    test_ds = monai.data.PersistentDataset(
        data=test_list, transform=val_transforms, cache_dir=CACHE_DIR
    )
    test_loader = monai.data.DataLoader(
        test_ds,
        batch_size=TEST_BATCH_SIZE,
        num_workers=TEST_WORKERS,
    )

    ###########################################################
    ### INITIALIZE MODEL AND LOSS                           ###
    ###########################################################

    ## Initialize models
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    model = monai.networks.nets.UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=2,
        channels=(
            16, 16, 16, 
            32, 32, 32, 
            64, 64, 64, 
            128, 128, 128, 
            512, 512, 512, 
            1024, 1024, 1024,
            2048
        ),
        strides=(
            1, 1, 2,
            1, 1, 2, 
            1, 1, 2,
            1, 1, 2,
            1, 1, 2,
            1, 1, 2
        ),
        dropout=0.25,
        num_res_units=1,
        norm=monai.networks.layers.Norm.BATCH,
    ).to(device)

    model_create_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    model_name = EXPERIMENT_RUN_NAME + model_create_time

    ## Loss functions and optimizers
    loss_function = monai.losses.DiceFocalLoss(to_onehot_y=True, softmax=True, focal_weight=0.2)
    optimizer = torch.optim.Adam(model.parameters(), INITIAL_LR)
    metric = DiceMetric(include_background=False)
    scheduler = timm.scheduler.CosineLRScheduler(
        optimizer=optimizer, t_initial = 25, t_in_epochs=True,
        cycle_mul = 2, cycle_decay = 0.9, cycle_limit = 20,
    )

    ###########################################################
    ### TRAIN                                               ###
    ###########################################################

    if TRAIN:
        
        logging.info("\n" + "".join(["="] * 80) + "\nTRAINING\n")
        ## Tensorboard
        tensorboard_dir = os.path.join("runs", model_name)

        ## Training Loop
        train(
            model=model,
            model_name=model_name,
            train_loader=train_loader,
            val_loader=val_loader,
            loss_function=loss_function,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            log_dir=tensorboard_dir,
            metric=metric,
            epochs=MAX_EPOCHS,
        )

    ###########################################################
    ### TEST                                                ###
    ###########################################################

    if TEST:
        
        ## Log test section
        logging.info("\n" + "".join(["="] * 80) + "\nTESTING RESULTS\n")
        logging.info(
            "NOTE: Using best model on validation set - not most recent model."
        )
        
        ## Load the desired model
        if TEST_MODEL:
            model.load_state_dict(torch.load(os.path.join("models", TEST_MODEL)))            
        else:
            model.load_state_dict(torch.load(os.path.join("models", model_name + ".pth")))
            
        ## Run the testing loop
        test_df = test_model(
            model=model,
            loader=test_loader,
            device=device,
        )
        logging.info(f"Mean Dice on Test Set: {test_df['dice'].mean():0.4f}")
        logging.info(f"StDev Dice on Test Set: {test_df['dice'].std():0.4f}")
        
        test_df.to_pickle(os.path.join("output", EXPERIMENT_RUN_NAME + "-TEST-OUT.pkl"))


if __name__ == "__main__":
    main()