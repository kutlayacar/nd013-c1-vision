import argparse
import glob
import os
import random

import numpy as np

from utils import get_module_logger

import shutil

def split(data_dir):
    """
    Create three splits from the processed records. The files should be moved to new folders in the 
    same directory. This folder should be named train, val and test.

    args:
        - data_dir [str]: data directory, /home/workspace/data/waymo
    """
    
    # TODO: Split the data present in `/home/workspace/data/waymo/training_and_validation` into train and val sets.
    # You should move the files rather than copy because of space limitations in the workspace.
    
    try:
        files = [filename for filename in glob.glob('/home/workspace/data/waymo/training_and_validation/*.tfrecord')]
        
    except Exception as err:
        print("unable to access file")
        
    np.random.shuffle(files)
    
    train_file, val_file, test_file = np.split(files, [int(.75*len(files)), int(.9*len(files))])
    
    train = os.path.join(data_dir, 'train')
    os.makedirs(train, exist_ok=True)
        
    for file in train_file:
        shutil.move(file, train)
        
    val = os.path.join(data_dir, 'val')
    os.makedirs(val, exist_ok=True)
        
    for file in val_file:
        shutil.move(file, val)
        
    test = os.path.join(data_dir, 'test')
    os.makedirs(test, exist_ok=True)
    
    for file in test_file:
        shutil.move(file, test)

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='Split data into training / validation / testing')
    parser.add_argument('--data_dir', required=True,
                        help='data directory')
    args = parser.parse_args()

    logger = get_module_logger(__name__)
    logger.info('Creating splits...')
    split(args.data_dir)