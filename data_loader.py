import cv2
from tensorflow import keras as keras
import numpy as np


class Dataloader(keras.utils.Sequence):
    """Load data from dataset and form batches
    
    Args:
        dataset: instance of Dataset class for image loading and preprocessing.
        batch_size: Integet number of images in batch.
        shuffle: Boolean, if `True` shuffle image indexes each epoch.
    """
    
    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(dataset))
        self.on_epoch_end()

    def __getitem__(self, i):
        
        # collect batch data
        start = i * self.batch_size
        stop = (i + 1) * self.batch_size
        x_data = []
        x2_data = []
        y_data = []
        for j in range(start, stop):
            x_data.append(self.dataset[j][0])
            y_data.append(self.dataset[j][1])
            x2_data.append(self.dataset[j][2])
        x_batch = [np.array(x_data), np.array(x2_data)]
        y_batch = np.array(y_data)
        
        return x_batch,y_batch
    
    def __len__(self):
        return len(self.indexes) // self.batch_size
    
    def on_epoch_end(self):
        if self.shuffle:
            self.indexes = np.random.permutation(self.indexes)