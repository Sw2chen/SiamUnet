import cv2
import numpy as np
import os

class Dataset:
    """The dataset for siamese model.
    """
    def __init__(
            self, 
            ids,
            images_dir, 
            masks_dir, 
            image_full_dir
    ):
        self.ids = ids
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.image_full_fps = [os.path.join(image_full_dir, image_id.split('/')[0])+'.jpg' for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]
    
    def __getitem__(self, i):
        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image_full = cv2.imread(self.image_full_fps[i])
        image_full = cv2.cvtColor(image_full, cv2.COLOR_BGR2RGB)
        image_full = cv2.resize(image_full, (256,256), interpolation=cv2.INTER_AREA)

        mask = cv2.imread(self.masks_fps[i], 0)
        mask[mask>0] = 255
        mask = mask.astype('float')
        
        # add background if mask is not binary
        if mask.shape[-1] != 1:
            background = 1 - mask.sum(axis=-1, keepdims=True)
            mask = np.concatenate((mask, background), axis=-1)
            
        return (image.astype(np.float32), mask.astype(np.float32), image_full.astype(np.float32))
        
    def __len__(self):
        return len(self.ids)