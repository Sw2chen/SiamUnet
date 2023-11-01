import tensorflow as tf
import segmentation_models as sm
import numpy as np
import cv2
import datetime
from dataset import Dataset
from model import *
from data_loader import Dataloader


BATCH_SIZE = 5
LR = 0.00001
EPOCHS = 40
x_dir = './augmentation_image'
x2_dir = './raw_image'
y_dir = './gt' 
ids_train = np.loadtxt('./ids_for_train.txt', dtype=str)
ids_valid = np.loadtxt('./ids_for_valid.txt', dtype=str)


# ===============================
# =========Compile Model=========
# ===============================
model = siamese_deep_unet(256) 
metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]

dice_loss = sm.losses.DiceLoss()
focal_loss = sm.losses.BinaryFocalLoss()
total_loss = dice_loss + (1 * focal_loss)

model.compile(optimizer=tf.keras.optimizers.Adam(LR), loss=total_loss, metrics=metrics)
# model.load_weights('./model.h5')
model.summary()




# ===============================
# ============Data set===========
# ===============================

# Dataset for train images
train_dataset = Dataset(
    ids_train,
    x_dir, 
    y_dir, 
    x2_dir
)

train_dataloader = Dataloader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Dataset for train images
valid_dataset = Dataset(
    ids_valid,
    x_dir, 
    y_dir, 
    x2_dir
)

valid_dataloader = Dataloader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="..\\logs\\fit\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
callbacks = [
    tensorboard_callback,
    tf.keras.callbacks.ModelCheckpoint('./model.h5', save_weights_only=True, save_best_only=True, mode='min'),
    tf.keras.callbacks.ReduceLROnPlateau(),
]

assert train_dataloader[0][0][0].shape == (BATCH_SIZE, 256, 256, 3)
assert train_dataloader[0][0][1].shape == (BATCH_SIZE, 256, 256, 3)
assert train_dataloader[0][1].shape == (BATCH_SIZE, 256, 256, 1)


model.fit_generator(
    generator=train_dataloader, 
    steps_per_epoch=len(train_dataloader), 
    epochs=EPOCHS, 
    callbacks=callbacks, 
    validation_data=valid_dataloader, 
    validation_steps=len(valid_dataloader),
)
model.save('./model/new_model.h5')
print('model saved!')





