from tensorflow.keras.layers import *
from tensorflow.keras.models import *
import tensorflow as tf


def shallow_siamese(x):
    """The network for full image. The size of input should be 256x256x3.
    """
    # Block 1
    x = Conv2D(2, (3, 3), padding='same', name='s_block1_conv1', activation=tf.nn.leaky_relu)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D()(x)

    # Block 2
    x = Conv2D(2, (3, 3), padding='same', name='s_block2_conv1', activation=tf.nn.leaky_relu)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D()(x)

    # Block 3
    x = Conv2D(2, (3, 3), padding='same', name='s_block3_conv1', activation=tf.nn.leaky_relu)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D()(x)

    # Block 4
    x = Conv2D(2, (3, 3), padding='same', name='s_block4_conv1', activation=tf.nn.leaky_relu)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D()(x)

    # Block 5
    x = Conv2D(2, (3, 3), padding='same', name='s_block5_conv1', activation=tf.nn.leaky_relu)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D()(x)
    
    # connect 
    x = Conv2D(1, (3, 3), padding='same', name='s_connect_conv1', activation=tf.nn.leaky_relu)(x)
    x = Conv2D(1, (3, 3), padding='same', name='s_connect_conv2', activation=tf.nn.leaky_relu)(x)
    x = Activation('sigmoid')(x)

    return x

def siamese_deep_unet(image_size):
    """The main model of SiamUnet.
    """

    inputs_1 = Input((image_size, image_size, 3))
    inputs_2 = Input((image_size, image_size, 3))
    # Block 1
    x = Conv2D(32, (3, 3), padding='same', name='block1_conv1', activation=tf.nn.leaky_relu)(inputs_1)
    block_1_out = Conv2D(32, (3, 3), padding='same', name='block1_conv2', activation=tf.nn.leaky_relu)(x)
    x = MaxPooling2D(padding='same', name=f'downsample_1')(block_1_out)
    
    # Block 2
    x = Conv2D(32, (3, 3), padding='same', name='block2_conv1', activation=tf.nn.leaky_relu)(x)
    block_2_out = Conv2D(32, (3, 3), padding='same', name='block2_conv2', activation=tf.nn.leaky_relu)(x)
    x = MaxPooling2D(padding='same', name=f'downsample_2')(block_2_out)
    
    # Block 3
    x = Conv2D(32, (3, 3), padding='same', name='block3_conv1', activation=tf.nn.leaky_relu)(x)
    block_3_out = Conv2D(32, (3, 3), padding='same', name='block3_conv2', activation=tf.nn.leaky_relu)(x)
    x = MaxPooling2D(padding='same', name=f'downsample_3')(block_3_out)
    
    # Block 4
    x = Conv2D(32, (3, 3), padding='same', name='block4_conv1', activation=tf.nn.leaky_relu)(x)
    block_4_out = Conv2D(32, (3, 3), padding='same', name='block4_conv2', activation=tf.nn.leaky_relu)(x)
    x = MaxPooling2D(padding='same', name=f'downsample_4')(block_4_out)
    
    # Block 5
    x = Conv2D(32, (3, 3), padding='same', name='block5_conv1', activation=tf.nn.leaky_relu)(x)
    block_5_out = Conv2D(32, (3, 3), padding='same', name='block5_conv2', activation=tf.nn.leaky_relu)(x)
    x = MaxPooling2D(padding='same', name=f'downsample_5')(block_5_out)

    # center block
    x = Conv2D(32, (3,3), padding='same', name='center_block_conv1', activation=tf.nn.leaky_relu)(x)
    x2 = shallow_siamese(inputs_2)
    x = concatenate([x, x2])
    x = Conv2D(32, (3,3), padding='same', name='center_block_conv2', activation=tf.nn.leaky_relu)(x)
    x = UpSampling2D(name=f'upsample_cemter_block')(x)

    # Block UP 1
    x = concatenate([x, block_5_out])
    x = Conv2D(32, (3, 3), padding='same', activation=tf.nn.leaky_relu)(x)
    x = Conv2D(32, (3, 3), padding='same', activation=tf.nn.leaky_relu)(x)
    x = UpSampling2D(name=f'upsample_1')(x)

    # Block UP 2
    x = concatenate([x, block_4_out])
    x = Conv2D(32, (3, 3), padding='same', activation=tf.nn.leaky_relu)(x)
    x = Conv2D(32, (3, 3), padding='same', activation=tf.nn.leaky_relu)(x)
    x = UpSampling2D(name=f'upsample_2')(x)
    
    # Block UP 3
    x = concatenate([x, block_3_out])
    x = Conv2D(32, (3, 3), padding='same', activation=tf.nn.leaky_relu)(x)
    x = Conv2D(32, (3, 3), padding='same', activation=tf.nn.leaky_relu)(x)
    x = UpSampling2D(name=f'upsample_3')(x)
    
    # Block UP 4
    x = concatenate([x, block_2_out])
    x = Conv2D(32, (3, 3), padding='same', activation=tf.nn.leaky_relu)(x)
    x = Conv2D(32, (3, 3), padding='same', activation=tf.nn.leaky_relu)(x)
    x = UpSampling2D(name=f'upsample_4')(x)
    
    # Block UP 5
    x = concatenate([x, block_1_out])
    x = Conv2D(32, (3, 3), padding='same', activation=tf.nn.leaky_relu)(x)
    x = Conv2D(32, (3, 3), padding='same', activation=tf.nn.leaky_relu)(x)
    
    # tail
    x = Conv2D(32, kernel_size=3, padding='same', activation=tf.nn.leaky_relu)(x)
    x = Conv2D(32, kernel_size=3, padding='same', activation=tf.nn.leaky_relu)(x)
    x = Conv2D(1, kernel_size=3, padding='same', activation=tf.nn.leaky_relu)(x)
    outputs = Activation('sigmoid')(x)
    
    model = Model(inputs=[inputs_1, inputs_2], outputs=outputs)

    return model