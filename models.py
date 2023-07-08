import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Activation, BatchNormalization
from tensorflow.keras.layers import UpSampling2D, Input, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV3Small, EfficientNetB0
import matplotlib.pyplot as plt
import numpy as np
INPUT_SHAPE = 3
OUTPUT_SHAPE = 1

def unet_decoder_block(input, filters, skip, stage):
    x = UpSampling2D((2, 2))(input)

    if skip is not None:
        x = Concatenate()([x, skip])

    x = Conv2D(filters, kernel_size=(3, 3), padding='same', strides=1, use_bias=False,)(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)

    x = Conv2D(filters, kernel_size=(3, 3), padding='same', strides=1, use_bias=False)(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)

    return x

def create_generator_small():
    inputs = Input(shape=(256, 256, 3), name="input_image")
    encoder = MobileNetV3Small(input_tensor=inputs,weights='imagenet', include_top=False)
    
    skip1 = encoder.get_layer(index=7).output
    skip2 = encoder.get_layer(index=21).output
    skip3 = encoder.get_layer(index=39).output
    skip4 = encoder.get_layer(index=153).output

    x = unet_decoder_block(encoder.output, 256, skip4, 0)
    x = unet_decoder_block(x, 128, skip3, 1)
    x = unet_decoder_block(x, 64, skip2, 2)
    x = unet_decoder_block(x, 32, skip1, 3)
    x = unet_decoder_block(x, 16, None, 4)

    x = Conv2D(1, kernel_size=(3, 3), padding='same', strides=1, kernel_initializer="he_normal")(x)

    outputs = Activation("tanh")(x)

    model = Model(inputs=inputs, outputs=outputs)

    return model

def create_generator_efficient():
    inputs = Input(shape=(256,256,3), name="input_image")
    encoder = EfficientNetB0(input_tensor=inputs,weights='imagenet',include_top=False)

    skip1 = encoder.get_layer("block1a_se_excite").output
    skip2 = encoder.get_layer("block2b_add").output
    skip3 = encoder.get_layer("block3b_add").output
    skip4 = encoder.get_layer("block5c_add").output

    x = unet_decoder_block(encoder.output, 256, skip4, 0)
    x = unet_decoder_block(x, 128, skip3, 1)
    x = unet_decoder_block(x, 64, skip2, 2)
    x = unet_decoder_block(x, 32, skip1, 3)
    x = unet_decoder_block(x, 16, None, 4)

    x = Conv2D(1, kernel_size=(3, 3), padding='same', strides=1, kernel_initializer="he_normal")(x)

    outputs = Activation("tanh")(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model



def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                                kernel_initializer=initializer, use_bias=False))

    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())

    result.add(tf.keras.layers.LeakyReLU())

    return result

def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                        padding='same',
                                        kernel_initializer=initializer,
                                        use_bias=False))

    result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())

    return result


def Generator(input_channels=INPUT_SHAPE, output_channels=OUTPUT_SHAPE):
    down_stack = [
        downsample(64, 4, apply_batchnorm=False), # (bs, 128, 128, 64)
        downsample(128, 4), # (bs, 64, 64, 128)
        downsample(256, 4), # (bs, 32, 32, 256)
        downsample(512, 4), # (bs, 16, 16, 512)
        downsample(512, 4), # (bs, 8, 8, 512)
        downsample(512, 4), # (bs, 4, 4, 512)
        downsample(512, 4), # (bs, 2, 2, 512)
        downsample(512, 4), # (bs, 1, 1, 512)
    ]

    up_stack = [
        upsample(512, 4, apply_dropout=True), # (bs, 2, 2, 1024)
        upsample(512, 4, apply_dropout=True), # (bs, 4, 4, 1024)
        upsample(512, 4, apply_dropout=True), # (bs, 8, 8, 1024)
        upsample(512, 4), # (bs, 16, 16, 1024)
        upsample(256, 4), # (bs, 32, 32, 512)
        upsample(128, 4), # (bs, 64, 64, 256)
        upsample(64, 4), # (bs, 128, 128, 128)
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(output_channels, 4,
                                         strides=2,
                                         padding='same',
                                         kernel_initializer=initializer,
                                         activation='tanh') # (bs, 256, 256, 3)

    concat = tf.keras.layers.Concatenate()

    inputs = tf.keras.layers.Input(shape=[None,None,input_channels])
    x = inputs

    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = concat([x, skip])

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


def Discriminator(input_channels=INPUT_SHAPE, output_channels=OUTPUT_SHAPE):
    initializer = tf.random_normal_initializer(0., 0.02)

    inp = tf.keras.layers.Input(shape=[256, 256, input_channels], name='input_image')
    tar = tf.keras.layers.Input(shape=[256, 256, output_channels], name='target_image')

    x = tf.keras.layers.concatenate([inp, tar])  # (bs, 256, 256, channels*2)

    down1 = downsample(64, 4, False)(x)  # (bs, 128, 128, 64)
    down2 = downsample(128, 4)(down1)  # (bs, 64, 64, 128)
    down3 = downsample(256, 4)(down2)  # (bs, 32, 32, 256)

    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (bs, 34, 34, 256)
    conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                  kernel_initializer=initializer,
                                  use_bias=False)(zero_pad1)  # (bs, 31, 31, 512)

    batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

    leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (bs, 33, 33, 512)

    last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                  kernel_initializer=initializer)(zero_pad2)  # (bs, 30, 30, 1)

    return tf.keras.Model(inputs=[inp, tar], outputs=last)

def create_discriminator(input_channels=INPUT_SHAPE, output_channels=OUTPUT_SHAPE):
    initializer = tf.random_normal_initializer(0., 0.02)

    inp = tf.keras.layers.Input(shape=[256, 256, input_channels], name='input_image')
    tar = tf.keras.layers.Input(shape=[256, 256, output_channels], name='target_image')

    x = tf.keras.layers.concatenate([inp, tar]) # (bs, 256, 256, channels*2)

    down1 = downsample(64, 4, False)(x) # (bs, 128, 128, 64)
    down2 = downsample(128, 4)(down1) # (bs, 64, 64, 128)
    down3 = downsample(256, 4)(down2) # (bs, 32, 32, 256)

    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3) # (bs, 34, 34, 256)
    conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                kernel_initializer=initializer,
                                use_bias=False)(zero_pad1) # (bs, 31, 31, 512)

    batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

    leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu) # (bs, 33, 33, 512)

    last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                kernel_initializer=initializer)(zero_pad2) # (bs, 30, 30, 1)

    return tf.keras.Model(inputs=[inp, tar], outputs=last)


def main():
    gen_small = create_generator_small()
    gen_eff = create_generator_efficient()
    gen_pix = Generator()

    print(gen_small.summary())

if __name__ == "__main__":
    main()