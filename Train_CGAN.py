from __future__ import absolute_import, division, print_function, unicode_literals
import os

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Activation, BatchNormalization
from tensorflow.keras.layers import UpSampling2D, Input, Concatenate
from tensorflow.keras.models import Model
from models import Generator, create_discriminator
import numpy as np
import time
import datetime

import matplotlib.pyplot as plt

import sys
import csv

print(tf.config.list_physical_devices('GPU'))

# Input image = satellite, Real Image = DEM
BUFFER_SIZE = 400
BATCH_SIZE = 1
IMAGE_SIZE = 256
INPUT_CHANNELS = 3
OUTPUT_CHANNELS = 1
LAMBDA = 100
EPOCHS = 400


def resize(input_image, real_image, height, width):
    input_image = tf.image.resize(input_image, [height, width],
                                  method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    real_image = tf.image.resize(real_image, [height, width],
                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    return input_image, real_image


def random_crop(input_image, real_image):
    stacked_image = tf.stack([input_image, real_image], axis=0)
    cropped_image = tf.image.random_crop(
        stacked_image, size=[2, IMAGE_SIZE, IMAGE_SIZE, 3])

    return cropped_image[0], cropped_image[1]


# normalizing the images to [-1, 1]

def normalize(input_image, real_image):
    input_image = (input_image / 127.5) - 1
    real_image = (real_image / 127.5) - 1

    return input_image, real_image


@tf.function()
def random_jitter(input_image, real_image):
    # resizing to 286 x 286 x 3
    input_image, real_image = resize(input_image, real_image, 286, 286)

    # randomly cropping to 256 x 256 x 3
    input_image, real_image = random_crop(input_image, real_image)

    if tf.random.uniform(()) > 0.5:
        # random mirroring
        input_image = tf.image.flip_left_right(input_image)
        real_image = tf.image.flip_left_right(real_image)

    return input_image, real_image


def load_image_train(input_image, real_image):
    # real_image = tf.image.grayscale_to_rgb(real_image)
    # input_image, real_image = random_jitter(input_image, real_image)
    input_image, real_image = normalize(input_image, real_image)
    # real_image = tf.image.rgb_to_grayscale(real_image)
    return input_image, real_image


def load_image_test(input_image, real_image):
    # real_image = tf.image.grayscale_to_rgb(real_image)
    # input_image, real_image = random_jitter(input_image, real_image)
    input_image, real_image = normalize(input_image, real_image)
    # real_image = tf.image.rgb_to_grayscale(real_image)
    return input_image, real_image


# load and prepare training images
def load_real_samples(filename):
    # load compressed arrays
    data = np.load(filename)
    # unpack arrays
    X1, X2 = data['arr_0'], data['arr_1']
    return [X1, X2]


with tf.device('CPU'):

    # load image data
    dataset = load_real_samples('Dataset/Train.npz')
    print('Loaded', dataset[0].shape, dataset[1].shape)
    dataset[0] = tf.cast(dataset[0], tf.float32)
    dataset[1] = tf.cast(dataset[1], tf.float32)

    train_dataset = tf.data.Dataset.from_tensor_slices((dataset[0], dataset[1]))
    train_dataset = train_dataset.map(load_image_train)
    train_dataset = train_dataset.shuffle(BUFFER_SIZE)
    train_dataset = train_dataset.batch(1)

generator = Generator()
tf.keras.utils.plot_model(generator, to_file="generator.png", show_shapes=True, dpi=64)
discriminator = Discriminator()

loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)

    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

    total_disc_loss = real_loss + generated_loss

    return total_disc_loss


def generator_loss(disc_generated_output, gen_output, target):
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

    # mean absolute error
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

    total_gen_loss = gan_loss + (LAMBDA * l1_loss)

    return total_gen_loss, gan_loss, l1_loss


generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

checkpoint_dir = './training_checkpoints_CGAN'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)
manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)


def generate_images(model, test_input, tar):
    prediction = model(test_input, training=True)
    plt.figure(figsize=(15, 15))

    display_list = [test_input[0], tar[0], prediction[0]]
    title = ['Input Image', 'Ground Truth', 'Predicted Image']

    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.title(title[i])
        # getting the pixel values between [0, 1] to plot it.
        if (display_list[i].shape[2] == 1):
            display_list[i] = tf.image.grayscale_to_rgb(display_list[i])
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
    plt.show()


def save_images(model, test_input, tar, e, path):
    prediction = model(test_input, training=True)
    plt.figure(figsize=(15,15))

    display_list = [test_input[0], tar[0], prediction[0]]
    title = ['Input Image', 'Ground Truth', 'Predicted Image']

    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.title(title[i])
        # getting the pixel values between [0, 1] to plot it.
        if(display_list[i].shape[2] == 1):
            plt.imshow(display_list[i][:, :, 0]*0.5 + 0.5, cmap='gray')
        else:
            plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
    plt.savefig(path + "/step_"+ str(e) + '.png')
    plt.close()


def save_data(data_name, *data):
    with open(data_name, 'a') as fd:
        writer = csv.writer(fd)
        writer.writerow(list(data))

gen_loss = tf.keras.metrics.Mean('gen_loss', dtype=tf.float32)

log_dir = "logs/"

summary_writer = tf.summary.create_file_writer(
    log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

@tf.function
def train_step(input_image, target):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image, training=True)

        disc_real_output = discriminator([input_image, target], training=True)
        disc_generated_output = discriminator([input_image, gen_output], training=True)

        gen_loss_now, gan_loss, l1_loss = generator_loss(disc_generated_output, gen_output, target)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

    generator_gradients = gen_tape.gradient(gen_loss_now,
                                            generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss,
                                                 discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(generator_gradients,
                                            generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                                discriminator.trainable_variables))

    gen_loss(gen_loss_now)
    


# load image data
datasetT = load_real_samples('Dataset/Test.npz')

pretty = [5, 6, 11, 13]
pretty_dataset = ([datasetT[0][x] for x in pretty], [datasetT[1][x] for x in pretty])

print('Loaded', datasetT[0].shape, datasetT[1].shape)
datasetT[0] = tf.cast(datasetT[0], tf.float32)
datasetT[1] = tf.cast(datasetT[1], tf.float32)

test_dataset = tf.data.Dataset.from_tensor_slices((datasetT[0], datasetT[1]))
test_dataset = test_dataset.map(load_image_test)
test_dataset = test_dataset.batch(1)

def fit(train_ds, epochs):
    example_input, example_target = next(iter(test_dataset.take(1)))

    epoch_times = []
    for epoch in range(epochs):
        start = time.time()


        for input_image, target in train_ds:
            train_step(input_image, target)

        with summary_writer.as_default():
            tf.summary.scalar('loss', gen_loss.result(), step=epoch)

        epoch_times.append(time.time() - start)

        # Test on the same image so that the progress of the model can be
        # easily seen.
        save_images(generator, example_input, example_target, epoch, './results_CGAN')

        manager.save()
        print('Model saved')
        print('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                           time.time() - start))
        print('Estimated Time left for remaining {} epochs is {} sec = {} hours\n'.format(epochs - epoch - 1,
                                                                                          (epochs - epoch - 1) * sum(
                                                                                              epoch_times) / len(
                                                                                              epoch_times),
                                                                                          (epochs - epoch - 1) * sum(
                                                                                              epoch_times) / len(
                                                                                              epoch_times) / 3600))

checkpoint.restore(manager.latest_checkpoint)

fit(train_dataset, 400)
