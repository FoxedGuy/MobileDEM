import tensorflow as tf
import matplotlib.pyplot as plt
import os
import numpy as np
from models import create_generator_efficient, create_discriminator


def normalize(input_image, real_image):
    input_image = (input_image / 127.5) - 1
    real_image = (real_image / 127.5) - 1

    return input_image, real_image


def load_image_test(input_image, real_image):
    # real_image = tf.image.grayscale_to_rgb(real_image)
    input_image, real_image = normalize(input_image, real_image)
    return input_image, real_image


def load_real_samples(filename):
    data = np.load(filename)
    X1, X2 = data['arr_0'], data['arr_1']
    return [X1, X2]


datasetT = load_real_samples('Dataset/Test.npz')
print('Loaded', datasetT[0].shape, datasetT[1].shape)
datasetT[0] = tf.cast(datasetT[0],tf.float32)
datasetT[1] = tf.cast(datasetT[1],tf.float32)

test_dataset = tf.data.Dataset.from_tensor_slices((datasetT[0], datasetT[1]))
test_dataset = test_dataset.map(load_image_test)
test_dataset = test_dataset.shuffle(3000)
test_dataset = test_dataset.batch(1)


def generate_images(model, test_input, total, tar=tf.zeros([0,0])):
    path = 'test_out_Efficient'
    prediction = model(test_input, training=True)
    pre = ['input', 'target', 'prediction']
    dirs = ['/inputs/', '/targets/', '/predictions/']
    plt.figure(figsize=(15,15))
    if(tf.size(tar) > 0):
        display_list = [test_input[0], tar[0], prediction[0]]
        title = ['Input Image', 'Ground Truth', 'Predicted Image']
    else:
        display_list = [test_input[0], prediction[0]]
        title = ['Input Image', 'Predicted Image']
    for i in range(len(title)):
        plt.subplot(1, 3, i+1)
        plt.title(title[i])
        # getting the pixel values between [0, 1] to plot it.
        if(display_list[i].shape[2] == 1):
            plt.imshow(display_list[i][:, :, 0]*0.5 + 0.5, cmap='gray')
            plt.imsave(path + dirs[i] + pre[i] + str(total) + ".png", display_list[i][:, :, 0]*0.5 + 0.5, cmap='gray')
        else:
            dl = display_list[i]*0.5 + 0.5
            dl = np.array(dl)#, dtype=np.uint8)
            dl = np.array(255*dl, dtype=np.uint8)
            plt.imshow(dl)
            plt.imsave(path + dirs[i] + pre[i] + str(total) +".png", dl)
        plt.axis('off')
    plt.show()
    if(not tf.size(tar) > 0):
        return prediction


gen = create_generator_efficient()
disc = create_discriminator()


generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

checkpoint_dir = './training_checkpoints_efficient'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=gen,
                                 discriminator=disc)
manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)

checkpoint.restore(manager.latest_checkpoint)

tot = 0
for inp, tar in test_dataset:
    generate_images(gen, inp, tot, tar)
    tot += 1
