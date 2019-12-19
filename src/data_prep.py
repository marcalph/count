import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pathlib



images = pathlib.Path("./ShanghaiTech")/"part_A/test_data/images/*.jpg"
target_dens = pathlib.Path("./ShanghaiTech/part_A/test_data/densities/*.npy")


list_ds = tf.data.Dataset.list_files(str(images))

def decode_img(img):
    img = tf.io.read_file(img)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img

AUTOTUNE = tf.data.experimental.AUTOTUNE
ds = list_ds.map(decode_img, num_parallel_calls=AUTOTUNE)

for image in ds.take(10):
    print(image.numpy().shape)
