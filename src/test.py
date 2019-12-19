import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.applications import VGG16
import matplotlib.pyplot as plt




SPLIT_WEIGHTS = (8, 1, 1)
splits = tfds.Split.TRAIN.subsplit(weighted=SPLIT_WEIGHTS)
(raw_train, raw_validation, raw_test), metadata = tfds.load('cats_vs_dogs', split=list(splits), with_info=True, as_supervised=True)

print(raw_train)
print(raw_validation)
print(raw_test)

get_label_name = metadata.features['label'].int2str

IMG_SIZE = 160 # All images will be resized to 160x160

def format_example(image, label):
    image = tf.cast(image, tf.float32)
    image = (image/127.5) - 1
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    return image, label

train = raw_train.map(format_example)
validation = raw_validation.map(format_example)
test = raw_test.map(format_example)
BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 1000
train_batches = train.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
validation_batches = validation.batch(BATCH_SIZE)
test_batches = test.batch(BATCH_SIZE)
for image_batch, label_batch in train_batches.take(1):
    pass

print(image_batch.shape)




IMG_SHAPE = (None, None, 3)
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')
print(base_model.summary())
namelist = [l.name for l in base_model.layers]
print(namelist.index("block_1_project_BN"))

block1 = tf.keras.Sequential([l for l in base_model.layers[:4]])
block2 = tf.keras.Sequential([l for l in base_model.layers if "block2" in l.name])
