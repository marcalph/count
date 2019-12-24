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
    # image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    return image, label

train = raw_train.map(format_example)
validation = raw_validation.map(format_example)
test = raw_test.map(format_example)
BATCH_SIZE = 1
SHUFFLE_BUFFER_SIZE = 1000
train_batches = train.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
validation_batches = validation.batch(BATCH_SIZE)
test_batches = test.batch(BATCH_SIZE)
for image_batch, label_batch in train_batches.take(1):
    pass

print(image_batch.shape)




IMG_SHAPE = (None, None, 3)
base_model = tf.keras.applications.VGG16(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')
namelist = [l.name for l in base_model.layers]
block1 = block1 = tf.keras.Sequential([l for l in base_model.layers[:namelist.index("block1_pool")+1]], name="convblock_1")
block2 = tf.keras.Sequential([l for l in base_model.layers if "block2" in l.name], name="convblock_2")
block3 = tf.keras.Sequential([l for l in base_model.layers if "block3"in l.name], name="convblock_3")
block4 = tf.keras.Sequential([l for l in base_model.layers if "block4"in l.name], name="convblock_4")
block5 = tf.keras.Sequential([l for l in base_model.layers if "block5" in l.name], name="convblock_5")
block2.build((None, None, None, 64))
block3.build((None, None, None, 128))
block4.build((None, None, None, 256))
block5.build((None, None, None, 512))

print(block1.summary())
print(block2.summary())
print(block3.summary())
print(block4.summary())
print(block5.summary())

feature_batch = block1(image_batch)
print(feature_batch.shape)
feature_batch = block2(feature_batch)
print(feature_batch.shape)

