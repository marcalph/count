# sdc-net
# spatial divide & conquer to count object from closed set ie [0, x] instead of [0; +inf[
# upsampling, decoding and dividing are done at the feature map level to avoid redundant computations
# architecture
# vgg16 encoder                 >> use conv block to generate feature maps
# unet decoder                  >> upsample and divide feature map
# count interval classifier   |
# division decider            | >> generate counts and division masks on divided feature maps



import tensorflow as tf
from tensorflow.keras.applications import VGG16
# get convolutionnal parts of vgg16 and use it as an encoder
print(tf.config.experimental.list_physical_devices("GPU"))


# upsampling block
class Upsample_WO_Transposition(tf.keras.Model):
    def __init__(self, up_out_num_filters, cat_out_num_filters):
        super(Upsample_WO_Transposition, self).__init__
        self.up_out_num_filters = up_out_num_filters
        self.cat_out_num_filters = cat_out_num_filters

    def call(self, feature_map_0, feature_map_1):
        feature_map_0 = tf.keras.UpSampling2D(size=2, interpolation="bilinear")(feature_map_0)
        feature_map_0 = tf.keras.layers.Conv2D(self.up_out_num_filters)(feature_map_0)
        # padding
        pad_x = feature_map_0.shape[1] - feature_map_1.shape[1]
        pad_y = feature_map_0.shape[2] - feature_map_1.shape[2]
        paddings = tf.constant([[pad_x//2, pad_x//2], [pad_y//2, pad_y//2]])
        feature_map_0 = tf.pad(feature_map_0, paddings)
        # concat then conv
        feat = tf.concat([feature_map_0, feature_map_1], axis=0)
        del feature_map_0, feature_map_1
        feat = tf.keras.layers.Conv2D(self.cat_out_num_filters)(feat)
        feat = tf.keras.layers.ReLU()(feat)
        feat = tf.keras.layers.Conv2D(self.cat_out_num_filters)(feat)
        feat = tf.keras.layers.ReLU()(feat)
        return feat



class SDCnet(tf.keras.Model):
    def __init__(self, num_class):
        super(SDCnet, self).__init__()
        self.num_class = num_class
        self.encoder = VGG16(weights="imagenet", include_top=False, trainable=False)

        self.interval_clf = tf.keras.Sequential([tf.keras.layers.AvgPool2D((2,2), stride=2),
                                                 tf.keras.layers.ReLU(),
                                                 tf.keras.layers.Conv2D(512, (1,1)), activation=tf/keras.activations.relu),
                                                 tf.keras.layers.Conv2D(num_class, (1,1))])

        self.div_decider = tf.keras.Sequential([tf.keras.layers.AvgPool2D((2,2), stride=2),
                                                tf.keras.layers.ReLU(),
                                                tf.keras.layers.Conv2D(512, (1,1), activation=tf.keras.activations.relu),
                                                tf.keras.layers.Conv2D(1, (1,1), activation=tf.keras.activations.sigmoid])

        self.up45 = Upsample_WO_Transposition(256, 512)
        self.up34 = Upsample_WO_Transposition(512, 512)

    def call(self, input_tensor):
        self.encoder(input_tensor)
        pass

    def upsample(self):
        pass
        # upsampling unet way






