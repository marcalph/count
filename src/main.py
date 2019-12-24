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
        paddings = tf.constant([[pad_x // 2, pad_x // 2], [pad_y // 2, pad_y // 2]])
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
    """SDCnet with vgg16 encoding base and bilinear upsampling of images
    """
    def __init__(self, num_class, div_times=2):
        super(SDCnet, self).__init__()
        self.num_class = num_class
        self.div_times = div_times
        encoder = VGG16(weights="imagenet", include_top=False, trainable=False)
        # conv features
        self.conv1 = tf.keras.Sequential([l for l in encoder.layers[:encoder.layers.index("block1_pool") + 1]], name="conv1")
        self.conv2 = tf.keras.Sequential([l for l in encoder.layers if "block2" in l.name], name="conv2")
        self.conv3 = tf.keras.Sequential([l for l in encoder.layers if "block3" in l.name], name="conv3")
        self.conv4 = tf.keras.Sequential([l for l in encoder.layers if "block4" in l.name], name="conv4")
        self.conv5 = tf.keras.Sequential([l for l in encoder.layers if "block5" in l.name], name="conv5")
        del encoder
        self.conv2.build((None, None, None, 64))
        self.conv3.build((None, None, None, 128))
        self.conv4.build((None, None, None, 256))
        self.conv5.build((None, None, None, 512))
        # interval clf
        self.interval_clf = tf.keras.Sequential([tf.keras.layers.AvgPool2D((2, 2), stride=2),
                                                 tf.keras.layers.ReLU(),
                                                 tf.keras.layers.Conv2D(512, (1, 1), activation=tf.keras.activations.relu),
                                                 tf.keras.layers.Conv2D(num_class, (1, 1))])
        # division decider
        self.div_decider = tf.keras.Sequential([tf.keras.layers.AvgPool2D((2, 2), stride=2),
                                                tf.keras.layers.ReLU(),
                                                tf.keras.layers.Conv2D(512, (1, 1), activation=tf.keras.activations.relu),
                                                tf.keras.layers.Conv2D(1, (1, 1), activation=tf.keras.activations.sigmoid)])
        self.up45 = Upsample_WO_Transposition(256, 512)
        self.up34 = Upsample_WO_Transposition(512, 512)

    def call(self, input_tensor):
        """forward pass
        """
        feature_map = dict()
        x = self.conv1(input_tensor)
        x = self.conv2(x)
        x = self.conv3(x)
        conv3_features = x if self.div_times > 1 else []
        x = self.conv4(x)
        conv4_features = x if self.div_times > 0 else []
        x = self.conv5(x)
        conv5_features = x if self.div_times > 0 else []
        x = self.interval_clf(x)
        feature_map = {'conv3': conv3_features, 'conv4': conv4_features, 'conv5': conv5_features, 'cls0': x}
        return feature_map

    def upsample(self, feature_map):
        """upsampling (w/o transposed conv)
        params
        ------
        feature_map is a dict
        """
        division_res = dict()
        division_res['cls0'] = feature_map['cls0']
        if self.div_times > 0:
            # div45: Upsample and get weight
            new_conv4 = self.up45(feature_map['conv5'], feature_map['conv4'])
            new_conv4_w = self.div_decider(new_conv4)
            new_conv4_w = tf.sigmoid(new_conv4_w)
            new_conv4_reg = self.interval_clf(new_conv4)
            del feature_map['conv5'], feature_map['conv4']
            division_res['cls1'] = new_conv4_reg
            division_res['w1'] = 1-new_conv4_w

        if self.div_times > 1:
            # div34: upsample and get weight
            new_conv3 = self.up34(new_conv4, feature_map['conv3'])
            new_conv3_w = self.lw_fc(new_conv3)
            # new_conv3_w = F.sigmoid(new_conv3_w)
            new_conv3_w = tf.sigmoid(new_conv3_w)
            new_conv3_reg = self.fc(new_conv3)
            del feature_map['conv3'], new_conv3, new_conv4
            division_res['cls2'] = new_conv3_reg
            division_res['w2'] = 1-new_conv3_w

        feature_map['cls0'] = []
        del feature_map
        return division_res

    def parse_merge(self, division_res):
        """compute count
        """
        pass