import tensorflow as tf

from utils.helper import get_activation_layer


class DepthWiseSeparableConvolution(tf.keras.layers.Layer):
    def __init__(self, kernel_size, filters, name,
                 down_sample=False, use_batch_normalization=False,
                 use_dropout=False, activation=None):
        super(DepthWiseSeparableConvolution, self).__init__()
        assert(not (use_batch_normalization and use_dropout), "Cannot have both batch normalization and dropout")
        strides = (1, 1)
        if down_sample:
            strides = (2, 2)

        self.l1 = tf.keras.layers.DepthwiseConv2D(kernel_size=kernel_size, padding='same',
                                                  strides=strides, depthwise_initializer='he_normal',
                                                  depthwise_regularizer=tf.keras.regularizers.l2(l=0.01),
                                                  bias_regularizer=tf.keras.regularizers.l2(l=0.01),
                                                  name=name+"_l1")
        self.l2 = tf.keras.layers.Conv2D(filters=filters, kernel_size=1, kernel_initializer='he_normal',
                                         kernel_regularizer=tf.keras.regularizers.l2(l=0.01),
                                         bias_regularizer=tf.keras.regularizers.l2(l=0.01),
                                         name=name+"_l2")
        self.l3 = get_activation_layer(activation, name=name+"_l3")
        self.l4 = None
        if use_batch_normalization:
            self.l4 = tf.keras.layers.BatchNormalization(name=name+"_l4")
        elif use_dropout:
            self.l4 = tf.keras.layers.SpatialDropout2D(rate=0.1, name=name+"_l4")

    def set_trainable(self, value):
        self.l1.trainable = value
        self.l2.trainable = value
        if self.l3 is not None:
            self.l3.trainable = value
        if self.l4 is not None:
            self.l4.trainable = value

    def call(self, x):
        x = self.l1(x)
        x = self.l2(x)
        if self.l3:
            x = self.l3(x)
        if self.l4:
            x = self.l4(x)

        return x

class DownStairConvolutionBlock(tf.keras.layers.Layer):
    def __init__(self,  kernel_size, filters, downsize, name):
        super(DownStairConvolutionBlock, self).__init__(name=name)
        self.l1 = None
        if downsize:
            self.l1 = tf.keras.layers.MaxPooling2D(name=name+"_l1")
        self.l2 = DepthWiseSeparableConvolution(kernel_size, filters, use_batch_normalization=False,
                                                use_dropout=False, activation='relu', name=name+"_l2")
        self.l3 = DepthWiseSeparableConvolution(kernel_size, filters, use_batch_normalization=False,
                                                use_dropout=False, activation='relu', name=name+"_l3")
        self.l4 = DepthWiseSeparableConvolution(kernel_size, filters, use_batch_normalization=False,
                                                use_dropout=False, activation='relu', name=name + "_l4")
        self.l5 = DepthWiseSeparableConvolution(kernel_size, filters, use_batch_normalization=False,
                                                use_dropout=False, activation='relu', name=name + "_l5")

    def set_trainable(self, value):
        if self.l1:
            self.l1.trainable = value
        self.l2.trainable = value
        self.l3.trainable = value
        self.l4.trainable = value
        self.l5.trainable = value

    def call(self, x):
        if self.l1:
            x = self.l1(x)
        x = self.l2(x)
        x = x + self.l3(x)  # skip connection
        x = x + self.l4(x)  # skip connection
        x = x + self.l5(x)  # skip connection
        return x


class UpStairBlock(tf.keras.layers.Layer):
    def __init__(self, input_filters, kernel_size, filters, name):
        super(UpStairBlock, self).__init__(name=name)

        self.l1 = tf.keras.layers.UpSampling2D(name=name+"_l1")
        # self.l1 = tf.keras.layers.Conv2DTranspose(kernel_size=kernel_size, filters=input_filters,
        #                                           strides=2, padding='same', name=name+"_l1")
        self.l2 = tf.keras.layers.Concatenate(name=name+"_l2")
        self.l3 = DepthWiseSeparableConvolution(kernel_size, filters, use_batch_normalization=False,
                                                use_dropout=False, name=name+"_l3")
        self.l4 = DepthWiseSeparableConvolution(kernel_size, filters, use_batch_normalization=False,
                                                use_dropout=False, name=name+"_l4")

    def set_trainable(self, value):
        self.l1.trainable = value
        self.l2.trainable = value
        self.l3.trainable = value

    def call(self, x, d):
        x = self.l1(x)
        x = self.l2([d, x])
        x = self.l3(x)
        x = self.l4(x)
        return x


class HeatmapLayer(tf.keras.layers.Layer):
    def __init__(self, kernel_size, filters, name):
        super(HeatmapLayer, self).__init__(name=name)

        self.l1 = tf.keras.layers.DepthwiseConv2D(kernel_size=kernel_size, padding='same', name=name+"_l1",
                                                  depthwise_regularizer=tf.keras.regularizers.l2(l=0.01),
                                                  bias_regularizer=tf.keras.regularizers.l2(l=0.01)
                                                  )
        self.l2 = tf.keras.layers.Conv2D(filters=filters, kernel_size=1, name=name+"_l2",
                                         kernel_regularizer=tf.keras.regularizers.l2(l=0.01),
                                         bias_regularizer=tf.keras.regularizers.l2(l=0.01)
                                         )
        self.l3 = tf.keras.layers.Permute((3, 1, 2), name=name+"_l3")
        # self.l4 = tf.keras.layers.ReLU(max_value=1)
        self.l4 = tf.keras.layers.Activation(tf.keras.activations.sigmoid, name=name+"_l4")
        # self.l4 = get_activation_layer('relu', name=name+"_l4")

    def set_trainable(self, value):
        self.l1.trainable = value
        self.l2.trainable = value
        self.l3.trainable = value
        self.l4.trainable = value

    def call(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        return x


