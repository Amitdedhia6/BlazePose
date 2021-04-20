import os
import tensorflow as tf

from config import NUM_KEYPOINTS
from utils.layers import DepthWiseSeparableConvolution, DownStairConvolutionBlock, HeatmapLayer,  UpStairBlock


class BlazePoseModel(tf.keras.Model):
    def __init__(self, model_type):
        super(BlazePoseModel, self).__init__()
        assert(model_type == "HEATMAP" or model_type == "REGRESSION")
        self.model_type = model_type

        self.downstair_1 = DownStairConvolutionBlock(kernel_size=3, filters=3, downsize=False, name='d1')
        self.downstair_2 = DownStairConvolutionBlock(kernel_size=3, filters=16, downsize=True, name='d2')
        self.downstair_3 = DownStairConvolutionBlock(kernel_size=3, filters=32, downsize=True, name='d3')
        self.downstair_4 = DownStairConvolutionBlock(kernel_size=3, filters=64, downsize=True, name='d4')
        self.downstair_5 = DownStairConvolutionBlock(kernel_size=3, filters=128, downsize=True, name='d5')
        self.downstair_6 = DownStairConvolutionBlock(kernel_size=3, filters=192, downsize=True, name='d6')
        # self.downstair_7 = DownStairConvolutionBlock(kernel_size=3, filters=32, downsize=False, name='d7')

        self.upstair_1 = UpStairBlock(input_filters=192, kernel_size=3, filters=128, name='u1')
        self.upstair_2 = UpStairBlock(input_filters=128, kernel_size=3, filters=64, name='u2')
        self.upstair_3 = UpStairBlock(input_filters=64, kernel_size=3, filters=32, name='u3')

        self.heatmap_layer = HeatmapLayer(kernel_size=3, filters=NUM_KEYPOINTS, name='heatmap')

        # self.regression_1 = DepthWiseSeparableConvolution(kernel_size=3, filters=32,
        #                                                   activation='relu', down_sample=False)
        # self.regression_2 = DepthWiseSeparableConvolution(kernel_size=3, filters=64,
        #                                                   activation='relu', down_sample=True)
        # self.regression_3 = DepthWiseSeparableConvolution(kernel_size=3, filters=128,
        #                                                   activation='relu', down_sample=True)
        # self.regression_4 = DepthWiseSeparableConvolution(kernel_size=3, filters=192,
        #                                                   activation='relu', down_sample=True)
        # self.regression_5 = DepthWiseSeparableConvolution(kernel_size=3, filters=192,
        #                                                   activation='relu', down_sample=True)
        # self.regression_6 = DepthWiseSeparableConvolution(kernel_size=3, filters=192,
        #                                                   activation='relu', down_sample=True)
        # self.regression_final = tf.keras.models.Sequential([
        #     tf.keras.layers.Conv2D(filters=3 * NUM_KEYPOINTS, kernel_size=2, activation=None),
        #     tf.keras.layers.Reshape((3 * NUM_KEYPOINTS, ))
        # ])

    def set_mode(self, mode):
        assert mode == "TRAINING" or mode == 'INFERENCE'
        train_mode = mode == 'TRAINING'
        self.downstair_1.set_trainable(train_mode)
        self.downstair_2.set_trainable(train_mode)
        self.downstair_3.set_trainable(train_mode)
        self.downstair_4.set_trainable(train_mode)
        self.downstair_5.set_trainable(train_mode)
        self.downstair_6.set_trainable(train_mode)
        # self.downstair_7.set_trainable(train_mode)
        self.upstair_1.set_trainable(train_mode)
        self.upstair_2.set_trainable(train_mode)
        self.upstair_3.set_trainable(train_mode)
        self.heatmap_layer.set_trainable(train_mode)

    def call(self, x):
        # x = [b, 256, 256, 3] where b = batch size, 256 refers to IMAGE_SIZE in config.py
        d1 = self.downstair_1(x)
        d2 = self.downstair_2(d1)
        d3 = self.downstair_3(d2)
        d4 = self.downstair_4(d3)
        d5 = self.downstair_5(d4)
        d6 = self.downstair_6(d5)
        # d7 = self.downstair_7(d6)

        u1 = self.upstair_1(d6, d5)
        u2 = self.upstair_2(u1, d4)
        u3 = self.upstair_3(u2, d3)

        if self.model_type == "HEATMAP":
            heatmap = self.heatmap_layer(u3)

        # r1 = self.regression_1(u3 + d2)
        # r2 = self.regression_2(r1) + d3
        # r3 = self.regression_3(r2) + d4
        # r4 = self.regression_4(r3) + d5
        # r5 = self.regression_5(r4)
        # r6 = self.regression_6(r5)
        # r_final = self.regression_final(r6)

        if self.model_type == "HEATMAP":
            return heatmap
        else:
            return None

    def save_to(self, path):
        assert(os.path.exists(os.path.dirname(path)))
        self.save_weights(path)

    def load_from(self, path):
        assert(os.path.exists(os.path.dirname(path)))
        self.load_weights(path, by_name=True)
