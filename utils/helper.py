import numpy as np
import tensorflow as tf
import keras.backend as K


# credit: https://github.com/princeton-vl/pose-hg-train/blob/master/src/pypose/draw.py
# Copyright (c) 2016, University of Michigan
def gaussian(img, pt, sigma):
    # Draw a 2D gaussian

    # Check that any part of the gaussian is in-bounds
    ul = [int(pt[0] - 3 * sigma), int(pt[1] - 3 * sigma)]
    br = [int(pt[0] + 3 * sigma + 1), int(pt[1] + 3 * sigma + 1)]
    if (ul[0] > img.shape[1] or ul[1] >= img.shape[0] or
            br[0] < 0 or br[1] < 0):
        # If not, just return the image as is
        return img

    # Generate gaussian
    size = 6 * sigma + 1
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2

    # The gaussian is not normalized, we want the center value to equal 1
    g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

    # Usable gaussian range
    g_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]

    # Image range
    img_x = max(0, ul[0]), min(br[0], img.shape[1])
    img_y = max(0, ul[1]), min(br[1], img.shape[0])

    # img[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    img[img_y[0]:img_y[1], img_x[0]:img_x[1]] = 1
    return img


def get_activation_layer(activation, name):
    if activation == 'relu':
        return tf.keras.layers.ReLU(name=name+'_relu', activity_regularizer=tf.keras.regularizers.l2(l=0.01))
    elif activation == 'softmax':
        return tf.keras.layers.Softmax(name=name+'_softmax')
    elif activation == 'sigmoid':
        return tf.keras.layers.Activation(tf.keras.activations.sigmoid, name=name+'_sigmoid')
    elif activation == 'tanh':
        return tf.keras.layers.Activation(tf.keras.activations.tanh, name=name+'_tanh')
    else:
        return None


def get_iou_gt_pred(y_true, y_pred):
    threshold = 0.5
    y_pred_binary = tf.cast(y_pred > threshold, tf.float32)

    keypoint_present = tf.cast(tf.reduce_sum(y_true, axis=(2, 3)) > 0, tf.float32)
    y_true_keypoint_present = y_true[keypoint_present > 0]
    y_pred_keypoint_present = y_pred_binary[keypoint_present > 0]
    y_true_keypoint_absent = y_true[keypoint_present == 0]
    y_pred_keypoint_absent = y_pred_binary[keypoint_present == 0]
    y_pred_keypoint_absent_sum = tf.reduce_sum(tf.cast(tf.reduce_sum(y_pred_keypoint_absent, axis=(1, 2)) > 0,
                                                       tf.float32))
    y_pred_keypoint_absent_sum = K.eval(y_pred_keypoint_absent_sum)

    intersection = tf.reduce_sum(y_true_keypoint_present * y_pred_keypoint_present, axis=(1, 2))
    binary_sum = tf.cast((y_true_keypoint_present + y_pred_keypoint_present) > 0, tf.float32)
    union = tf.reduce_sum(binary_sum, axis=(1, 2))
    iou = intersection / union
    iou = tf.reduce_mean(iou)
    iou = K.eval(iou)


    return iou, y_true_keypoint_absent.shape[0], y_pred_keypoint_absent_sum