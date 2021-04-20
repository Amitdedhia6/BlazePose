from tensorflow import keras
import tensorflow as tf
import tensorflow_addons as tfa

from config import NUM_KEYPOINTS


def __heatmap_mae_loss(y_true, y_pred):
    loss = (y_pred - y_true)
    loss = tf.reduce_mean(loss, axis=(2, 3))
    return loss


def __heatmap_custom_mae_loss(y_true, y_pred):
    keypoint_present = tf.cast(tf.reduce_sum(y_true, axis=(2, 3)) > 0, tf.float32)
    # num_keypoint = tf.reduce_sum(keypoint_present, axis=1)

    y_true_clipped = y_true[keypoint_present > 0]
    y_pred_clipped = y_pred[keypoint_present > 0]

    d = tf.math.abs(y_true_clipped - y_pred_clipped)
    loss = 10 * y_true_clipped * d + (1 - y_true_clipped) * d
    loss = tf.reduce_sum(loss, axis=(1, 2))
    loss = tf.reduce_sum(loss, axis=0)

    return loss, y_true_clipped.shape[0]


def __heatmap_custom_log_loss(y_true, y_pred):
    keypoint_present = tf.cast(tf.reduce_sum(y_true, axis=(2, 3)) > 0, tf.float32)
    # num_keypoint = tf.reduce_sum(keypoint_present, axis=1)

    y_true_clipped = y_true[keypoint_present > 0]
    y_pred_clipped = y_pred[keypoint_present > 0]
    alpha = 0.25

    loss = -1 * (10 * y_true_clipped * (1 - y_pred_clipped) * tf.math.log(y_pred_clipped + tf.keras.backend.epsilon()) +
                 (1 - y_true_clipped) * y_pred_clipped * tf.math.log(1 - y_pred_clipped + tf.keras.backend.epsilon()))
    return loss, y_true_clipped.shape[0]


def __heatmap_mse_loss(y_true, y_pred):
    loss = ((y_pred - y_true) ** 2)
    loss = tf.reduce_mean(loss, axis=(2, 3))
    return loss


def __heatmap_custom_mse_loss(y_true, y_pred):
    keypoint_present = tf.cast(tf.reduce_sum(y_true, axis=(2, 3)) > 0, tf.float32)
    # num_keypoint = tf.reduce_sum(keypoint_present, axis=1)

    y_true_clipped = y_true[keypoint_present > 0]
    y_pred_clipped = y_pred[keypoint_present > 0]

    loss = (y_true_clipped - y_pred_clipped) ** 2
    loss = tf.reduce_mean(loss, axis=(1, 2))
    return loss, y_true_clipped.shape[0]


def __heatmap_custom_dice_loss(y_true, y_pred):
    keypoint_present = tf.cast(tf.reduce_sum(y_true, axis=(2, 3)) > 0, tf.float32)
    y_true_clipped = y_true[keypoint_present > 0]
    y_pred_clipped = y_pred[keypoint_present > 0]

    intersection = tf.reduce_sum(y_true_clipped * y_pred_clipped, axis=(1, 2)) + \
          tf.keras.backend.epsilon()
    sum = tf.reduce_sum(y_true_clipped + y_pred_clipped, axis=(1, 2)) + \
          tf.keras.backend.epsilon()
    dice_coeff = 2 * intersection / sum
    loss = 1 - dice_coeff
    return loss, y_true_clipped.shape[0]


def __heatmap_custom_dice_log_loss(y_true, y_pred):
    keypoint_present = tf.cast(tf.reduce_sum(y_true, axis=(2, 3)) > 0, tf.float32)
    y_true_keypoint_present = y_true[keypoint_present > 0]
    y_pred_keypoint_present = y_pred[keypoint_present > 0]
    y_true_keypoint_absent = y_true[keypoint_present == 0]
    y_pred_keypoint_absent = y_pred[keypoint_present == 0]

    intersection = tf.reduce_sum(y_true_keypoint_present * y_pred_keypoint_present, axis=(1, 2)) + \
                   tf.keras.backend.epsilon()
    sum = tf.reduce_sum(y_true_keypoint_present + y_pred_keypoint_present, axis=(1, 2)) + \
          tf.keras.backend.epsilon()
    dice_coeff = 2 * intersection / sum
    dice_loss = 1 - dice_coeff

    log_loss_keypoint_present = -1 * (10 * y_true_keypoint_present * (1 - y_pred_keypoint_present) *
                                      tf.math.log(y_pred_keypoint_present + tf.keras.backend.epsilon()) +
                                      (1 - y_true_keypoint_present) * y_pred_keypoint_present *
                                      tf.math.log(1 - y_pred_keypoint_present + tf.keras.backend.epsilon()))

    log_loss_keypoint_present = tf.reduce_mean(log_loss_keypoint_present, axis=(1, 2))
    log_loss_keypoint_absent = -1 * (10 * (1 - y_true_keypoint_absent) * y_pred_keypoint_absent *
                                     tf.math.log(1 - y_pred_keypoint_absent + tf.keras.backend.epsilon()))
    loss_keypoint_absent = tf.reduce_mean(log_loss_keypoint_absent, axis=(1, 2))

    loss_keypoint_present = dice_loss + log_loss_keypoint_present
    loss = tf.concat([loss_keypoint_present, loss_keypoint_absent],axis=0)
    return loss


def __heatmap_focal_loss(y_true, y_pred):
    y_true_reshaped = tf.reshape(y_true, [tf.shape(y_true)[0], tf.shape(y_true)[1], -1])
    y_pred_reshaped = tf.reshape(y_pred, [tf.shape(y_pred)[0], tf.shape(y_pred)[1], -1])
    loss_obj = tfa.losses.SigmoidFocalCrossEntropy(alpha=0.25, gamma=4)
    loss = loss_obj(y_true=y_true_reshaped, y_pred=y_pred_reshaped)
    return loss


def heatmap_pipeline_loss(y_true, y_pred):
    # loss, num_keypoints = __heatmap_custom_dice_loss(y_true, y_pred)
    loss = __heatmap_custom_dice_log_loss(y_true, y_pred)
    return loss


def regression_pipeline_loss(y_true, y_pred):
    g = tf.reshape(y_true, [-1, NUM_KEYPOINTS, 3])
    d = tf.reshape(y_pred, [-1, NUM_KEYPOINTS, 3])

    loss_1 = keras.losses.mean_squared_error(g[:, :, 0:2], d[:, :, 0:2])
    loss_2 = keras.losses.binary_crossentropy(tf.expand_dims(g[:, :, 2], -1),
                                              tf.keras.activations.sigmoid(tf.expand_dims(d[:, :, 2], -1)))
    return tf.math.add(loss_1, loss_2)
