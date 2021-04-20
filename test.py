import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from time import time
import cv2


from blaze_pose_model import BlazePoseModel
from config import IMAGE_SIZE


def load_image(image_path):
    img_raw = tf.io.read_file(image_path)
    img = tf.io.decode_image(img_raw)
    if img.shape[2] != 3:
        return None
    img = tf.image.convert_image_dtype(img, tf.float32)
    # img = tf.image.per_image_standardization(img)
    img = tf.image.resize_with_pad(img, IMAGE_SIZE, IMAGE_SIZE)
    img = tf.expand_dims(img, axis=0)
    return img


labels = ["nose", "left_eye", "right_eye", "left_ear", "right_ear",
          "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
          "left_wrist", "right_wrist", "left_hip", "right_hip",
          "left_knee", "right_knee", "left_ankle", "right_ankle"]


model = BlazePoseModel("HEATMAP")
# model_file_path = r'D:\Aerobi\Models and Data\PoseEstimation\blaze_pose\models\BlazePose_train_Heatmap_E-019_L-0.490.h5'
model_file_path = r'D:\Aerobi\Models and Data\PoseEstimation\blaze_pose\models\BlazePose_train_Heatmap_E-057_L-10.456.h5'
image_folder = r'D:\Aerobi\Models and Data\PoseEstimation\blaze_pose\test'
input_shape = (1, IMAGE_SIZE, IMAGE_SIZE, 3)
model.build(input_shape)
model.load_from(model_file_path)

image_list = os.listdir(image_folder)
# image_list = ['000000047684.jpg', '000000056733.jpg', '000000196742.jpg', '000000433122.jpg', '000000513936.jpg',
#               '000000567530.jpg', '000000394698.jpg']
for image_file_name in image_list:
    image_path = os.path.join(image_folder, image_file_name)
    img = load_image(image_path)
    if img is None:
        continue

    start_time = time()
    model.set_mode('INFERENCE')
    prediction = model(img, training=False)
    prediction = tf.transpose(prediction, perm=[0, 1, 3, 2])
    end_time = time()
    prediction = np.squeeze(prediction, axis=0)
    max = tf.math.reduce_max(prediction)
    min = tf.math.reduce_min(prediction)
    fps = 1 / (end_time - start_time)
    # prediction = prediction > 0.5
    print(f"{image_file_name} MIN = {min}, MAX = {max}, FPS = {fps:.2f}")

    fig, axes = plt.subplots(4, 5, figsize=(16, 16))  # the number of images in the grid is 5*5 (25)
    threshold = 0.5

    for i, ax in enumerate(axes.flat):
        if i == prediction.shape[0]:
            break
        show_img = np.zeros((64, 64))
        ax.imshow(prediction[i], cmap='gray',vmin=0, vmax=1)
        # ax.imshow(img[0])
        cx, cy = np.unravel_index(prediction[i].argmax(), prediction[i].shape)
        if prediction[i, cx, cy] > threshold:
            patches = [Circle((cy, cx), radius=1, color='red')]
            for p in patches:
                ax.add_patch(p)
        ax.set_title(labels[i])
    ax.imshow(img[0])
    plt.subplots_adjust(left=0, bottom=0, right=1, top=0.97, wspace=0, hspace=0.25)
    plt.show()

    pass