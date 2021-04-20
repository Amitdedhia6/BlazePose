import math
import numpy as np
import os
from pycocotools.coco import COCO
import tensorflow as tf
import random

from config import DATA_DIR, BATCH_SIZE, IMAGE_SIZE, NUM_KEYPOINTS, HEATMAP_SIZE, GAUSSIAN_SIGMA
from utils.helper import gaussian
from utils.sliced_ordered_dict import SlicableOrderedDict


class BlazePoseDataset(tf.keras.utils.Sequence):
    def __init__(self, dataset_type, epoch_end_cb=None):
        assert(dataset_type == "HEATMAP" or dataset_type == "REGRESSION" or
               dataset_type == "INFERENCE" or dataset_type == "INFERENCE-HEATMAP")
        self.type = dataset_type    # train, test
        self.annotation_filepath = os.path.join(DATA_DIR, 'annotations', 'person_keypoints.json')
        self.dir = ''
        self.annotation_ids = []
        self.allow_shuffle = True
        self.generate_heatmap_data = False
        self.epoch_end_cb = epoch_end_cb

        if self.type == 'HEATMAP':
            self.dir = os.path.join(DATA_DIR, "train")
            self.generate_heatmap_data = True
        elif self.type == 'REGRESSION':
            self.dir = os.path.join(DATA_DIR, "train")
            self.generate_heatmap_data = False
        elif self.type == 'INFERENCE':
            self.dir = os.path.join(DATA_DIR, "test")
            self.generate_heatmap_data = False
        elif self.type == 'INFERENCE-HEATMAP':
            self.dir = os.path.join(DATA_DIR, "test")
            self.generate_heatmap_data = True

        file_list = os.listdir(self.dir)

        # Following lines are used for testing purpose. Should always be commented.
        # file_list_len = 64
        # if self.type == 'INFERENCE-HEATMAP' or self.type == 'INFERENCE':
        #     file_list_len = file_list_len // 10
        # if len(file_list) > file_list_len:
        #     file_list = file_list[0:file_list_len]
        # self.allow_shuffle = False

        filenames_dict = {file_list[i]: 1 for i in range(len(file_list))}
        self.annotation_id_image_data_map = SlicableOrderedDict()
        self.annotation_data_map = {}

        coco = COCO(self.annotation_filepath)  # load annotations
        image_ids = list(coco.imgs.keys())
        for _, image_id in enumerate(image_ids):
            image_meta_data = coco.imgs[image_id]
            image_file_name = image_meta_data['file_name']
            if image_file_name in filenames_dict:
                annotation_ids = coco.getAnnIds(imgIds=image_id)
                annotations = coco.loadAnns(annotation_ids)
                for ann in annotations:
                    if ann['num_keypoints'] <= 0:
                        continue
                    if ann['id'] in self.annotation_data_map:
                        continue
                    if ann['iscrowd']:
                        continue
                    if ann['category_id'] != 1:
                        continue

                    self.annotation_id_image_data_map[ann['id']] = image_meta_data
                    self.annotation_data_map[ann['id']] = (ann['keypoints'], ann['area'])
            else:
                continue

        self.annotation_ids = list(self.annotation_id_image_data_map.keys())

    def __len__(self):
        return math.ceil(len(self.annotation_id_image_data_map) / BATCH_SIZE)

    def __getitem__(self, idx):
        start_index = idx * BATCH_SIZE
        end_index = (idx + 1) * BATCH_SIZE
        if end_index > len(self.annotation_id_image_data_map):
            end_index = len(self.annotation_id_image_data_map)

        batch_size = end_index - start_index

        x = np.empty(shape=[batch_size, IMAGE_SIZE, IMAGE_SIZE, 3], dtype=np.float32)
        if self.generate_heatmap_data:
            y = np.zeros(shape=[batch_size, NUM_KEYPOINTS, HEATMAP_SIZE, HEATMAP_SIZE], dtype=np.float32)
            meta = None
        else:
            # Below: X,Y position coordinates (0-255) and visibility
            y = np.empty(shape=[batch_size, NUM_KEYPOINTS * 3], dtype=np.float32)
            meta = np.empty(shape=[batch_size, 1])  # segmented area of the person

        ix = 0
        for item in self.annotation_ids[start_index:end_index]:
            annotation_id = item

            image_meta_data = self.annotation_id_image_data_map[annotation_id]
            image_file_name = image_meta_data['file_name']
            w = image_meta_data['width']
            h = image_meta_data['height']
            image_path = os.path.join(self.dir, image_file_name)
            img_raw = tf.io.read_file(image_path)
            img = tf.io.decode_image(img_raw)
            if img.shape[2] == 1:
                img = tf.image.grayscale_to_rgb(img)
            img = tf.image.convert_image_dtype(img, tf.float32)
            # img = tf.image.per_image_standardization(img)
            img = tf.image.resize_with_pad(img, IMAGE_SIZE, IMAGE_SIZE)
            x[ix] = img

            key_points = np.array(self.annotation_data_map[annotation_id][0], dtype=float)
            key_points = np.reshape(key_points, [NUM_KEYPOINTS, 3])
            # The key points are (x,Y, visibility). X refers to columns and Y refers to rows.
            # The image data is (Rows, columns). Hence we need to swap X and Y
            key_points[:, [0, 1]] = key_points[:, [1, 0]]

            # Adjust the original values of key-points, width and height because the image was padded to make it square
            rows_padded = 0
            cols_padded = 0
            if w > h:
                rows_padded = w - h
            elif h > w:
                cols_padded = h - w

            key_points[:, 0] += rows_padded / 2 * (key_points[0:, 2] > 0)
            key_points[:, 1] += cols_padded / 2 * (key_points[0:, 2] > 0)
            w += cols_padded
            h += rows_padded

            area = self.annotation_data_map[annotation_id][1]

            if self.generate_heatmap_data:
                key_points[:, 0] = HEATMAP_SIZE * key_points[:, 0] / w
                key_points[:, 1] = HEATMAP_SIZE * key_points[:, 1] / h

                for j in range(NUM_KEYPOINTS):
                    heatmap_image = y[ix, j]
                    key_point_item = key_points[j]
                    if key_point_item[2] == 2:
                        heatmap_image = gaussian(heatmap_image, (key_point_item[0], key_point_item[1]),
                                                 GAUSSIAN_SIGMA)
                        y[ix, j] = heatmap_image
                    else:
                        continue
            else:
                key_points[:, 0] = IMAGE_SIZE * key_points[:, 0] / w
                key_points[:, 1] = IMAGE_SIZE * key_points[:, 1] / h
                key_points[:, 2] = key_points[:, 2] > 0

                y[ix] = np.reshape(key_points, NUM_KEYPOINTS * 3)
                meta[ix] = area * IMAGE_SIZE * IMAGE_SIZE / (w * h)
            ix += 1

        y = tf.convert_to_tensor(y)
        if meta is not None:
            meta = tf.convert_to_tensor(meta)

        # if self.generate_heatmap_data:
        #     # channel last
        #     y = tf.transpose(y, perm=[0, 2, 3, 1])

        return x, y, meta

    def shuffle(self):
        if self.allow_shuffle:
            random.shuffle(self.annotation_ids)
