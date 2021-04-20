import math
import statistics
import tensorflow as tf

from blaze_pose_dataset import BlazePoseDataset
from blaze_pose_model import BlazePoseModel
from config import EXISTING_MODEL_PATH, IMAGE_SIZE, INITIAL_LEARNING_RATE
from config import NUM_EPOCHS, START_EPOCH, TRAIN_MODE, USE_EXISTING_MODEL
from train.callback import BlazePoseTrainingCallback
from train.custom_loss import heatmap_pipeline_loss, regression_pipeline_loss
from train.oks import ObjectKeyPointSimilarity
from utils.helper import get_iou_gt_pred


class BlazePoseTrainer:
    def __init__(self):
        super(BlazePoseTrainer, self).__init__()
        assert (TRAIN_MODE == "HEATMAP" or TRAIN_MODE == "REGRESSION")

        self.training_mode = TRAIN_MODE
        self.model = BlazePoseModel(self.training_mode)
        self.train_dataset = BlazePoseDataset(self.training_mode)
        if TRAIN_MODE == "HEATMAP":
            self.test_dataset = BlazePoseDataset('INFERENCE-HEATMAP')
        else:
            self.test_dataset = BlazePoseDataset('INFERENCE')
        self.training_callback = BlazePoseTrainingCallback(self.model)
        self.val_loss_queue = []
        pass

    def _add_to_val_loss_queue(self, val):
        self.val_loss_queue.append(val)
        if len(self.val_loss_queue) > 6:
            self.val_loss_queue.pop(0)

    def _train_heatmap(self):
        loss = heatmap_pipeline_loss

        def get_lr(curr_lr):
            if len(self.val_loss_queue) < 6:
                return curr_lr
            else:
                count = 0
                for i in range(5):
                    if self.val_loss_queue[i] < self.val_loss_queue[i+1]:
                        count += 1

                if count == 5:
                    return curr_lr * 0.9
                else:
                    return curr_lr

        learning_rate = INITIAL_LEARNING_RATE
        for epoch in range(START_EPOCH - 1, START_EPOCH - 1 + NUM_EPOCHS):
            learning_rate = get_lr(learning_rate)

            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

            total_val_loss = 0
            total_train_loss = 0
            num_train_images = 0
            num_val_images = 0

            iou_list = []
            total_no_keypoints_list = []
            total_no_keypoints_error_list = []

            total_train_keypoints = 0
            total_test_keypoints = 0

            self.train_dataset.shuffle()
            self.model.set_mode("TRAINING")
            for x, y, meta in self.train_dataset:
                num_train_images += x.shape[0]
                with tf.GradientTape() as tape:
                    prediction = self.model(x, training=True)
                    # loss_tensor, num_keypoints = loss(y, prediction)
                    loss_tensor = loss(y, prediction)
                    total_train_loss += tf.math.reduce_sum(loss_tensor)
                    # total_train_keypoints += num_keypoints

                    # The crucial step - train the model.
                    grads = tape.gradient(loss_tensor, self.model.trainable_variables)
                    optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

            total_train_loss = total_train_loss / num_train_images
            # total_train_loss = total_train_loss / total_train_keypoints
            self.training_callback.on_epoch_end_heatmap_training(epoch, total_train_loss)

            # now check how is the performance on test images
            self.model.set_mode("INFERENCE")
            for x, y, meta in self.test_dataset:
                num_val_images += x.shape[0]
                prediction = self.model(x, training=False)
                # loss_tensor, num_keypoints = loss(y, prediction)
                loss_tensor = loss(y, prediction)
                total_val_loss += tf.math.reduce_sum(loss_tensor)
                # total_test_keypoints += num_keypoints
                iou, total_absent, total_error = get_iou_gt_pred(y, prediction)
                iou_list.append(iou)
                total_no_keypoints_list.append(total_absent)
                total_no_keypoints_error_list.append(total_error)

            total_val_loss = total_val_loss / num_val_images
            # total_val_loss = total_val_loss / total_test_keypoints
            self._add_to_val_loss_queue(total_val_loss)

            print(f"Epoch {epoch + 1}, lr:{learning_rate:.5f}, train_loss: {total_train_loss:.4f}, "
                  f"val_loss: {total_val_loss:.4f}, iou_metric: [{statistics.mean(iou_list):.3f}, "
                  f"{sum(total_no_keypoints_list)}, {sum(total_no_keypoints_error_list):.0f}]")

    def _train_regression(self):
        loss = regression_pipeline_loss
        train_metric = ObjectKeyPointSimilarity()
        test_metric = ObjectKeyPointSimilarity()
        optimizer = tf.keras.optimizers.Adam()

        for epoch in range(NUM_EPOCHS):
            # Start a new count for a training loop.
            train_metric.reset_states()
            test_metric.reset_states()

            total_train_loss = 0
            total_val_loss = 0
            num_train_images = 0
            num_val_images = 0

            # run the loop on train images
            for x, y, meta in self.train_dataset:
                # Calculate prediction and loss in such way it can be use later to calculate gradient.
                num_train_images += x.shape[0]
                with tf.GradientTape() as tape:
                    prediction = self.model(x, training=True)
                    loss_tensor = loss(y, prediction)
                    total_train_loss += tf.math.reduce_sum(loss_tensor)

                    # The crucial step - train the model.
                    grads = tape.gradient(loss_tensor, self.model.trainable_variables)
                    optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

                    # Accumulate the loss and metrics.
                    train_metric.update_state(y, prediction, meta)
            total_train_loss = tf.math.round(total_train_loss / num_train_images)
            train_metric_value = train_metric.result()

            # now check how is the performance on test images
            for x, y, meta in self.test_dataset:
                num_val_images += x.shape[0]
                prediction = self.model(x, training=False)
                loss_tensor = loss(y, prediction)
                total_val_loss += tf.math.reduce_sum(loss_tensor)
                test_metric.update_state(y, prediction, meta)
            total_val_loss = tf.math.round(total_val_loss / num_val_images)

            test_metric_value = test_metric.result()
            self.training_callback.on_epoch_end_regression_training(epoch, total_val_loss, test_metric_value)

            print(f"Epoch {epoch + 1}, Train_loss: {total_train_loss:.0f}, oks-train: {train_metric_value:.6f}, "
                  f"Test_loss: {total_val_loss:.0f}, oks-test: {test_metric_value:.6f}")

    def train(self):
        input_shape = (1, IMAGE_SIZE, IMAGE_SIZE, 3)
        # self.model.build(input_shape)

        self.model(tf.random.uniform(shape=input_shape))
        self.model.summary()
        if USE_EXISTING_MODEL:
            self.model.load_from(EXISTING_MODEL_PATH)

        if self.training_mode == 'HEATMAP':
            self._train_heatmap()
        elif self.training_mode == 'REGRESSION':
            self._train_regression()