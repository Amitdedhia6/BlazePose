import os

from config import MODEL_SAVE_DIR


class BlazePoseTrainingCallback:
    def __init__(self, model):
        self.max_num_saved_models = 5
        self.model = model
        self.losses = {}            # dictionary of {epoch: loss}
        self.oks_values = {}     # dictionary of {epoch: metric value}
        self.saved_files = {}       # dictionary of {epoch: filepath}

    def on_epoch_end_heatmap_training(self, epoch, loss_value):
        # min_loss = 100000000.0
        # curr_loss = loss_value
        # max_loss = -1
        # epoch_with_max_loss = -1
        # if len(self.losses) > 0:
        #     for epoch_no in self.losses:
        #         if self.losses[epoch_no] < min_loss:
        #             min_loss = self.losses[epoch_no]
        #         if self.losses[epoch_no] > max_loss:
        #             max_loss = self.losses[epoch_no]
        #             epoch_with_max_loss = epoch_no
        #
        # if curr_loss < min_loss:
        #     model_file_name = f'BlazePose_train_Heatmap_E-{epoch:0>2d}_L-{curr_loss:.4f}.h5'
        #     model_save_path = os.path.join(MODEL_SAVE_DIR, model_file_name)
        #     if len(self.losses) >= self.max_num_saved_models:
        #         os.remove(self.saved_files[epoch_with_max_loss])
        #         del self.losses[epoch_with_max_loss]
        #         del self.saved_files[epoch_with_max_loss]
        #
        #     self.model.save_to(model_save_path)
        #     self.losses[epoch] = curr_loss
        #     self.saved_files[epoch] = model_save_path

        curr_loss = loss_value
        model_file_name = f'BlazePose_train_Heatmap_E-{epoch+1:0>3d}_L-{curr_loss:.3f}.h5'
        model_save_path = os.path.join(MODEL_SAVE_DIR, model_file_name)
        self.model.save_to(model_save_path)
        self.losses[epoch] = curr_loss
        self.saved_files[epoch] = model_save_path

    def on_epoch_end_regression_training(self, epoch, loss_value, oks_value):
        # min_oks = 100000000.0
        # curr_oks = oks_value
        # max_oks = -1
        # epoch_with_min_oks = -1
        # if curr_oks < 0.1:
        #     return
        #
        # if len(self.oks_values) > 0:
        #     for epoch_no in self.oks_values:
        #         if self.oks_values[epoch_no] < min_oks:
        #             min_oks = self.oks_values[epoch_no]
        #             epoch_with_min_oks = epoch_no
        #         if self.oks_values[epoch_no] > max_oks:
        #             max_oks = self.oks_values[epoch_no]
        #
        # if curr_oks > max_oks:
        #     model_file_name = f'BlazePose_train_Regression_E-{epoch:0>2d}_OKS-{curr_oks:.4f}.h5'
        #     model_save_path = os.path.join(MODEL_SAVE_DIR, model_file_name)
        #     if len(self.oks_values) >= self.max_num_saved_models:
        #         os.remove(self.saved_files[epoch_with_min_oks])
        #         del self.oks_values[epoch_with_min_oks]
        #         del self.saved_files[epoch_with_min_oks]
        #
        #     self.model.save_to(model_save_path)
        #     self.oks_values[epoch] = oks_value
        #     self.saved_files[epoch] = model_save_path

        curr_oks = oks_value
        curr_loss = loss_value
        model_file_name = f'BlazePose_train_Regression_E-{epoch+1:0>3d}_L-{curr_loss:.0f}_OKS-{curr_oks:.4f}.h5'
        model_save_path = os.path.join(MODEL_SAVE_DIR, model_file_name)
        self.model.save_to(model_save_path)