# Root Directory where all images and annotation files are present.
# The folder should have these sub-folders: annotations, test and train
# 'annotations' file should contain `person_keypoints.json` (COCO format)
# test and train should contain corresponding images

# Data
DATA_DIR = r'D:\Aerobi\Models and Data\PoseEstimation\blaze_pose'
NUM_KEYPOINTS = 17          # number of key points
IMAGE_SIZE = 256            # Size of image input to the model. All images will be resized to this size.
HEATMAP_SIZE = 64           # Size of output heatmap
BATCH_SIZE = 32

# Training
NUM_EPOCHS = 25
START_EPOCH = 1
INITIAL_LEARNING_RATE = 0.001
GAUSSIAN_SIGMA = 2          # Sigma value to be passed while generating gaussian heatmap
TRAIN_MODE = 'HEATMAP'      # 'HEATMAP' or 'REGRESSION'
MODEL_SAVE_DIR = r'D:\Aerobi\Models and Data\PoseEstimation\blaze_pose/models'
USE_EXISTING_MODEL = False  # True if we want to use weights from existing saved model
# Below - Path of the saved model (to load weights from), used if USE_EXISTING_MODEL is True
EXISTING_MODEL_PATH = r'D:\Aerobi\Models and Data\PoseEstimation\blaze_pose/models/BlazePose_train_Heatmap_E-056_L-12.619.h5'

