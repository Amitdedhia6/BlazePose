import tensorflow as tf
from train.blaze_pose_trainer import BlazePoseTrainer


gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)
else:
    print("No GPUs found. Training would be very slow.")

print()
trainer = BlazePoseTrainer()
trainer.train()

print()
print("Training completed")

