from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import LearningRateScheduler

class myCallBack:
    def custom_callbacks(self):
        def scheduler(epoch,lr):
            if epoch > 10:
                return 1e-4
            else :
                return lr

        my_callbacks = [ModelCheckpoint('model.h5',monitor='val_accuracy',save_best_only=True),
                        TensorBoard(log_dir='logs',write_graph=True,write_images=True,histogram_freq=1,embeddings_freq=1),
                        LearningRateScheduler(scheduler)]
        return my_callbacks