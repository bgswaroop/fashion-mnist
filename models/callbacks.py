import os
import tensorflow as tf
from utils.session_setup import SessionSetup
import logging

logger = logging.getLogger(__name__)


class LogLossOnBatchEnd(tf.keras.callbacks.Callback):
    def __init__(self, num_batches=25):
        self.results_path = str(SessionSetup().get_session_folder_path())
        self.num_batches = num_batches
        super().__init__()

    def on_batch_end(self, batch, logs=None):
        if batch != 0 and batch % self.num_batches == 0:
            if not os.path.exists(self.results_path):
                os.mkdir(self.results_path)
            logger.info("batch {} ".format(batch) + str(logs))


class SaveModelOnEpochEnd(tf.keras.callbacks.Callback):

    def __init__(self, num_epochs=1, file_prefix=None):
        super().__init__()
        self.results_path = str(SessionSetup().get_session_folder_path())
        self.num_epochs = num_epochs

        if type(file_prefix) is str:
            self.file_prefix = file_prefix + "_"
        else:
            self.file_prefix = ""

    def on_epoch_end(self, epoch, logs=None):

        if epoch % self.num_epochs == 0:
            if not os.path.exists(self.results_path):
                os.mkdir(self.results_path)

            self.model.save(
                self.results_path + os.path.sep + "{}trained_model_epoch{}_acc{}_loss{}_valAcc{}_valLoss{}.h5".format(
                    self.file_prefix,
                    epoch,
                    str(round(logs['acc'], 2)),
                    str(round(logs['loss'], 2)),
                    str(round(logs['val_acc'], 2)),
                    str(round(logs['val_loss'], 2)))
            )

            logger.info("Saved the intermediate model for epoch number : {}".format(epoch))
        logger.info("Learning rate: {}".format(float(tf.keras.backend.get_value(self.model.optimizer.lr))))