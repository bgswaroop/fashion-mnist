import tensorflow as tf
from models.callbacks import SaveModelOnEpochEnd
from utils.session_setup import SessionSetup
import logging
import pickle
import numpy as np

logger = logging.getLogger(__name__)


class ConvolutionalNeuralNet1:
    def __init__(self):
        self.model = None

    def build_model(self):
        input_layer = tf.keras.Input(shape=(28, 28, 1), dtype=float, name="Input")
        x = tf.keras.layers.Conv2D(filters=20, kernel_size=(3, 3), activation=tf.keras.activations.relu,
                                   kernel_regularizer=tf.keras.regularizers.l2(0.0001))(input_layer)
        x = tf.keras.layers.Conv2D(filters=20, kernel_size=(3, 3), activation=tf.keras.activations.relu,
                                   kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)
        x = tf.keras.layers.Flatten()(x)

        x = tf.keras.layers.Dense(units=128, activation=tf.keras.activations.relu,
                                  kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
        output_layer = tf.keras.layers.Dense(units=10, activation=tf.keras.activations.softmax)(x)

        self.model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
        self.model.summary()

    def compile_model(self):
        if self.model is None:
            self.build_model()

        self.model.compile(optimizer=tf.keras.optimizers.Adam(),
                           loss=tf.keras.losses.CategoricalCrossentropy(),
                           metrics=['acc'])

    def train_model(self, train_data, val_data, file_prefix=None):
        if self.model is None:
            self.build_model()
            self.compile_model()

        history = self.model.fit(x=train_data.batch(128), verbose=2,
                                 callbacks=[SaveModelOnEpochEnd(file_prefix=file_prefix)],
                                 epochs=150, validation_data=val_data.batch(128), initial_epoch=1,
                                 use_multiprocessing=True)

        # save the training history
        filename = "{}_train_history.pkl".format(file_prefix)
        full_file_path = str(SessionSetup().get_session_folder_path().joinpath(filename))
        with open(full_file_path, 'wb') as f:
            pickle.dump(history.history, f)

        return history

    def predict_model(self, test_data):
        predictions = self.model.predict(x=test_data.batch(tf.data.experimental.cardinality(test_data))).argmax(axis=1)
        ground_truth = np.stack([x for _, x in test_data]).argmax(axis=1)

        from sklearn.metrics import precision_recall_fscore_support as score
        precision, recall, f1_score, _ = score(ground_truth, predictions)

        results = {"confusion_matrix": tf.math.confusion_matrix(ground_truth, predictions),
                   "accuracy             ": np.sum(predictions == ground_truth) / len(predictions),
                   "class-wise precision ": precision,
                   "class-wise recall    ": recall,
                   "class-wise f1_score  ": f1_score}

        for metric, value in results.items():
            if metric == "confusion_matrix":
                logger.info("{}:\n{}".format(metric, str(np.round(value, decimals=2))))
            else:
                logger.info("{}:{}".format(metric, str(np.round(value, decimals=2))))

        return results

    def select_best_model(self, file_prefix=None):

        if type(file_prefix) is not str:
            file_prefix = ""

        # # ask the user to pick the desired model
        import tkinter as tk
        from tkinter.filedialog import askopenfilename

        root = tk.Tk()
        root.withdraw()
        # root.mainloop()
        # root.update()
        session_dir = SessionSetup().get_session_folder_path()
        selected_best_model = askopenfilename(
            initialdir=str(session_dir),
            title="Select the best {} model".format(file_prefix),
            filetypes=[("HDF5 Files", "*.h5")]
        )
        logger.info("Selected best model: {}\n".format(selected_best_model))

        self.update_weights(selected_best_model)

        # mark loaded layers as not trainable
        for layer in self.model.layers:
            layer.trainable = False

        full_filename = session_dir.joinpath("{}_best_model.h5".format(file_prefix))
        self.model.save(full_filename)

    def train_flow(self, train_data, val_data, file_prefix=None):
        self.build_model()
        self.compile_model()
        history = self.train_model(train_data, val_data, file_prefix)
        # self.select_best_model(file_prefix)
        return history

    def update_weights(self, filename):
        if self.model is None:
            self.build_model()
        self.model.load_weights(filename)


if __name__ == "__main__":
    net = ConvolutionalNeuralNet1()
    net.select_best_model("Test")
