import os
from pathlib import Path
import pandas as pd
import numpy as np
import scipy.io
from tqdm import tqdm
import tensorflow as tf
import pickle
import logging

logger = logging.getLogger(__name__)


class FashionMnistUtils:
    def __init__(self):
        self.pwd = os.path.dirname(os.path.realpath(__file__))
        self.test_csv = Path(self.pwd).joinpath("fashion-mnist_test.csv")
        self.train_csv = Path(self.pwd).joinpath("fashion-mnist_train.csv")

    def __load_data(self, partition="train"):
        if partition is "train":
            data_frame = pd.read_csv(self.train_csv, index_col='label', dtype=np.float, memory_map=True)
        elif partition is "test":
            data_frame = pd.read_csv(self.test_csv, index_col='label', dtype=np.float, memory_map=True)
        else:
            raise ValueError("Invalid partition: expected 'train' or 'test'")

        # reshape the 784 row pixels into 28*28 images
        num_samples = len(data_frame)
        data = [None] * num_samples
        labels = np.empty(num_samples)
        for idx in np.arange(num_samples):
            data[idx] = np.array(data_frame.iloc[[idx]]).reshape((28, 28))
            labels[idx] = int(data_frame.index[idx])

        return data, labels

    def __extract_corf_contours(self, batch_data):
        # Python to MATLAB connectivity
        # takes about 12 seconds
        import matlab.engine
        eng = matlab.engine.start_matlab()
        eng.cd(str(Path(self.pwd).parent.joinpath("CORFPushPull")), nargout=0)

        batch_binary_map = np.empty_like(batch_data).astype(int)
        batch_corf_response = np.empty_like(batch_data).astype(float)
        mat_filename = str(Path(self.pwd).joinpath("cache/input_image.mat"))
        for idx, img in tqdm(enumerate(batch_data)):
            scipy.io.savemat(mat_filename, {"img": img})

            bm, cr = eng.contour_detection_from_python(1.0, 4.0, 1.8, 0.007, nargout=2)

            batch_binary_map[idx] = np.array(bm._data).reshape((28, 28), order='F')  # MATLAB stores in col major order
            batch_corf_response[idx] = np.array(cr._data).reshape((28, 28), order='F')

        return batch_binary_map, batch_corf_response

    def load_train_val_data(self, val_split=0.2):
        """
        Load the train + validation tensorflow data
        :param val_split: float value must be in the range 0-1
        :return: (train_grayscale, val_grayscale), (train_corf, val_corf)
        """

        # Load the data
        train_data_cache_file = Path(self.pwd).joinpath("cache/fashion_mnist_train_contours.pkl")
        if train_data_cache_file.exists():
            with open(str(train_data_cache_file), "rb") as f:
                grayscale_images, labels, _, corf_images = pickle.load(f)
        else:
            grayscale_images, labels = self.__load_data(partition="train")
            binary_maps, corf_images = self.__extract_corf_contours(grayscale_images)
            with open(str(train_data_cache_file), 'wb') as f:
                pickle.dump([grayscale_images, labels, binary_maps, corf_images], f)

        # Reshaping the arrays to match the tensorflow inputs of 4D arrays
        labels = labels.astype(int)
        grayscale_images = np.expand_dims(np.stack(grayscale_images), axis=-1)
        corf_images = np.expand_dims(corf_images, axis=-1)

        # Split the train data into train + validation
        num_images = len(grayscale_images)
        indices = np.arange(num_images)
        np.random.seed(seed=123)
        np.random.shuffle(indices)

        val_indices = indices[:round(num_images * val_split)]
        train_indices = indices[round(num_images * val_split):]

        # Prepare the tensorflow data sets
        train_grayscale = tf.data.Dataset.from_tensor_slices(
            (tf.convert_to_tensor(grayscale_images[train_indices]), tf.one_hot(labels[train_indices], depth=10))
        )
        train_corf = tf.data.Dataset.from_tensor_slices(
            (tf.convert_to_tensor(corf_images[train_indices]), tf.one_hot(labels[train_indices], depth=10))
        )
        val_grayscale = tf.data.Dataset.from_tensor_slices(
            (tf.convert_to_tensor(grayscale_images[val_indices]), tf.one_hot(labels[val_indices], depth=10))
        )
        val_corf = tf.data.Dataset.from_tensor_slices(
            (tf.convert_to_tensor(corf_images[val_indices]), tf.one_hot(labels[val_indices], depth=10))
        )

        return (train_grayscale, val_grayscale), (train_corf, val_corf)

    def load_test_data(self):
        """
        Load the tensorflow data set for evaluation
        :return: test_grayscale, test_corf
        """

        # Load the data
        test_data_cache_file = Path(self.pwd).joinpath("cache/fashion_mnist_test_contours.pkl")
        if test_data_cache_file.exists():
            with open(str(test_data_cache_file), "rb") as f:
                grayscale_images, labels, _, corf_images = pickle.load(f)
        else:
            grayscale_images, labels = self.__load_data(partition="test")
            binary_maps, corf_images = self.__extract_corf_contours(grayscale_images)
            with open(str(test_data_cache_file), 'wb') as f:
                pickle.dump([grayscale_images, labels, binary_maps, corf_images], f)

        # Reshaping the arrays to match the tensorflow inputs of 4D arrays
        labels = labels.astype(int)
        grayscale_images = np.expand_dims(np.stack(grayscale_images), axis=-1)
        corf_images = np.expand_dims(corf_images, axis=-1)

        # Prepare the tensorflow data sets
        test_grayscale = tf.data.Dataset.from_tensor_slices(
            (tf.convert_to_tensor(grayscale_images), tf.one_hot(labels, depth=10))
        )
        test_corf = tf.data.Dataset.from_tensor_slices(
            (tf.convert_to_tensor(corf_images), tf.one_hot(labels, depth=10))
        )

        return test_grayscale, test_corf


if __name__ == "__main__":
    data_utils = FashionMnistUtils()

    # Load the train data
    train_data = data_utils.load_train_val_data()

    # Load test data
    test_data = data_utils.load_test_data()