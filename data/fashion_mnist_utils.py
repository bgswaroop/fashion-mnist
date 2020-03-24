import os
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import pickle
import logging
import imageio

logger = logging.getLogger(__name__)


class FashionMnistUtils:
    def __init__(self):
        self.pwd = os.path.dirname(os.path.realpath(__file__))
        self.test_csv = Path(self.pwd).joinpath("fashion-mnist_test.csv")
        self.train_csv = Path(self.pwd).joinpath("fashion-mnist_train.csv")
        self.image_shape = (28, 28)
        self.pad_width = 4
        self.pad_image_shape = tuple([dim + 2 * self.pad_width for dim in self.image_shape])

    def __load_data(self, partition="train", generate_noise=False):
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
            data[idx] = np.array(data_frame.iloc[[idx]]).reshape(self.image_shape) / 255.0
            # pad the images
            data[idx] = np.pad(data[idx], self.pad_width, mode="constant", constant_values=0)
            labels[idx] = int(data_frame.index[idx])

        # Adding a Gaussian noise to the images
        if generate_noise:
            for idx in np.arange(num_samples):
                data[idx] += np.random.normal(loc=0, scale=0.1, size=self.pad_image_shape)

        return np.stack(data), labels

    def __extract_corf_contours(self, data):
        # Python to MATLAB connectivity
        # takes about 12 seconds
        import matlab.engine
        eng = matlab.engine.start_matlab()
        eng.cd(str(Path(self.pwd).parent.joinpath("CORFPushPull")), nargout=0)

        binary_maps = np.empty_like(data).astype(int)
        corf_response = np.empty_like(data).astype(float)
        for idx, img in tqdm(enumerate(data)):

            # Compute the CORF response
            matlab_img = matlab.double(initializer=img.tolist(), size=img.shape, is_complex=False)
            bm, cr = eng.contour_detection_from_python(matlab_img, 1.0, 4.0, 1.8, 0.007, nargout=2)

            # MATLAB stores data in col major order
            binary_maps[idx] = np.array(bm._data).reshape(self.pad_image_shape, order='F')
            corf_response[idx] = np.array(cr._data).reshape(self.pad_image_shape, order='F')

        return binary_maps, corf_response

    def load_train_val_data(self, val_split=0.2, generate_noise=False):
        """
        Load the train + validation tensorflow data
        :param generate_noise: boolean - Generates Gaussian noise and add to the input data
        :param val_split: float value must be in the range 0-1
        :return: (train_grayscale, val_grayscale), (train_corf, val_corf)
        """

        # Load the data
        if generate_noise:
            train_data_cache_file = Path(self.pwd).joinpath("cache/f_mnist_noise_train.pkl")
        else:
            train_data_cache_file = Path(self.pwd).joinpath("cache/f_mnist_clean_train.pkl")

        if train_data_cache_file.exists():
            with open(str(train_data_cache_file), "rb") as f:
                grayscale_images, labels, _, corf_images = pickle.load(f)
        else:
            # pad and load the images
            grayscale_images, labels = self.__load_data(partition="train", generate_noise=generate_noise)
            binary_maps, corf_images = self.__extract_corf_contours(grayscale_images)

            # crop the images
            grayscale_images = grayscale_images[:, self.pad_width:-self.pad_width, self.pad_width:-self.pad_width]
            binary_maps = binary_maps[:, self.pad_width:-self.pad_width, self.pad_width:-self.pad_width]
            corf_images = corf_images[:, self.pad_width:-self.pad_width, self.pad_width:-self.pad_width]

            with open(str(train_data_cache_file), 'wb') as f:
                pickle.dump([grayscale_images, labels, binary_maps, corf_images], f)

        # Reshaping the arrays to match the tensorflow inputs of 4D arrays
        labels = labels.astype(int)
        grayscale_images = np.expand_dims(grayscale_images, axis=-1)
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

    def load_test_data(self, generate_noise=False):
        """
        Load the tensorflow data set for evaluation
        :param generate_noise: boolean - Generates Gaussian noise and add to the input data
        :return: test_grayscale, test_corf
        """

        # Load the data
        if generate_noise:
            test_data_cache_file = Path(self.pwd).joinpath("cache/f_mnist_noise_test.pkl")
        else:
            test_data_cache_file = Path(self.pwd).joinpath("cache/f_mnist_clean_test.pkl")

        if test_data_cache_file.exists():
            with open(str(test_data_cache_file), "rb") as f:
                grayscale_images, labels, _, corf_images = pickle.load(f)
        else:
            # pad and load the images
            grayscale_images, labels = self.__load_data(partition="test", generate_noise=generate_noise)
            binary_maps, corf_images = self.__extract_corf_contours(grayscale_images)

            # crop the images
            grayscale_images = grayscale_images[:, self.pad_width:-self.pad_width, self.pad_width:-self.pad_width]
            binary_maps = binary_maps[:, self.pad_width:-self.pad_width, self.pad_width:-self.pad_width]
            corf_images = corf_images[:, self.pad_width:-self.pad_width, self.pad_width:-self.pad_width]

            with open(str(test_data_cache_file), 'wb') as f:
                pickle.dump([grayscale_images, labels, binary_maps, corf_images], f)

        # Reshaping the arrays to match the tensorflow inputs of 4D arrays
        labels = labels.astype(int)
        grayscale_images = np.expand_dims(grayscale_images, axis=-1)
        corf_images = np.expand_dims(corf_images, axis=-1)

        # Prepare the tensorflow data sets
        test_grayscale = tf.data.Dataset.from_tensor_slices(
            (tf.convert_to_tensor(grayscale_images), tf.one_hot(labels, depth=10))
        )
        test_corf = tf.data.Dataset.from_tensor_slices(
            (tf.convert_to_tensor(corf_images), tf.one_hot(labels, depth=10))
        )

        return test_grayscale, test_corf

    def save_generated_images(self):

        logger.info("Begin saving the clean training images to disk")
        data_file = Path(self.pwd).joinpath("cache/f_mnist_clean_train.pkl")
        if data_file.exists():
            with open(str(data_file), "rb") as f:
                grayscale_images, _, _, corf_maps = pickle.load(f)

            for idx in np.arange(len(grayscale_images)):
                # Save the clean images
                img_file = Path(self.pwd).joinpath("cache/clean_images/{}.png".format(str(idx).zfill(5)))
                imageio.imwrite(img_file, grayscale_images[idx])

                # Save the corf maps generated from clean images
                img_file = Path(self.pwd).joinpath("cache/corf_maps_from_clean_images/{}.png".format(str(idx).zfill(5)))
                imageio.imwrite(img_file, corf_maps[idx])

        logger.info("Begin saving the noisy training images to disk")
        data_file = Path(self.pwd).joinpath("cache/f_mnist_noise_train.pkl")
        if data_file.exists():
            with open(str(data_file), "rb") as f:
                grayscale_images, _, _, corf_maps = pickle.load(f)

            for idx in np.arange(len(grayscale_images)):
                # Save the noisy images
                img_file = Path(self.pwd).joinpath("cache/noisy_images/{}.png".format(str(idx).zfill(5)))
                imageio.imwrite(img_file, grayscale_images[idx])

                # Save the corf maps generated from noisy images
                img_file = Path(self.pwd).joinpath("cache/corf_maps_from_noisy_images/{}.png".format(str(idx).zfill(5)))
                imageio.imwrite(img_file, corf_maps[idx])


if __name__ == "__main__":
    data_utils = FashionMnistUtils()
    # Load the train data
    train_data = data_utils.load_train_val_data()

    # Load test data
    test_data = data_utils.load_test_data()

    # Save the generated data for visualization
    data_utils.save_generated_images()
