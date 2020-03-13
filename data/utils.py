import os
from pathlib import Path
import pandas as pd
import numpy as np
import matlab.engine
import scipy.io
from tqdm import tqdm
import pickle


class Utils:
    def __init__(self):
        self.test_csv = Path(os.path.dirname(os.path.realpath(__file__))).joinpath("fashion-mnist_test.csv")
        self.train_csv = Path(os.path.dirname(os.path.realpath(__file__))).joinpath("fashion-mnist_train.csv")

    def load_data(self, partition="train"):
        if partition is "train":
            data_frame = pd.read_csv(self.train_csv, index_col='label', dtype=np.float, memory_map=True)
        elif partition is "test":
            data_frame = pd.read_csv(self.train_csv, index_col='label', dtype=np.float, memory_map=True)
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

    @staticmethod
    def extract_corf_contours(batch_data):

        # takes about 12 seconds
        eng = matlab.engine.start_matlab()
        eng.cd(r'D:\GitCode\fashion-mnist\CORFPushPull', nargout=0)
        # eng.addpath(eng.genpath(r"C:\Program Files\MATLAB\R2019b\toolbox\images\images"), nargout=0)
        eng.workspace['p'] = str(Path(r"C:/Program files/MATLAB/R2019b/toolbox/images/images"))
        a = eng.eval('addpath(genpath(p))')
        # eng.license("test", "Image_Toolbox", nargout=1)
        batch_binary_map = np.empty_like(batch_data).astype(int)
        batch_corf_response = np.empty_like(batch_data).astype(float)
        for idx, img in tqdm(enumerate(batch_data)):
            scipy.io.savemat("cache/input_image.mat", {"img": img})

            bm, cr = eng.contour_detection_from_python(1.0, 4.0, 1.8, 0.007, nargout=2)

            batch_binary_map[idx] = np.array(bm._data).reshape((28, 28), order='F')    # MATLAB stores in col major order
            batch_corf_response[idx] = np.array(cr._data).reshape((28, 28), order='F')

        return batch_binary_map, batch_corf_response


if __name__ == "__main__":
    data_utils = Utils()
    # todo: implement cache for training
    train_data_cache_file = Path(os.path.dirname(os.path.realpath(__file__))).joinpath("cache/train_data.pkl")
    if train_data_cache_file.exists():
        with open(str(train_data_cache_file), "rb") as f:
            train_grayscale_images, train_labels, train_binary_maps, train_corf_response = pickle.load(f)
    else:
        train_grayscale_images, train_labels = data_utils.load_data(partition="train")
        train_binary_maps, train_corf_response = data_utils.extract_corf_contours(train_grayscale_images)
        with open(str(train_data_cache_file), 'wb') as f:
            pickle.dump([train_grayscale_images, train_labels, train_binary_maps, train_corf_response], f)

    # todo: implement cache for testing
    test_data_cache_file = Path(os.path.dirname(os.path.realpath(__file__))).joinpath("cache/test_data.pkl")
    if test_data_cache_file.exists():
        with open(str(test_data_cache_file), "rb") as f:
            test_grayscale_images, test_labels, test_binary_maps, test_corf_response = pickle.load(f)
    else:
        test_grayscale_images, test_labels = data_utils.load_data(partition="test")
        test_binary_maps, test_corf_response = data_utils.extract_corf_contours(test_grayscale_images)
        with open(str(test_data_cache_file), 'wb') as f:
            pickle.dump([test_grayscale_images, test_labels, test_binary_maps, test_corf_response], f)
