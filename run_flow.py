import utils.session_setup
from data.fashion_mnist_utils import FashionMnistUtils
from models.cnn import ConvolutionalNeuralNet1
from utils.visualize import Visualize
import logging

logger = logging.getLogger(__name__)


class RunFlow:
    def __init__(self):
        self.train_grayscale = None
        self.train_corf = None
        self.test_grayscale = None
        self.test_corf = None
        self.model_grayscale = None
        self.model_corf = None

    def execute_all(self, generate_noise):
        self.__load_data(generate_noise)
        # self.__train_model()
        # self.__select_best_model()
        # self.__evaluate_model()

    def __load_data(self, generate_noise=False):
        data_utils = FashionMnistUtils()
        logger.info("Loading the train data")
        (self.train_grayscale, self.val_grayscale), (self.train_corf, self.val_corf) = \
            data_utils.load_train_val_data(generate_noise=generate_noise)
        logger.info("Loading the train data")
        self.test_grayscale, self.test_corf = data_utils.load_test_data(generate_noise=generate_noise)

    def __train_model(self):
        logger.info("Training Grayscale Model")
        self.model_grayscale = ConvolutionalNeuralNet1()
        h1 = self.model_grayscale.train_flow(self.train_grayscale, self.val_grayscale, file_prefix="grayscale")
        Visualize.plot_training_curves(history=h1, title="Grayscale")

        logger.info("Training CORF Model")
        self.model_corf = ConvolutionalNeuralNet1()
        h2 = self.model_corf.train_flow(self.train_corf, self.val_corf, file_prefix="corf")
        Visualize.plot_training_curves(history=h2, title="CORF_Response_Maps")

        Visualize.plot_combined_training_curves(grayscale_history=h1, corf_history=h2)

    def __select_best_model(self):
        logger.info("Selecting Grayscale Model from tkinter GUI")
        self.model_grayscale = ConvolutionalNeuralNet1()
        self.model_grayscale.select_best_model("grayscale")

        logger.info("Selecting CORF Model from tkinter GUI")
        self.model_corf = ConvolutionalNeuralNet1()
        self.model_grayscale.select_best_model("corf")

    def __evaluate_model(self):
        from utils.session_setup import SessionSetup
        filename = SessionSetup().get_session_folder_path()

        logger.info("Evaluating Grayscale Model")
        self.model_grayscale = ConvolutionalNeuralNet1()
        self.model_grayscale.compile_model()
        self.model_grayscale.update_weights(
            str(filename.joinpath("grayscale_best_model.h5"))
        )
        grayscale_results = self.model_grayscale.predict_model(self.test_grayscale)
        Visualize.plot_confusion_matrix(grayscale_results, title="grayscale")

        logger.info("Evaluating CORF Model")
        self.model_corf = ConvolutionalNeuralNet1()
        self.model_corf.compile_model()
        self.model_corf.update_weights(
            str(filename.joinpath("corf_best_model.h5"))
        )
        corf_results = self.model_corf.predict_model(self.test_corf)
        Visualize.plot_confusion_matrix(corf_results, title="corf")


if __name__ == "__main__":
    flow = RunFlow()
    flow.execute_all(generate_noise=True)
    pass
