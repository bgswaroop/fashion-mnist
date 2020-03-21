import os
import sys
import logging
from datetime import datetime
# import tensorflow as tf
from configuration import root_session_dir, session_dir


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class SessionSetup(metaclass=Singleton):
    def __init__(self):
        self.__session_folder_path = None

        self.__setup_session_directories()
        self.__setup_logger()

    def __setup_session_directories(self):
        if session_dir.exists() and session_dir.is_dir():
            self.__session_folder_path = session_dir
        else:
            default_time_format = '%Y-%m-%d_%H.%M.%S'
            session_folder_name = datetime.now().strftime(default_time_format)
            self.__session_folder_path = root_session_dir.joinpath("runtime_data", session_folder_name)
            os.makedirs(str(self.__session_folder_path))

    def __setup_logger(self):
        # root logger
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

        log_filepath = str(self.__session_folder_path.joinpath("run.log"))
        file_handler = logging.FileHandler(log_filepath)
        sys.stdout = open(log_filepath, "a")
        sys.stderr = open(log_filepath, "a")

        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # # Log messages to console
        # stream_handler = logging.StreamHandler()
        # stream_handler.setFormatter(formatter)
        # logger.addHandler(stream_handler)

        # Log tensorflow messages to the specified log file
        # tf_logger = tf.get_logger()
        tf_logger = logging.getLogger('tensorflow')
        tf_logger.setLevel(logging.DEBUG)
        tf_logger.addHandler(file_handler)
        # I'm not sure why still I have some logs from tensorflow which get logged to my console

        logger.info("Logger file successfully initialized")

    def get_session_folder_path(self):
        return self.__session_folder_path


if __name__ == "__main__" or __name__ == "utils.session_setup":
    SessionSetup()
