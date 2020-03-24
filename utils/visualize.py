from utils.session_setup import SessionSetup
from matplotlib import pyplot as plt
# import pickle


class Visualize:
    @staticmethod
    def plot_training_curves(history, title):
        # grayscale_history = pickle.load(session_dir.joinpath("grayscale_train_history.pkl"))
        # corf_history = pickle.load(session_dir.joinpath("corf_train_history.pkl"))

        epochs = history.epoch
        # Plot the cost function (Loss)
        training_loss = history.history["loss"]
        validation_loss = history.history["val_loss"]

        plt.plot(epochs, training_loss, label='Training Loss')
        plt.plot(epochs, validation_loss, label='Validation Loss')

        plt.legend()
        plt.title("{} - Learning Plots".format(title))
        plt.xlabel('Epochs')
        plt.ylabel('Avg. Cross Entropy Loss')
        plt.tight_layout()

        plot_path = SessionSetup().get_session_folder_path().joinpath("{}_learning_plots_loss.png".format(title))
        plt.savefig(plot_path)
        plt.show()

        # Plot the accuracy
        # plt.close()
        training_acc = history.history["acc"]
        validation_acc = history.history["val_acc"]

        plt.plot(epochs, training_acc, label='Training Accuracy')
        plt.plot(epochs, validation_acc, label='Validation Accuracy')

        plt.legend()
        plt.title("{} - Learning Plots".format(title))
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.tight_layout()

        plot_path = SessionSetup().get_session_folder_path().joinpath("{}_learning_plots_acc.png".format(title))
        plt.savefig(plot_path)
        plt.show()

    @staticmethod
    def plot_combined_training_curves(grayscale_history, corf_history):
        epochs = grayscale_history.epoch
        # Plot the cost function (Loss)
        training_acc = grayscale_history.history["loss"]
        validation_acc = grayscale_history.history["val_loss"]
        plt.plot(epochs, training_acc, label='Grayscale Training Loss')
        plt.plot(epochs, validation_acc, label='Grayscale Validation Loss')

        training_acc = corf_history.history["loss"]
        validation_acc = corf_history.history["val_loss"]
        plt.plot(epochs, training_acc, label='CORF Maps Training Loss')
        plt.plot(epochs, validation_acc, label='CORF Maps Validation Loss')

        plt.legend()
        plt.title("Learning Plots")
        plt.xlabel('Epochs')
        plt.ylabel('Avg. Cross Entropy Loss')
        plt.tight_layout()

        plot_path = SessionSetup().get_session_folder_path().joinpath("Learning_plots_combined_loss.png")
        plt.savefig(plot_path)
        plt.show()

        # Plot the cost function (Loss)
        training_acc = grayscale_history.history["acc"]
        validation_acc = grayscale_history.history["val_acc"]
        plt.plot(epochs, training_acc, label='Grayscale Training Accuracy')
        plt.plot(epochs, validation_acc, label='Grayscale Validation Accuracy')

        training_acc = corf_history.history["acc"]
        validation_acc = corf_history.history["val_acc"]
        plt.plot(epochs, training_acc, label='CORF Maps Training Accuracy')
        plt.plot(epochs, validation_acc, label='CORF Maps Validation Accuracy')

        plt.legend()
        plt.title("Learning Plots")
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.tight_layout()

        plot_path = SessionSetup().get_session_folder_path().joinpath("Learning_plots_combined_acc.png")
        plt.savefig(plot_path)
        plt.show()

    @staticmethod
    def plot_confusion_matrix(results, title=None):
        import seaborn as sn
        import pandas as pd

        df_cm = pd.DataFrame(results["confusion_matrix"].numpy(), range(10), range(10))
        plt.figure(figsize=(10, 7))
        sn.set(font_scale=1.4)  # for label size
        sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}, fmt='g')  # font size
        plt.yticks(rotation=0)

        # plt.text(4, 14, "Accuracy : {}".format(results["accuracy             "]))

        plt.title("Confusion Matrix - {} (Overall Accuracy: {})".
                  format(title.title(), round(results["accuracy             "], 3)))
        plt.xlabel('Predicted')
        plt.ylabel('Ground Truth')
        plot_path = SessionSetup().get_session_folder_path().joinpath("{}_confusion_matrix.png".format(title))
        plt.savefig(plot_path)

        plt.show()
