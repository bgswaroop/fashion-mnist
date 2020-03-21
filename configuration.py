import os
from pathlib import Path

# Set up the directories for the current training / evaluation session
root_session_dir = Path(os.path.dirname(os.path.abspath(__file__)))  # This is the project root

# session_dir = root_session_dir.joinpath(r"runtime_data/CleanTraining_CleanPredictions")
# session_dir = root_session_dir.joinpath(r"runtime_data/CleanTraining_NoisePredictions")
# session_dir = root_session_dir.joinpath(r"runtime_data/NoiseTraining_NoisePredictions")
session_dir = Path(r"None")  # setting session_dir to an invalid path, will create a new session directory
