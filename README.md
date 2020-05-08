# fashion-mnist
Experiments with object recognition using fashion MNIST dataset

### Entry point into the code 
```run_flow.py```

### Dependencies
- python 3.7 (also works with python 3.6, partially tested)
- matlab engine for python - follow the instructions [here](https://www.mathworks.com/help/matlab/matlab-engine-for-python.html) to set up MATLAB engine for python.
- ```requirements.txt```

### Setting up of MATLAB Engine for python on [Peregrine](https://www.rug.nl/society-business/centre-for-information-technology/research/services/hpc/facilities/peregrine-hpc-cluster?lang=en)
- ```cd "matlabroot\extern\engines\python"```
- ```python setup.py build -b $HOME/build install```
