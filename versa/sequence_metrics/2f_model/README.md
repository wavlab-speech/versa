# 2f-model

MATLAB implementation heavily adapted from [Perceptual-Coding-In-Python](https://github.com/stephencwelch/Perceptual-Coding-In-Python/tree/master). MATLAB engine is necessary for the code to run properly. A few considerations:

## 1. Install MATLAB python engine. For macos this entails the following commands:
`cd /Applications/MATLAB_R2019b.app/extern/engines/python`
`python setup.py install`

## 2. Export the matlab python engine path
`export PYTHONPATH=/Applications/MATLAB_R2019b.app/extern/engines/python/build/lib/:$PYTHONPATH`
Alternatively one could also add the path directly from within the `compute_2f_metric.py` script:
```
import sys
sys.path.append("/Applications/MATLAB_R2019b.app/extern/engines/python/build/lib/")
```

## 3. Python version:
Lastly, not all MATLAB versions are compatible with any python version, therefore please check the following [compat. matrix](https://www.mathworks.com/support/requirements/python-compatibility.html) for picking the python version suiting your matlab version. 

This code has been tested with MATLAB2019b/python 3.7