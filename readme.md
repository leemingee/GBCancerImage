
# GBCancerImage
Implementation of a CNN for tumor classification in the [Camelyon16 Grand Challenge](https://camelyon17.grand-challenge.org/)


# Code Structure

There are two ways that you can use this project code, one is using colab notebook runtime, by cloning the needed python modules into the runtime workspace, you can run the code from scratch.

Also, which is more preferred by myself, download this git repo and open it as a project in Pycharm. By running the whole program via the main.py, the whole program can be easily tested and debugged.

The whole file tree in this project:
```bash
├───.idea 
├───.ipynb_checkpoints
├───checkpoint # contains the checkpoint of model
├───data # contains the data downloaded from Google bucket
│   └───091
│       ├───2 # used for test in demo
│       │   ├───all
│       │   └───tissue_only
│       └───3 # used for train in demo
│           ├───no_tumor
│           └───tumor
├───test # contain all the saved pics
├──__pycache__
├───module # Contain the tf.keras subclass which construct the model
├───utils # data preprocess
├───test # contain all the saved pics
└───ops # some useful functions can for tf

```

# Requirements
1. Python
2. [OpenSlide](https://openslide.org/download/) 3.4.1 (C library). If installing through [GitHub](https://github.com/openslide/openslide), additional packages are required, all of which can be found in the Readme.   
3. OpenSlide Python
4. tensorflow-gpu (Conda [distribution](https://anaconda.org/anaconda/tensorflow-gpu))
5. matplotlib
6. numpy
7. scikit-image
8. scikit-learn
9 pandas
10. opencv-python

Other notes: 
OpenSlide 3.4.1  is required, as early versions of the package do not support Phillips TIFF, the format of the annotated whole slide images (WSI).  

# Implementation
- This program is programmed and tested on one Precision 7820 Tower workstation, Intel Xeon Silver 4116 CPU and NVIDIA Quadro P5000 16GB. And the training process can be really quick.
- If runs on Colab, it may take few minutes.

This model is trained based on tif pics number 091 in the [dataset](https://camelyon17.grand-challenge.org/), and tested on the tif pics number 110.

There are still four todo for this program, welcome to reach [me](ming.li2@columbia.edu) about any questions or interests.
