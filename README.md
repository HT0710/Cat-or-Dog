# <p align="center">Cat or Dog</p>

![16_prediction](https://user-images.githubusercontent.com/95120444/236637964-e35077ac-50a1-46b4-9e31-6f36cee74b72.png)

---
## Table of Contents
- [Cat or Dog](#cat-or-dog)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Install](#install)
    - [Version](#version)
    - [Requirements](#requirements)
  - [Usage](#usage)
    - [Testing](#testing)
    - [Training](#training)
  - [Contact](#contact)


## Introduction
Is it a Dog or a Cat?

In this project, I trying to classify images of dogs and cats using CNN deep learning. I build from scratch the `VGG` neural network architecture. The model achieved **97% accuracy** on test set.

The dataset can be downloaded from Kaggle: [Link](https://www.kaggle.com/datasets/karakaggle/kaggle-cat-vs-dog-dataset)

My pre-trained model can be downloaded: [Here](https://drive.google.com/file/d/1xDxMeu_OgLQvPK-98-h9XNe47kEvIYKY/view?usp=sharing)

![VGG](https://user-images.githubusercontent.com/95120444/236639251-27372056-fcab-4c20-90f7-dae2fec49a56.png)

## Install
- Download directly: [Link](https://github.com/HT0710/Cat-or-Dog/archive/refs/heads/main.zip)
- Clone with `Git`
    ```bash
    git clone https://github.com/HT0710/Cat-or-Dog.git
    ```

### Version
- **`Python 3.9`**

### Requirements
```bash
pip install -r requirements.txt
```
> Note: CUDA is required for using GPU.
> Full GPU setup instructions can be found: [Here](https://gist.github.com/HT0710/639ec6ad96f9c46e0d209ba2e50ee168)

## Usage
### Testing
1. Download my pre-trained model: [Here](https://drive.google.com/file/d/1xDxMeu_OgLQvPK-98-h9XNe47kEvIYKY/view?usp=sharing)
2. Create and put it into a folder called `Models`
3. Run `test.py`
```bash
python test.py
``` 
Use arguments:
```bash
python test.py --help

# -r: Random test in the Dataset (Default if run without arguments)
# -i: Test on a single image
# -f: Test on a folder of image
```
Use camera `cam_test.py`
```bash
python cam_test.py

# Output video size and full screen mode can change in file
``` 
### Training
> Full GPU setup instructions can be found: [Here](https://gist.github.com/HT0710/639ec6ad96f9c46e0d209ba2e50ee168)

Required:
1. Linux distro (Ubuntu based recomended)
2. Nvidia GPU and CUDA setup
3. Dataset (Can be download in Kaggle: [Link](https://www.kaggle.com/datasets/karakaggle/kaggle-cat-vs-dog-dataset))

Setting hyperparameter in `train.py`
```bash
python train.py
``` 
Use arguments:
```bash
python train.py --help

# -bs: Set batch size (Default: 16)
# -e: Set number of epochs (Default: 20)
# -lr: Set learning rate (Default: 0.00001)
# -ss: Model save frequency (Default: 2)
```

## Contact
Open an issue: [New issue](https://github.com/HT0710/Cat-or-Dog/issues/new)

Mail: pthung7102002@gmail.com

---