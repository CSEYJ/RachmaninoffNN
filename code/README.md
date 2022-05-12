This directory contains the reimplementation of DeepJ in PyTorch.  


## Setup Instructions
Our program was tested at the GCP VM instance installed with Ubuntu 20.04 LTS and Tesla T4 GPU. To preprocess and train the model, the device should have at least 120 GB of storage and CUDA compatible GPU accelerator. While it will likely work with other versions, we tested our framework with `Python 3.8`, `CUDA 11.4`, `Pytorch 1.11.0`, and `NVIDIA 470.103.01 driver`.

### Dataset
Download the dataset with the following command:
```
sh ./download.sh
```
This will generate the required dataset to `../data` directory.

### Environment Setup
Install the dependencies:  
```
sh ./setup.sh
```

## Usage
### Training
To train the model, type:
```
python3 train.py
```
Following command will save the preprocessed data into `../data` directory. At next call to `train.py,` it will used the existing preprocessed data to avoid the redundant works.

During the execution, the program will save the weight file, `model.h5` along with the cached data (epoch and batch positions) under `./out` directory.

To generate the midi files, type:

```
python3 generate.py
```
It will generate three midi files under `./out/samples`.
