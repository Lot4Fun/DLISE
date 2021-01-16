# DLISE: Deep Learning Internal Structure Estimation

## Overview

Sea surface features are strongly related to vertical structure in the ocean (Ex. temperature, salinity, potential density, etc). This trial attempts to predict the vertical structure (temperature and salinity) from sea surface height (SSH) and temperature (SST) by Deep Learning. We assume that Argo profiles are obtained in the North Pacific Subtropical Gyre since internal structures in the ocean depend on the region.

Spacial resolution of our dataset is .25 degree x 0.25 degree and it is not enough to resolve submeso-scale (~10km) phenomenon. Therefore our proposed method aims to predict the difference in the order of meso-scale (~100km).

Proposed method predicts three-dimensional grid data with only in-situ observation data (Argo profiles) and do not use any numerical simulation model (Ex. [OFES](http://www.jamstec.go.jp/ofes/)).

## Requirement

In addition to following system requeirements, you need to create [Copernicus Marine Environment Monitoring Service (CMEMS)](https://marine.copernicus.eu/) account and get login ID and PASSWORD.

- Python 3.8

    ```text
    $ python
    Python 3.8.0 (default, Oct 28 2019, 16:14:01) 
    [GCC 8.3.0] on linux
    Type "help", "copyright", "credits" or "license" for more information.
    ```

- CUDA 10.2

    ```text
    $ nvcc -V
    nvcc: NVIDIA (R) Cuda compiler driver
    Copyright (c) 2005-2019 NVIDIA Corporation
    Built on Wed_Oct_23_19:24:38_PDT_2019
    Cuda compilation tools, release 10.2, V10.2.89
    ```

- lhasa (`lha` command is necessary to )

    ```bash
    sudo apt-get install lhasa
    ```

## Install

Clone this repository.

```bash
git clone https://github.com/pystokes/DLISE.git
```

Install necessary libraries.

```bash
cd DLISE
pip install -r requirements.txt
```

## Usage

Only the following patterns to load trained weights are supported.

|Support|Train on|Detect on|
|:---:|:---:|:---:|
|:heavy_check_mark:|Single-GPU|Single-GPU|
|:heavy_check_mark:|Multi-GPU|Single-GPU|
|Not supported|Single-GPU|Multi-GPU|
|:heavy_check_mark:|Multi-GPU|Multi-GPU|

### Download dataset

Change configuratoin in [`tools/download_dataset.sh`](https://github.com/pystokes/DLISE/blob/master/tools/download_dataset.sh) and run following command. If you need to download CMEMS dataset, enter CMEMS ID and PASSWORD.

```bash
cd DLISE/tools
sudo bash download_dataset.sh

CMEMS ID: YOUR_ID
CMEMS Password: YOUR_PASWORD
```

### Preprocess

ADD LATER

### Train

1. Modify `Requirements` in [config.py](https://github.com/pystokes/DLISE/blob/master/config.py) at first.

    ```python
    ADD LATER
    ```

2. Set hyperparameters for train in [config.py](https://github.com/pystokes/DLISE/blob/master/config.py).

    ```python
    ADD LATER
    ```

3. Run script in train mode.

    ```bash
    python execute.py train [-g GPU_ID]
    ```

    If train on multi-GPU, separate GPU IDs with commas.

    ```bash
    # Example: Use two GPUs (0 and 1)
    python execute.py train -g 0,1
    ```

### Prediction

1. Set path to trained weights at the `trained_weight_path` in the `config.json` created in train phase.

    ```python
    ADD LATER
    ```

2. Change other configurations in `predict`.

    ```python
    ADD LATER
    ```

3. Run script in detection mode.

    ```bash
    python execute.py predict -c /PATH/TO/config.json -x /INPUT/DIR [-y /OUTPUT/DIR]
    ```
