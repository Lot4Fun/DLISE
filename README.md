# DLISE: Deep Learning Internal Structure Estimation

## Overview

Estimate a vertical profile (Temperature/Salinity) of ocean from SSH/SST by Deep Learning. This trial is apllied in the North Pacific Subtropical Gyre.

## Requirement

- Python 3.8
- CUDA 10.2

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

### Detect

1. Set path to trained weights at the `trained_weight_path` in the `config.json` created in train phase.

    ```python
    ADD LATER
    ```

2. Change other configurations above `detect`. Especialy `conf_threshold` affects the final results of detection.

3. Run script in detection mode.

    ```bash
    python execute.py detect -c /PATH/TO/config.json -x /INPUT/DIR [-y /OUTPUT/DIR]
    ```

## Data

### Directory structure

```text
ADD LATER
```
