# __Internal Structure Estimation__
IN PRODUCTION

## Description
Estimate a vertical profile (Temperature/Salinity) of ocean from SSH/SST by Deep Learning.
This trial is apllied in the North Pacific Subtropical Gyre.

## Concept
![Concept](https://github.com/pystokes/internal_structure/blob/master/documents/Conceptual_diagram.png)

## Demo
```
python impulso.py inference -m MODEL-PATH -m X -x X-DIR [-y Y-DIR]
```

## Requirement
Python 3.6  
torch==1.0.1  
torchvision==0.2.2.post3  

## Install
```
git clone https://github.com/pystokes/internal_structure.git
```

## Usage
### Create dataset
```
python impulso.py dataset
```

### Prepare
```
python impulso.py prepare -d DATA-ID
```

### Train
To resume training, specify MODEL-PATH.
```
python impulso.py train -e EXPERIMENT-ID [-m MODEL-PATH]
```

### Inference
```
python impulso.py inference -m MODEL-PATH -x INPUT_DIR [-y OUTPUT_DIR]
```

## Author
[LotFun](https://github.com/pystokes)
