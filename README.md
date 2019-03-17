# __Internal Structure Estimation__
IN PRODUCTION

## Description
Estimate internal structure of ocean by Deep Learning.

## Development memo

### Limitation
- This estimation function is applied in the North Pacific Subtropical Gyre.

## Demo
```
python impulso.py estimate -e XXXX-XXXX-XXXX -m X -x X=DIR -y Y-DIR
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
To resume training, specify MODEL-ID.
```
python impulso.py train -e EXPERIMENT-ID [-m MODEL-ID]
```

### Test
```
python impulso.py test -e EXPERIMENT-ID -m MODEL-ID
```

### Predict
```
python impulso.py predict -e EXPERIMENT-ID -m MODEL-ID -x INPUT_DIR -y OUTPUT_DIR
```

## License
- Permitted: Private Use  
- Forbidden: Commercial Use  

## Author
[LotFun](https://github.com/pystokes)

## Specification
