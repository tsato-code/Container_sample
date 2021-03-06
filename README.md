# Sample code for machine learning container
## Overview
- This repository is a sample code for a machine learning container.
- Train and evaluate using LightGBM or TensorFlow.
- The [OnlineNewsPopularity](https://archive.ics.uci.edu/ml/datasets/Online+News+Popularity) dataset is used.
<div align="center">
<img src="docs/fig01.png" width=650pt>
</div>

## Module I/O interface
| Module   	| Description					| Input 	            | Output           	            | I/O restriction |
|----------	|----------------------------	|-------------------	|----------------------------	|- |
| Prepare  	| Get dataset and transforme 	| Data file (*.csv) 	| Feature files (*.pkl)         | ---  |
| Train    	| Train the model 				| Feature files (*.pkl) | Model file (*.pkl)   	        | --- |
| 			| 					        	| ---			    	| Importance plot file (*.png)  | --- |
|			|								| ---				    | Loss plot file (*.png)		| --- |
| Evaluate  | Evaluate the Model 			| Model file (*.pkl) 	| ---       					| --- |
|          	|								| Feature files (*.pkl) | ---							| --- |

## Setup
You need to prepare [docker](https://www.docker.com/) environment.

## Usage
[Kaggle Python docker image](https://console.cloud.google.com/gcr/images/kaggle-images/GLOBAL/python?gcrImageListsize=30) (16 GB image size) is used.

### Quikstart
#### Remove container
```
$ inv remove
```

#### Run container
```
$ inv run
```

#### Step 1: Prepare
```
$ inv prepare
```

#### Step 2: Train
```
$ inv train
```

#### Step 3: Evaluate
```
$ inv evaluate
```

### Test
```
$ pytest
```

## Folder tree
```
.
├── Dockerfile
├── README.md
├── docker-compose.yml
├── docs
│   ├── fig01.png
│   └── overview_description.key
├── invoke.yml
├── project
│   ├── config.yml
│   ├── data
│   │   ├── pickle
│   │   │   ├── X_test.pkl
│   │   │   ├── X_train.pkl
│   │   │   ├── Y_test.pkl
│   │   │   └── Y_train.pkl
│   │   └── raw
│   │       ├── OnlineNewsPopularity
│   │       │   ├── OnlineNewsPopularity.csv
│   │       │   └── OnlineNewsPopularity.names
│   │       └── OnlineNewsPopularity.zip
│   ├── figure
│   │   ├── lbg_loss.png
│   │   └── lgb_importance.png
│   ├── logs
│   │   ├── evaluate.py.log
│   │   ├── logger.py.log
│   │   ├── prepare.py.log
│   │   └── train.py.log
│   ├── models
│   │   └── lgb.pkl
│   ├── notebook
│   │   ├── lgb.ipynb
│   │   └── nn_3layer.ipynb
│   ├── source
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── evaluate.py
│   │   ├── hello.py
│   │   ├── logger.py
│   │   ├── prepare.py
│   │   └── train.py
│   └── tests
│       └── test_hello.py
└── tasks.py
```