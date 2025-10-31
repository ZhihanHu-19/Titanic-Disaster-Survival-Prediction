# Titanic-Disaster-Survival-Prediction (Python & R, Dockerized)

This repo builds two reproducible containers (Python and R) to train and run a simple logistic regression model on the Titanic dataset:

- **Python**: `scikit-learn` pipeline  
- **R**: `glm` (binomial)

to pridict the survial of a passenger based on some demographics.
The repo can clone and run both containers in a few steps.

---

## 1. Prerequisites

- **Docker Desktop** 
- **GitHub**

---

## 2. Repository Structure
```bash
── src/
    ├── data/
        ├── train.csv
        ├── test.csv
        ├── gender_submission.csv (example submission file)
    ├── py_app/
        ├── Dockerfile
        ├── requirements.txt
        └── py_app.py
    └── r_app/
        ├── Dockerfile
        ├── install_packages.R        
        └── r_app.R            
```
> Data files are intentionally **not** in the repo. '.gitignore' excludes 'data/' and '*.csv'.

---

## 3. Get the Data

1. Download **'train.csv'** and **'test.csv'** for the Titanic dataset: (https://www.kaggle.com/competitions/titanic/data)
2. Put both files under:


    src/data/train.csv


    src/data/test.csv

---

## 4. Run the **Python** container

## Build

### from repo root
```bash
docker build -t titanic-app -f src/py_app/Dockerfile .
```
## Run
```bash
docker run --rm -v "$PWD/src/data:/app/src/data" titanic-app
```

What you'll see:
1. Step-by-step prints (load-> preprocess -> train -> accuracy -> predict)
2. Training accuracy
3. First 10 test predictions
4. Output file: src/data/predictions.csv

## 5. Run the **R** container

### Build

#### from repo root
docker build -t titanic-r -f src/r_app/Dockerfile .

### Run
docker run --rm -v "$PWD/src/data:/app/data" titanic-r

What you'll see:
1. Step-by-step prints (load-> preprocess -> train -> accuracy -> predict)
2. Training accuracy
3. First 10 test predictions
4. Output file: src/data/predictions_r.csv

## 6. Quick Start

### 0) Clone
```bash
git clone https://github.com/ZhihanHu-19/Titanic-Disaster-Survival-Prediction.git
```
After that, a folder named "Titanic-Disaster-Survival-Prediction" will appear in the directory.

### 1) Put data
    src/data/train.csv
    src/data/test.csv

### 2) Python container
```bash
docker build -t titanic-app -f src/py_app/Dockerfile .
```
```bash
docker run --rm -v "$PWD/src/data:/app/src/data" titanic-app
```
### -> outputs src/data/predictions.csv

### 3) R container
```bash
docker build -t titanic-r -f src/r_app/Dockerfile .
```

```bash
docker run --rm -v "$PWD/src/data:/app/data" titanic-r
```
### -> outputs src/data/predictions_r.csv