# optml_miniprj


## Set up

### Install Python dependencies

### A Use poetry :
Use poetry instead or anaconda (poetry is recommended). Install poetry using:  https://python-poetry.org/docs/#installation

* [Do once when clone] Install all python dependencies: `poetry install`
* Get a shell where all python packages are installed and you are ready to go: `poetry shell`
* Add pip dependencies using poetry: `poetry add <dep_name>`
* Export the requirements.txt: `poetry export --without-hashes -f requirements.txt --output requirements.txt`

### B. Use pip:
* [Optional]Create your virtual environment
* Use `ip install -r requirements.txt`


### Get data dependencies

1. Download this file: https://drive.google.com/file/d/0B7EVK8r0v71pZjFTYXZWM3FlRnM/view?usp=sharing&resourcekey=0-dYn9z10tMJOBAkviAcfdyQ
2. unzip into folder a `<root>/data`

### [OPTIONAL] Create Inception features (only CUDA)
If you want the model to output FID and Inception score you need to run the following script:
```bash
mkdir inception_features
python prepare_inception.py --dataroot data/ --output inception_features/celeb_features.pkl
```



## Run locally

1. set up arguments in [args.py](./src/args.py) or pass them using `--<argname> <argvalue>`
2. run
    ```
    python run_gan.py --optim <name> # Name can be: [Adam, SGD, ExtraAdam, ExtraSGD]
    ```


## Run server

Use the following:
```bash
sbatch sbatch.script "--optim <name>"   # Name can be: [Adam, SGD, ExtraAdam, ExtraSGD]
```
