# optml_miniprj


## Poetry usage (recommended):
Use poetry instead or anaconda (poetry is recommended). Install poetry using:  https://python-poetry.org/docs/#installation

* [Do once when clone] Install all python dependencies: `poetry install`
* Get a shell where all python packages are installed and you are ready to go: `poetry shell`
* Add pip dependencies using poetry: `poetry add <dep_name>`
* Export the requirements.txt: `poetry export --without-hashes -f requirements.txt --output requirements.txt`



## Set it up:

1. Download this file: https://drive.google.com/file/d/0B7EVK8r0v71pZjFTYXZWM3FlRnM/view?usp=sharing&resourcekey=0-dYn9z10tMJOBAkviAcfdyQ
2. unzip into folder a `<root>/data`


## Run locally

1. set up arguments in [args.py](./src/args.py) or pass them using "--<argname> <argvalue>"
2. run `python run_gan.py`


## Run server

Use the following:
```bash
sbatch sbatch.script "--optim <name>"   # Name can be: 'SGD, Adam, ExtraAdam, ExtraSGD)
```
