# Evaluating Optimization Methods for Generative Adversarial Networks

In this project, we investigate and compare different optimization methods for GAN training

This project is the work of:
* Orest Gkini
* Theodoros Bitsakis
* Theofilos Belmpas

<hr>

## Set up

### Install Python dependencies
You have 2 options:

**A. Use poetry :**

Install poetry using:  https://python-poetry.org/docs/#installation

* Install all python dependencies running `poetry install` on the root dir
* Get a shell where all python packages are installed and you are ready to go: `poetry shell`
* Add pip dependencies using poetry: `poetry add <dep_name>`
* Export the requirements.txt: `poetry export --without-hashes -f requirements.txt --output requirements.txt`

**B. Use pip:**

* [Optional] Create your virtual environment
* Use `pip install -r requirements.txt`


### Get data dependencies

1. Download this file: https://drive.google.com/file/d/0B7EVK8r0v71pZjFTYXZWM3FlRnM/view?usp=sharing&resourcekey=0-dYn9z10tMJOBAkviAcfdyQ
2. unzip into folder a `<root>/data`

### [OPTIONAL] Create Inception features
If you want the model to output FID and Inception score you need to run the following script:
```bash
mkdir inception_features
python prepare_inception.py --dataroot data/ --output inception_features/celeb_features.pkl
```

<hr>

## Run

1. Set up arguments in [args.py](./src/args.py) or pass at runtime them using `--<argname> <argvalue>`
2. Start training:
    * Run locally
        ```
        python run_gan.py --optim <name> # Name can be: [Adam, SGD, ExtraAdam, ExtraSGD]
        ```
    * Run on IZAR server
        ```bash
        sbatch sbatch.script "--optim <name>"   # Name can be: [Adam, SGD, ExtraAdam, ExtraSGD]
        ```

<hr>

## Reproducibility

**Run-logs to csv**

After you have run the models and created some data under the `tensorboard` directory, run the following to extract a dataframe from the run logs:
```bash
python cmd/parse_logs.py --tb_dir tensorboard/ --out_dir <dir> # By default we output on tensorboard, but if you change it make sure to change this also
```

**Create figures**

To create the figures that we have in our report you can either use your own run-logs or download our runs from [google-cloud](https://drive.google.com/file/d/1fMa3u94tJV4WGxNtBagzc-z3GqEonu_M/view?usp=sharing)
```bash
python cmd/create_figs.py --fid_csv <path> --loss_csv <path> --figs_output <dir>
```
