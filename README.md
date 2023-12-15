# AS HW5
#### Implemented by: Pistsov Georgiy 202

You can find report here: [wandb report](wow.ru)

## Installation guide

Current repository is for Linux

(optional, not recommended) if you are trying to install it on macos run following before install:
```shell
make switch_to_macos
```

Then you run:

```shell
make install
```

## Download checkpoints:

```shell
make download_checkpoints
```
Both checkpoint for s1 and s3 sinc filters will be in default_test_model/s1/ and default_test_model/s3/ respectively

## Train model:

```shell
make train
```
Config for training you can find in src/config.json


## Test something:

The audios to test should be in "test_data_folder/"

To test on model with s1:

```shell
make test_s1
```

To test on model with s3:

```shell
make test_s3
```

## Run any other python script:

If you want to run any other custom python script, you can just start it with "poetry run"
For example:

Instead of:

```shell
python train.py -r default_test_model/model_best.pth
```

You can use:

```shell
poetry run python train.py -r default_test_model/model_best.pth
```

## How to train my models

For s1: 

```shell
poetry run python train.py -c src/configs/config_s1.json
```

For s3: 

```shell
poetry run python train.py -c src/configs/config_s3.json
```

## Credits

This repository is based on a heavily modified fork
of [pytorch-template](https://github.com/victoresque/pytorch-template) repository.