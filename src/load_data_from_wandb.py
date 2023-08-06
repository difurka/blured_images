import os
import zipfile

import wandb


def load_from_WB():
    """ Загрузка датасета с сайта W&B. """

    DIR = './data/'
    DIR_ZIP = './artifacts/my-dataset:v0/'

    run = wandb.init(project="pipeline_competition")
    artifact = run.use_artifact('balakinakate2022/pipeline_competition/my-dataset:v0', type='dataset')
    artifact.download()

    if (os.path.exists(DIR) == False):
        os.mkdir(DIR)
    with zipfile.ZipFile(DIR_ZIP+"shift-cv-winter-2023.zip", 'r') as zip_ref:
        zip_ref.extractall(DIR)

if __name__ == '__main__':
    load_from_WB()