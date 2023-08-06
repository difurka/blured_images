""" Составление файла submission.csv по обученной модели. """

import os
from collections.abc import Callable

import numpy as np
import pandas as pd
import torch
from src.model import get_model
from src.prepare_data import get_datasets
from src.utils import prediction
from torch.utils.data import DataLoader


def predict(device: Callable):
    """
    Вычисление вероятности размытости для test_loader.

    Arg:
        device: "cpu" или "cuda"
    """
    model = get_model().to(device)
    path_of_best_model = './outs/best_model.pth'
    if os.path.exists( path_of_best_model):
        print("Load best model")
        model.load_state_dict(torch.load( path_of_best_model))
    _, _, test_dataset = get_datasets()
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=32)
    probs_ims = prediction(model, test_loader, device)
    preds_inception_v3 = [np.round(prob[1], 1) for prob in probs_ims]
    test_filenames = [path.name for path in test_dataset.files]
    submit = pd.DataFrame({'filename': test_filenames, 'blur': preds_inception_v3})
    submit.to_csv('./outs/submission.csv', index=False) 

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    predict(device)