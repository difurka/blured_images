""" Проверка модели: вычисление roc_auc. """

import os
from collections.abc import Callable

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from src.model import get_model
from src.prepare_data import get_datasets
from src.utils import prediction


def validate(percents: float, device: Callable):
    """
    Функция валидации полученной модели.

    Arg:
        percents: какой процент валидационного датасета участвует
        device: "cpu" или "cuda"
    """

    _, val_dataset, _ = get_datasets()
    idxs = list(map(int, np.random.uniform(0,650, int(percents/100.0 * 650)))) # индексы  20 рандомных изображений
    imgs = [val_dataset[id][0].unsqueeze(0) for id in idxs] # изображения

    model = get_model().to(device)
    path_of_best_model = './outs/best_model.pth'
    if os.path.exists( path_of_best_model):
        print("Load best model")
        model.load_state_dict(torch.load( path_of_best_model))
    # вероятности предсказаний к определенному классу
    probs_ims = prediction(model, imgs, device)

    y_pred = np.argmax(probs_ims,-1)

    actual_labels = [val_dataset[id][1] for id in idxs]
    preds_class = [i for i in y_pred]

    roc_auc = roc_auc_score(actual_labels, preds_class, average='weighted')
    print(roc_auc)

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    validate(20, device)