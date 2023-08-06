"""Functions for training and validation."""

from collections.abc import Callable
from typing import Tuple

import numpy as np
import os
import random
import torch
import torch.nn as nn
import wandb
from src.prepare_data import get_datasets
from torch.utils.data import DataLoader
from tqdm import tqdm


def seed_everything(seed: int):
    """
    Make default settings for random values.

    Args:
        seed (int): seed for random
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True 
    # будет работать - если граф вычислений не будет меняться во время обучения
    torch.backends.cudnn.benchmark = True  # оптимизации


def model_learning(    
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: Callable,
    epochs: int,
    batch_size: int,
    device: Callable,
    ):
    """
    Make learning of model for epochs.

    Args:
        
        model: current model
        optimizer: optimizer for this learning
        criterion: loss function for this learning
        epochs: number of epochs
        batch_size: size of batch
        device: set 'cpu' or 'cuda'

    Returns: 
        dicts with losses and accuracies
    """

    train_dataset, val_dataset, _ = get_datasets()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    best_val_loss = 1
    best_val_acc = 0
    log_template = "\nEpoch {ep:03d} train_loss: {t_loss:0.4f} \
    val_loss {v_loss:0.4f} train_acc {t_acc:0.4f} val_acc {v_acc:0.4f}"
    with tqdm(desc="epoch", total=epochs) as pbar_outer:
        for epoch in range(epochs):
            train_loss, train_acc = fit_epoch(model, train_loader, criterion, optimizer, device)
            print("loss", train_loss)
            val_loss, val_acc = eval_epoch(model, val_loader, criterion, device)
            # если loss и acc на val, улучшили показатели, сохраняем модель,
            # для будущих предсказаний
            if best_val_loss >= val_loss and best_val_acc <= val_acc:
                if (os.path.exists('./outs') == False):
                    os.mkdir('./outs')
                best_val_loss = val_loss
                best_val_acc = val_acc
                torch.save(model.state_dict(), './outs/best_model.pth')
                print(f"\n\nSave model's completed on {epoch+1} epoch's")

            wandb.log({"train_loss": train_loss, "train_acc": train_acc, 
                       "val_loss": val_loss, "val_acc": val_acc, 
                       "epoch": epoch})
            pbar_outer.update(1)
            tqdm.write(log_template.format(ep=epoch+1, t_loss=train_loss,\
                                           v_loss=val_loss, t_acc=train_acc, v_acc=val_acc))
               


def fit_epoch(
        model: nn.Module,
        train_loader: DataLoader,
        criterion: Callable,
        optimizer: torch.optim.Optimizer,
        device: Callable
    ) -> Tuple[float, float]:
    """
    Проводим обучение на одном баче.

    Args:
        model (nn.Module): используемая модель
        train_loader (DataLoader): даталоадер для обучения
        criterion (Callable): функция потерь
        optimizer (torch.optim.Optimizer): оптимайзер
        device (Callable): 'cpu' или 'cuda'

    Returns:
        Tuple[float, float]: loss и accuracy
    """
    running_loss = 0.0
    running_corrects = 0
    processed_data = 0
  
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        preds = torch.argmax(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        processed_data += inputs.size(0)
    train_loss = running_loss / processed_data
    train_acc = running_corrects.cpu().numpy() / processed_data
    return train_loss, train_acc


def eval_epoch(
        model: nn.Module, 
        val_loader: DataLoader, 
        criterion: Callable, 
        device: Callable
    ) -> Tuple[float, float]:
    """
    Проводим оценивание на одном баче.
    Args:
        model (nn.Module): используемая модель
        val_loader (DataLoader): даталоадер для оценивания
        criterion (Callable): функция потерь
        device (Callable): 'cpu' или 'cuda'

    Returns:
        Tuple[float, float]: loss и accuracy
    """
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    processed_size = 0

    for inputs, labels in val_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            preds = torch.argmax(outputs, 1)

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        processed_size += inputs.size(0)
    val_loss = running_loss / processed_size
    val_acc = running_corrects.double() / processed_size
    return val_loss, val_acc


def prediction(
        model: nn.Module, 
        test_loader: DataLoader, 
        device: Callable
    ) -> np.ndarray:
    """
    Определение класса для набора из test_loader.
    Args:
        model: модель для вычислений
        test_loader: набор для определения класса
        device: "cpu" или "cuda"

    Returns:
        numpy.ndarray: _description_
    """
    with torch.no_grad():
        logits = []
    
        for inputs in test_loader:
            inputs = inputs.to(device)
            model.eval()
            outputs = model(inputs).cpu()
            logits.append(outputs)
            
    probs = nn.functional.softmax(torch.cat(logits), dim=-1).numpy()
    return probs