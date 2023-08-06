"""Подбор гиперпараметров с помощью sweep W&B."""

import configs.sweep_config as cfg
import torch
import torch.nn as nn
import wandb
from src.model import get_model
from src.utils import model_learning


def sweep_func():
    """Подбор гиперпараметров с помощью sweep W&B."""
    wandb.init()
    wandb.init(project="pipeline")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=wandb.config.lr)
    criterion = nn.CrossEntropyLoss()
    model_learning(model, optimizer, criterion, epochs=wandb.config.epochs,
                    batch_size=wandb.config.batch_size, device=device)

if __name__ == '__main__':
    sweep_id = wandb.sweep(sweep=cfg.sweep_configuration, project='pipeline_competition')
    wandb.agent(sweep_id, function=sweep_func, count=1)