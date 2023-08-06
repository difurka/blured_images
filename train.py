""" Обучение модели. """

import argparse
from collections.abc import Callable

import configs.config as config
import src.utils as utils
import torch
import torch.nn as nn
import wandb
from src.model import get_model


def  train_model( 
        config: dict,
        device: Callable = torch.device('cpu'),
    ):
    """
    Build all together: initialize the model,
    optimizer and loss function.

    Args:
        batch_size (int): set batch size
        epochs (int): number of epochs
        lr (float): learning rate
        seed (int): seed for randoms
        device : set "cpu" or "cuda"
    """

    wandb.login()
    with wandb.init(project="pipeline_competition",config=config):
        config = wandb.config
        utils.seed_everything(config.seed)
        model = get_model().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
        criterion = nn.CrossEntropyLoss()
        utils.model_learning(model, optimizer, criterion, epochs=config.epochs,
                       batch_size=config.batch_size, device=device)

def train():
    """ Определение гиперпараметров, запуск обучения. """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, help='total training epochs')
    parser.add_argument('--batch_size', type=int, help='total batch size')
    parser.add_argument('--seed', type=int, help='global training seed')
    parser.add_argument('--lr', type=float, help='global learning rate')
    args = parser.parse_args()

    config_for_training = dict(
        architecture = config.architecture,  
        dataset = config.dataset,
        epochs = args.epochs if (args.epochs) else config.epochs,
        batch_size = args.batch_size if (args.batch_size) else config.batch_size,
        lr= args.lr if (args.lr) else config.lr,
        seed = args.seed if (args.seed) else config.seed
    )
    train_model(config_for_training, device=device)

if __name__ == '__main__':
    train()
